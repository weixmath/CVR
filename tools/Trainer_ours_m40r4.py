import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from tensorboardX import SummaryWriter
import math
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from otk.utils import normalize
class ModelNetTrainer(object):
    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, \
                 model_name, log_dir, num_views=12):
        self.optimizer = optimizer
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.model_name = model_name
        self.log_dir = log_dir
        self.num_views = num_views
        self.model.cuda()
        if self.log_dir is not None:
            self.writer = SummaryWriter(log_dir)
    def train(self, n_epochs):
        best_acc = 0
        i_acc = 0
        self.model.train()
        scalar = GradScaler()
        # if self.model_name =='view-gcn':
            # self.model.weight_init(self.train_loader)
        scheduler = CosineAnnealingLR(self.optimizer,T_max=((len(self.train_loader.dataset.rand_view_num)//20)*60),eta_min=1e-5)
        for epoch in range(n_epochs):
            rand_idx = np.random.permutation(int(len(self.train_loader.dataset.rand_view_num)))
            filepaths_new = []
            rand_view_num_new = []
            view_coord_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.train_loader.dataset.filepaths[
                                        rand_idx[i] * self.num_views:(rand_idx[i] + 1) * self.num_views])
                rand_view_num_new.append(self.train_loader.dataset.rand_view_num[rand_idx[i]])
                view_coord_new.append(self.train_loader.dataset.view_coord[rand_idx[i]])
            self.train_loader.dataset.filepaths = filepaths_new
            self.train_loader.dataset.rand_view_num = np.array(rand_view_num_new)
            self.train_loader.dataset.view_coord= np.array(view_coord_new)
            # plot learning rate
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.writer.add_scalar('params/lr', lr, epoch)
            # train one epoch
            out_data = None
            in_data = None
            for i, data in enumerate(self.train_loader):
                if epoch == 0:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr * ((i + 1) / (len(self.train_loader.dataset.rand_view_num) // 20))
                if self.model_name == 'svcnn':
                    in_data = Variable(data[1].cuda())
                else:
                    view_coord = data[3].cuda()
                    rand_view_num = data[2].cuda()
                    N, V, C, H, W = data[1].size()
                    in_data = Variable(data[1]).cuda()
                    in_data_new = []
                    for ii in range(N):
                        in_data_new.append(in_data[ii,0:rand_view_num[ii]])
                    in_data = torch.cat(in_data_new, 0)

                target = Variable(data[0]).cuda().long()
                target2 = Variable(data[0]).cuda().long()
                vert = [[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1], [-1, 1, 1], [-1, 1, -1], [-1, -1, 1],
                             [-1, -1, -1]]
                # vert = [[1,1],[1,-1],[-1,1],[-1,-1]]
                # vert = [[1,1,1,1],
                #         [-1,1,1,1],[1,-1,1,1],[1,1,-1,1],[1,1,1,-1],
                #         [-1,-1,1,1],[-1,1,-1,1],[-1,1,1,-1],[1,-1,-1,1],[1,-1,1,-1],[1,1,-1,-1],
                #         [-1,-1,-1,1],[-1,-1,1,-1],[-1,1,-1,-1],[1,-1,-1,-1],
                #         [-1,-1,-1,-1]]
                vert = torch.Tensor(vert).cuda()
                # target_ = target.unsqueeze(1).repeat(1, 2*(10+5)).view(-1)
                self.optimizer.zero_grad()
                with autocast():
                    if self.model_name == 'svcnn':
                        out_data = self.model(in_data)
                        loss = self.loss_fn(out_data, target)
                    else:
                        out_data,cos_sim,cos_sim2,pos= self.model(in_data,rand_view_num,N)
                        cos_loss = cos_sim[torch.where(cos_sim>-1)].mean()
                        cos_loss2 = cos_sim2[torch.where(cos_sim2>-1)].mean()
                    	# part_loss = self.loss_fn(part.reshape(-1,40),target2)
                        pos_loss = torch.norm(normalize(vert)-normalize(pos),p=2,dim=-1).mean()

                    loss = self.loss_fn(out_data, target)+0.1*pos_loss

                self.writer.add_scalar('train/train_loss', loss, i_acc + i + 1)

                pred = torch.max(out_data, 1)[1]
                results = pred == target
                correct_points = torch.sum(results.long())

                acc = correct_points.float() / results.size()[0]
                self.writer.add_scalar('train/train_overall_acc', acc, i_acc + i + 1)
                # print('lr = ', str(param_group['lr']))
                scalar.scale(loss).backward()
                scalar.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 20)
                scalar.step(self.optimizer)
                scalar.update()
                #loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 20)
                #self.optimizer.step()
                if epoch>0:
                    scheduler.step()
                # self.optimizer.zero_grad()
                log_str = 'epoch %d, step %d: train_loss %.3f;cos_loss %.3f;pos_loss%.3f; train_acc %.3f' % (epoch + 1, i + 1, loss,cos_loss2,pos_loss, acc)
                if (i + 1) % 1 == 0:
                    print(log_str)
            i_acc += i
            # evaluation
            if (epoch + 1) % 1 == 0:
                with torch.no_grad():
                    loss, val_overall_acc, val_mean_class_acc = self.update_validation_accuracy(epoch)
                self.writer.add_scalar('val/val_mean_class_acc', val_mean_class_acc, epoch + 1)
                self.writer.add_scalar('val/val_overall_acc', val_overall_acc, epoch + 1)
                self.writer.add_scalar('val/val_loss', loss, epoch + 1)
                # self.model.save(self.log_dir, epoch)
            # save best model
                if val_overall_acc > best_acc:
                    best_acc = val_overall_acc
                print('best_acc', best_acc)
        # export scalar data to JSON for external processing
        self.writer.export_scalars_to_json(self.log_dir + "/all_scalars.json")
        self.writer.close()
    def update_validation_accuracy(self, epoch):
        all_correct_points = 0
        all_points = 0
        count = 0
        wrong_class = np.zeros(40)
        samples_class = np.zeros(40)
        all_loss = 0
        self.model.eval()
        for _, data in enumerate(self.val_loader, 0):
            if self.model_name == 'svcnn':
                in_data = Variable(data[1].cuda())
            else:
                view_coord = data[3].cuda()
                rand_view_num = data[2].cuda()
                N, V, C, H, W = data[1].size()
                in_data = Variable(data[1]).cuda()
                in_data_new = []
                for ii in range(N):
                    in_data_new.append(in_data[ii, 0:rand_view_num[ii]])
                in_data = torch.cat(in_data_new, 0)
            target = Variable(data[0]).cuda()
            #
            if self.model_name == 'svcnn':
                out_data = self.model(in_data)
            else:
                out_data,cos_sim,cos_sim2,pos= self.model(in_data, rand_view_num, N)
                # cos_loss = cos_sim[torch.where(cos_sim > 0)].mean()
            pred = torch.max(out_data, 1)[1]
            all_loss += self.loss_fn(out_data, target).cpu().data.numpy()
            results = pred == target

            for i in range(results.size()[0]):
                if not bool(results[i].cpu().data.numpy()):
                    wrong_class[target.cpu().data.numpy().astype('int')[i]] += 1
                samples_class[target.cpu().data.numpy().astype('int')[i]] += 1
            correct_points = torch.sum(results.long())

            all_correct_points += correct_points
            all_points += results.size()[0]

        print('Total # of test models: ', all_points)
        class_acc = (samples_class - wrong_class) / samples_class
        val_mean_class_acc = np.mean(class_acc)
        acc = all_correct_points.float() / all_points
        val_overall_acc = acc.cpu().data.numpy()
        loss = all_loss / len(self.val_loader)


        print('val mean class acc. : ', val_mean_class_acc)
        print('val overall acc. : ', val_overall_acc)
        print('val loss : ', loss)
        # print('cos loss : ', cos_loss)
        print(class_acc)
        self.model.train()

        return loss, val_overall_acc, val_mean_class_acc
