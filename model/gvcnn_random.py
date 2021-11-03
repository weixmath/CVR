import torch
import torchvision
import torch.nn as nn
# from torchsummary import summary


def fc_bn_block(input, output):
    return nn.Sequential(
        nn.Linear(input, output),
        # nn.BatchNorm1d(output),
        nn.ReLU(inplace=True))


def cal_scores(scores):
    n = len(scores)
    s = 0
    for score in scores:
        s += torch.ceil(score * n)
    s /= n
    return s

def group_fusion(view_group, weight_group):
    shape_des = map(lambda a, b: a * b, view_group, weight_group)
    shape_des = sum(shape_des) / sum(weight_group)
    return shape_des

def group_pooling(final_views, views_score, group_num):
    interval = 1.0 / group_num

    def onebatch_grouping(onebatch_views, onebatch_scores):
        viewgroup_onebatch = [[] for i in range(group_num)]
        scoregroup_onebatch = [[] for i in range(group_num)]

        for i in range(group_num):
            left = i * interval
            right = (i + 1) * interval
            for j, score in enumerate(onebatch_scores):
                if left <= score < right:
                    viewgroup_onebatch[i].append(onebatch_views[j])
                    scoregroup_onebatch[i].append(score)
                else:
                    pass
        # print(len(scoregroup_onebatch))
        view_group = [sum(views) / len(views) for views in viewgroup_onebatch if len(views) > 0]
        weight_group = [cal_scores(scores) for scores in scoregroup_onebatch if len(scores) > 0]
        onebatch_shape_des = group_fusion(view_group, weight_group)
        return onebatch_shape_des

    shape_descriptors = []
    for (onebatch_views, onebatch_scores) in zip(final_views, views_score):
        shape_descriptors.append(onebatch_grouping(onebatch_views, onebatch_scores))
    shape_descriptor = torch.stack(shape_descriptors, 0)
    # shape_descriptor: [B, 1024]
    return shape_descriptor


class GVCNN(nn.Module):
    def __init__(self, num_classes=40, group_num=8, model_name='GOOGLENET', pretrained=True):
        super(GVCNN, self).__init__()

        self.num_classes = num_classes
        self.group_num = group_num

        if model_name == 'GOOGLENET':
            base_model = torchvision.models.googlenet(pretrained=pretrained)

        self.FCN = nn.Sequential(*list(base_model.children())[:6])
        self.CNN = nn.Sequential(*list(base_model.children())[:-2])
        self.FC = nn.Sequential(fc_bn_block(256 * 28 * 28, 256),
                                fc_bn_block(256, 1))
        self.fc_block_1 = fc_bn_block(1024, 512)
        self.drop_1 = nn.Dropout(0.5)
        self.fc_block_2 = fc_bn_block(512, 256)
        self.drop_2 = nn.Dropout(0.5)
        self.linear = nn.Linear(256, self.num_classes)

    def forward(self,views,rand_view_num,N):
        '''
        params views: B V C H W (B 12 3 224 224)
        return result: B num_classes
        '''
        # print(views.size())
        # views = views.cpu()
        # batch_size, num_views, channel, image_size = views.size(0), views.size(1), views.size(2), views.size(3)
        #
        # views = views.view(batch_size * num_views, channel, image_size, image_size)
        raw_views = self.FCN(views)
        # print(raw_views.size())
        # raw_views: [B*V 256 28 28]
        final_views = self.CNN(views)
        # final_views: [B*V 1024 1 1]
        final_views = final_views.squeeze()
        m = 0
        pred_final = []
        for i in range(N):
            raw_views0 = raw_views[m:m+rand_view_num[i],:,:,:]
            final_views0 = final_views[m:m+rand_view_num[i],:].unsqueeze(0)
            m = m + rand_view_num[i]
            views_score = self.FC(raw_views0.view(rand_view_num[i],-1)).unsqueeze(0)
            views_score = torch.sigmoid(torch.tanh(torch.abs(views_score)))
            # views_score = views_score.view(batch_size, num_views, -1)
            # views_score: [B V]
            shape_descriptor = group_pooling(final_views0, views_score, self.group_num)
            # print(shape_descriptor.size())
            out = self.fc_block_1(shape_descriptor)
            out = self.drop_1(out)
            out = self.fc_block_2(out)
            viewcnn_feature = out
            out = self.drop_2(out)
            pred = self.linear(out)
            pred_final.append(pred)
        pred_final = torch.cat(pred_final,0)
        return pred_final