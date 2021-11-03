import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from .Model import Model
mean = torch.tensor([0.485, 0.456, 0.406],dtype=torch.float, requires_grad=False)
std = torch.tensor([0.229, 0.224, 0.225],dtype=torch.float, requires_grad=False)
def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1) - 1,
                                                                 -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

class SVCNN(Model):
    def __init__(self, name, nclasses=40, pretraining=True, cnn_name='resnet18'):
        super(SVCNN, self).__init__(name)
        if nclasses == 40:
            self.classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                               'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                               'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                               'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                               'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
        elif nclasses==15:
            self.classnames = ['bag', 'bed', 'bin', 'box', 'cabinet', 'chair', 'desk', 'display'
                , 'door', 'pillow', 'shelf', 'sink', 'sofa', 'table', 'toilet']
        else:
            self.classnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11',
                               '12', '13', '14', '15', '16', '17', '18', '19', '20', '21',
                               '22', '23', '24', '25', '26', '27', '28', '29', '30', '31',
                               '32', '33', '34', '35', '36', '37', '38', '39', '40', '41',
                               '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54']

        self.nclasses = nclasses
        self.pretraining = pretraining
        self.cnn_name = cnn_name
        self.use_resnet = cnn_name.startswith('resnet')
        self.mean = torch.tensor([0.485, 0.456, 0.406],dtype=torch.float, requires_grad=False)
        self.std = torch.tensor([0.229, 0.224, 0.225],dtype=torch.float, requires_grad=False)

        if self.use_resnet:
            if self.cnn_name == 'resnet18':
                self.net = models.resnet18(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512, self.nclasses)
            elif self.cnn_name == 'resnet34':
                self.net = models.resnet34(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512, self.nclasses)
            elif self.cnn_name == 'resnet50':
                self.net = models.resnet50(pretrained=self.pretraining)
                self.net.fc = nn.Linear(2048, self.nclasses)
        else:
            if self.cnn_name == 'alexnet':
                self.net_1 = models.alexnet(pretrained=self.pretraining).features
                self.net_2 = models.alexnet(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg11':
                self.net_1 = models.vgg11_bn(pretrained=self.pretraining).features
                self.net_2 = models.vgg11_bn(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg16':
                self.net_1 = models.vgg16(pretrained=self.pretraining).features
                self.net_2 = models.vgg16(pretrained=self.pretraining).classifier

            self.net_2._modules['6'] = nn.Linear(4096, self.nclasses)

    def forward(self, x):
        if self.use_resnet:
            return self.net(x)
        else:
            y = self.net_1(x)
            return self.net_2(y.view(y.shape[0], -1))

class view_GCN(Model):
    def __init__(self,name, model, nclasses=40, cnn_name='resnet18', num_views=20):
        super(view_GCN,self).__init__(name)
        if nclasses == 40:
            self.classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                               'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                               'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                               'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                               'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
        elif nclasses==15:
            self.classnames = ['bag', 'bed', 'bin', 'box', 'cabinet', 'chair', 'desk', 'display'
                , 'door', 'pillow', 'shelf', 'sink', 'sofa', 'table', 'toilet']
        else:
            self.classnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11',
                               '12', '13', '14', '15', '16', '17', '18', '19', '20', '21',
                               '22', '23', '24', '25', '26', '27', '28', '29', '30', '31',
                               '32', '33', '34', '35', '36', '37', '38', '39', '40', '41',
                               '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54']
        self.nclasses = nclasses
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float, requires_grad=False)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float, requires_grad=False)
        self.use_resnet = cnn_name.startswith('resnet')
        if self.use_resnet:
            self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
            self.net_2 = model.net.fc
        else:
            self.net_1 = model.net_1
            self.net_2 = model.net_2
        self.num_views = num_views
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight)
    def forward(self,x):
        y = self.net_1(x)
        y = y.view(
            (int(x.shape[0] / self.num_views), self.num_views, y.shape[-3], y.shape[-2], y.shape[-1]))  # (8,12,512,7,7)
        return self.net_2(torch.max(y, 1)[0].view(y.shape[0], -1))