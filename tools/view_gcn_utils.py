import torch
import torch.nn as nn
import torch.nn.functional as Functional
def position_embedding(input, d_model):
    input = input.view(-1, 1)
    dim = torch.arange(d_model // 2, dtype=torch.float32, device=input.device).view(1, -1)
    sin = torch.sin(input / 10000 ** (2 * dim / d_model))
    cos = torch.cos(input / 10000 ** (2 * dim / d_model))

    out = torch.zeros((input.shape[0], d_model), device=input.device)
    out[:, ::2] = sin
    out[:, 1::2] = cos
    return out

def sinusoid_encoding_table(max_len, d_model, padding_idx=None):
    pos = torch.arange(max_len, dtype=torch.float32)
    out = position_embedding(pos, d_model)

    if padding_idx is not None:
        out[padding_idx] = 0
    return out
def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist
def my_pad_sequence(sequences,view_num,N,max_length, padding_value=0):
    C = sequences.size(-1)
    out_tensor = torch.empty(N,max_length,C).fill_(padding_value).cuda()
    count = 0
    for i in range(0,N):
        out_tensor[i,:view_num[i],:] = sequences[count:count+view_num[i],:]
        count = count + view_num[i]
    return out_tensor

def generate_mask(view_num,B,max_len):
    seq_len = torch.arange(0,max_len).unsqueeze(0).repeat(B,1).cuda()
    sss = view_num.reshape(B,-1).repeat(1,max_len)
    return sss>seq_len
def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance = distance.double()
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def knn(nsample, xyz, new_xyz):
    dist = square_distance(xyz, new_xyz)
    id = torch.topk(dist,k=nsample,dim=1,largest=False)[1]
    id = torch.transpose(id, 1, 2)
    return id
class View_selector(nn.Module):
    def __init__(self, n_views, sampled_view):
        super(View_selector, self).__init__()
        self.n_views = n_views
        self.s_views = sampled_view
        self.cls = nn.Sequential(
            nn.Linear(512*self.s_views, 256*self.s_views),
            nn.LeakyReLU(0.2),
            nn.Linear(256*self.s_views, 40*self.s_views))
    def forward(self,F,vertices,k):
        id = farthest_point_sample(vertices,self.s_views)
        vertices1 = index_points(vertices,id)
        id_knn = knn(k,vertices,vertices1)
        F = index_points(F,id_knn)
        vertices = index_points(vertices,id_knn)
        F1 = F.transpose(1,2).reshape(F.shape[0],k,self.s_views*F.shape[-1])
        F_score = self.cls(F1).reshape(F.shape[0],k,self.s_views,40).transpose(1,2)
        F1_ = Functional.softmax(F_score,-3)
        F1_ = torch.max(F1_,-1)[0]
        F1_id = torch.argmax(F1_,-1)
        F1_id = Functional.one_hot(F1_id,4).float()
        F1_id_v = F1_id.unsqueeze(-1).repeat(1,1,1,3)
        F1_id_F = F1_id.unsqueeze(-1).repeat(1, 1, 1, 512)
        F_new = torch.mul(F1_id_F,F).sum(-2)
        vertices_new = torch.mul(F1_id_v,vertices).sum(-2)
        return F_new,F_score,vertices_new

class View_selector_RI(nn.Module):
    def __init__(self, n_views, sampled_view):
        super(View_selector_RI, self).__init__()
        self.n_views = n_views
        self.s_views = sampled_view
        self.cls = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 40))
    def forward(self,F0,vertices0,k):
        score_init = self.cls(F0)
        score= Functional.softmax(score_init,-1)
        score = torch.max(score,-1)[0]
        id0 = torch.argmax(score,-1)
        vertices = vertices0.clone()
        F = F0.clone()
        vertices[torch.arange(0,F.shape[0]),torch.zeros_like(id0),:],vertices[torch.arange(0,F.shape[0]),id0,:] = vertices0[torch.arange(0,F.shape[0]),id0,:],vertices0[torch.arange(0,F.shape[0]),torch.zeros_like(id0),:]
        F[torch.arange(0,F.shape[0]),torch.zeros_like(id0),:],F[torch.arange(0,F.shape[0]),id0,:] = F0[torch.arange(0,F.shape[0]),id0,:],F0[torch.arange(0,F.shape[0]),torch.zeros_like(id0),:]
        vertices.contiguous()
        F.contiguous()
        id = farthest_point_sample(vertices,self.s_views)
        vertices1 = index_points(vertices,id)
        id_knn = knn(k,vertices,vertices1)
        F = index_points(F,id_knn)
        vertices = index_points(vertices,id_knn)
        F1 = F.transpose(1,2).reshape(F.shape[0],k,self.s_views,F.shape[-1])
        F_score = self.cls(F1).reshape(F.shape[0],k,self.s_views,40).transpose(1,2)
        F1_ = Functional.softmax(F_score,-3)
        F1_ = torch.max(F1_,-1)[0]
        F1_id = torch.argmax(F1_,-1)
        F1_id = Functional.one_hot(F1_id,4).float()
        F1_id_v = F1_id.unsqueeze(-1).repeat(1,1,1,3)
        F1_id_F = F1_id.unsqueeze(-1).repeat(1, 1, 1, 512)
        F_new = torch.mul(F1_id_F,F).sum(-2)
        vertices_new = torch.mul(F1_id_v,vertices).sum(-2)
        return F_new,score_init,vertices_new

class View_selector_RI_CAM(nn.Module):
    def __init__(self, n_views, sampled_view):
        super(View_selector_RI_CAM, self).__init__()
        self.n_views = n_views
        self.s_views = sampled_view
        self.cls = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 40))
    def forward(self,F0,vertices0,k):
        B,N,C = F0.size()
        pooled_F = torch.sum(F0,1)/self.n_views
        pred = self.cls(pooled_F)
        pred = torch.max(pred,-1)[1]
        score = self.cls(F0)
        score_F = score[torch.arange(B),:,pred]
        _,idx = torch.topk(score_F,self.s_views,dim=-1)
        F_new = index_points(F0,idx)
        vertices_new = index_points(vertices0,idx)
        return F_new,score,vertices_new
class View_selector_RI_CAM_softmax(nn.Module):
    def __init__(self, n_views, sampled_view):
        super(View_selector_RI_CAM_softmax, self).__init__()
        self.n_views = n_views
        self.s_views = sampled_view
        self.cls = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 40))
    def forward(self,F0,vertices0,k):
        B,N,C = F0.size()
        pooled_F = torch.sum(F0,1)/self.n_views
        pred = self.cls(pooled_F)
        pred = torch.max(pred,-1)[1]
        score = torch.softmax(self.cls(F0),-1)
        score_F = score[torch.arange(B),:,pred]
        _,idx = torch.topk(score_F,self.s_views,dim=-1)
        F_new = index_points(F0,idx)
        vertices_new = index_points(vertices0,idx)
        return F_new,score,vertices_new
class View_selector_Critical(nn.Module):
    def __init__(self,n_views,sampled_view):
        super(View_selector_Critical,self).__init__()
        self.n_views = n_views
        self.s_views = sampled_view
    def forward(self,F0,vertices0,k):
        B,N,C = F0.size()
        F_max = torch.max(F0,1)[1]
        idx = []
        for i in range(B):
            out,numb = torch.unique(F_max[i], return_inverse=False, return_counts=True, dim=-1)
            out = out.reshape(1,1,-1)
            if out.size()[-1]<self.s_views:
                m = torch.nn.ReplicationPad1d([self.s_views-out.size()[-1],0])
                out2 = m(out.float()).long()
            else:
                top_id = torch.topk(numb,dim=-1,k=self.s_views)[1]
                out2 = out[:,:,top_id]
            idx.append(out2.squeeze(1))
        idx = torch.cat(idx,0)
        F_new = index_points(F0,idx)
        vertices_new = index_points(vertices0,idx)
        return F_new,vertices_new
class View_selector_Critical2(nn.Module):
    def __init__(self,n_views,sampled_view):
        super(View_selector_Critical2,self).__init__()
        self.n_views = n_views
        self.s_views = sampled_view
    def forward(self,F0,vertices0,k):
        B,N,C = F0.size()
        count =torch.argsort(F0,dim=1,descending=False)
        count = torch.sum(count,-1)
        idx = torch.topk(count,dim=-1,k=self.s_views)[1]
        F_new = index_points(F0,idx)
        vertices_new = index_points(vertices0,idx)
        return F_new,vertices_new
class KNN_dist(nn.Module):
    def __init__(self,k):
        super(KNN_dist, self).__init__()
        self.R = nn.Sequential(
            nn.Linear(10,10),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(10,10),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(10,1),
        )
        self.k=k
    def forward(self,F,vertices):
        id = knn(self.k, vertices, vertices)
        F = index_points(F,id)
        v = index_points(vertices,id)
        v_0 = v[:,:,0,:].unsqueeze(-2).repeat(1,1,self.k,1)
        v_F = torch.cat((v_0, v, v_0-v,torch.norm(v_0-v,dim=-1,p=2).unsqueeze(-1)),-1)
        v_F = self.R(v_F)
        F = torch.mul(v_F, F)
        F = torch.sum(F,-2)
        return F
class view_transformer_multi_scale(nn.Module):
    def __init__(self,k,n_view,d_model=512, d_k=256, d_v=256, h=4, d_ff=2048, dropout=0,use_mask=True):
        super(view_transformer_multi_scale,self).__init__()
        self.Encoder = EncoderLayer(d_model=d_model, d_k=d_k, d_v=d_v, h=h, d_ff=d_ff, dropout=dropout)
        self.Encoder2 = EncoderLayer(d_model=d_model, d_k=d_k, d_v=d_v, h=h, d_ff=d_ff, dropout=dropout)
        self.Encoder3 = EncoderLayer(d_model=d_model, d_k=d_k, d_v=d_v, h=h, d_ff=d_ff, dropout=dropout)
        self.k1,self.k2 = k[0],k[1]
        self.head = h
        self.n_view = n_view
        self.use_mask = use_mask
        self.fusion = nn.Sequential(
            nn.Conv1d(512*2,512,1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Conv1d(512,512,1)
        )
    def forward(self,vertices,F):
        B, N, C = F.size()
        id1 = knn(self.k1, vertices, vertices)
        id1 = torch.nn.functional.one_hot(id1,self.n_view).sum(-2).unsqueeze(1).repeat(1,self.head,1,1)
        mask1 = id1 < 1

        id2 = knn(self.k2, vertices, vertices)
        id2 = torch.nn.functional.one_hot(id2,self.n_view).sum(-2).unsqueeze(1).repeat(1,self.head,1,1)
        mask2 = id2 < 1

        # id3 = knn(self.k3, vertices, vertices)
        # id3 = torch.nn.functional.one_hot(id3,self.n_view).sum(-2).unsqueeze(1).repeat(1,self.head,1,1)
        # mask3 = id3 < 1

        F1 = self.Encoder(queries=F, keys=F, values=F, attention_mask=mask1)
        F2 = self.Encoder2(queries=F, keys=F, values=F, attention_mask=mask2)
        # F3 = self.Encoder3(queries=F, keys=F, values=F, attention_mask=mask3)
        yy = (torch.cat((F1, F2), -1)).transpose(1, 2)
        yy = (self.fusion(yy)).transpose(1, 2)
        return yy
class view_transformer_wocoord(nn.Module):
    def __init__(self,d_model=512, d_k=256, d_v=256, h=4, d_ff=2048, dropout=0,use_mask=True):
        super(view_transformer_wocoord,self).__init__()
        self.Encoder = EncoderLayer(d_model=d_model, d_k=d_k, d_v=d_v, h=h, d_ff=d_ff, dropout=dropout)
        self.head = h
        self.use_mask = use_mask
    def forward(self,F):
        B, N, C = F.size()
        F1 = self.Encoder(queries=F, keys=F, values=F, attention_mask=None)
        return F1
class view_transformer(nn.Module):
    def __init__(self,k,n_view,d_model=512, d_k=256, d_v=256, h=4, d_ff=2048, dropout=0,use_mask=True):
        super(view_transformer,self).__init__()
        self.Encoder = EncoderLayer(d_model=d_model, d_k=d_k, d_v=d_v, h=h, d_ff=d_ff, dropout=dropout)
        self.k = k
        self.head = h
        self.n_view = n_view
        self.use_mask = use_mask
    def forward(self,vertices,F):
        B, N, C = F.size()
        id = knn(self.k, vertices, vertices)
        id2 = torch.nn.functional.one_hot(id,self.n_view).sum(-2).unsqueeze(1).repeat(1,self.head,1,1)
        mask = id2 < 1
        if self.use_mask ==True:
            F1 = self.Encoder(queries=F, keys=F, values=F, attention_mask=mask)
        else:
            F1 = self.Encoder(queries=F, keys=F, values=F, attention_mask=None)
        return F1
class view_transformer2(nn.Module):
    def __init__(self,k,n_view,d_model=512, d_k=256, d_v=256, h=4, d_ff=2048, dropout=0,use_mask=True):
        super(view_transformer2,self).__init__()
        self.Encoder = EncoderLayer_bn(d_model=d_model, d_k=d_k, d_v=d_v, h=h, d_ff=d_ff, dropout=dropout)
        self.k = k
        self.head = h
        self.n_view = n_view
        self.use_mask = use_mask
    def forward(self,vertices,F):
        B, N, C = F.size()
        id = knn(self.k, vertices, vertices)
        id2 = torch.nn.functional.one_hot(id,self.n_view).sum(-2).unsqueeze(1).repeat(1,self.head,1,1)
        mask = id2 < 1
        if self.use_mask ==True:
            F1 = self.Encoder(queries=F, keys=F, values=F, attention_mask=mask)
        else:
            F1 = self.Encoder(queries=F, keys=F, values=F, attention_mask=None)
        return F1
class view_transformer_vector(nn.Module):
    def __init__(self,k,n_view,d_model=512, d_k=512, d_v=512, h=1, d_ff=2048, dropout=0,use_mask=True):
        super(view_transformer_vector,self).__init__()
        self.Encoder = EncoderLayer_vector(d_model=d_model, d_k=d_k, d_v=d_v, h=h, d_ff=d_ff, dropout=dropout)
        self.k = k
        self.head = h
        self.n_view = n_view
        self.use_mask = use_mask
    def forward(self,vertices,F):
        B, N, C = F.size()
        id = knn(self.k, vertices, vertices)
        id2 = torch.nn.functional.one_hot(id,self.n_view).sum(-2).unsqueeze(1).repeat(1,self.head,1,1)
        mask = id2 < 1
        if self.use_mask ==True:
            F1 = self.Encoder(queries=F, keys=F, values=F, attention_mask=mask)
        else:
            F1 = self.Encoder(queries=F, keys=F, values=F, attention_mask=None)
        return F1

class LocalGCN(nn.Module):
    def __init__(self,k,n_views):
        super(LocalGCN,self).__init__()
        self.conv = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.k = k
        self.n_views = n_views
        self.KNN = KNN_dist(k=self.k)
    def forward(self,F,V):
        F = self.KNN(F, V)
        F = F.view(-1, 512)
        F = self.conv(F)
        F = F.view(-1, self.n_views, 512)
        return F

class NonLocalMP(nn.Module):
    def __init__(self,n_view):
        super(NonLocalMP,self).__init__()
        self.n_view=n_view
        self.Relation = nn.Sequential(
            nn.Linear(2 * 512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.Fusion = nn.Sequential(
            nn.Linear(2 * 512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, F):
        F_i = torch.unsqueeze(F, 2)
        F_j = torch.unsqueeze(F, 1)
        F_i = F_i.repeat(1, 1, self.n_view, 1)
        F_j = F_j.repeat(1, self.n_view, 1, 1)
        M = torch.cat((F_i, F_j), 3)
        M = self.Relation(M)
        M = torch.sum(M,-2)
        F = torch.cat((F, M), 2)
        F = F.view(-1, 512 * 2)
        F = self.Fusion(F)
        F = F.view(-1, self.n_view, 512)
        return F

