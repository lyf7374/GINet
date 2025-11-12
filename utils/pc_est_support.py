import numpy as np
import torch.nn.init as init
import torch
import torch.nn as nn
import torch.nn.functional as F
from annoy import AnnoyIndex

dropout= 0.1
batch_size = 1


def normalization(planes, norm='bn', NN=False):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    if NN==True:
        m = nn.BatchNorm1d(planes)
    return m

class ConvD(nn.Module):
    def __init__(self, inplanes, planes, dropout=dropout, norm='bn', first=False, padding = 0):
        super(ConvD, self).__init__()

        self.first = first
        self.maxpool = nn.MaxPool3d(2,2,padding = padding)

        self.dropout = dropout
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(0.2,inplace=False)
        self.conv1 = nn.Conv3d(inplanes, planes, 3, 1, 1, bias=False)
        self.bn1   = normalization(planes, norm)

        self.conv2 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        self.bn2   = normalization(planes, norm)

        self.conv3 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        self.bn3   = normalization(planes, norm)

    def forward(self, x):
        if not self.first:
            x = self.maxpool(x)
        x = self.bn1(self.conv1(x))
        y = self.relu(self.bn2(self.conv2(x)))
        if self.dropout > 0:
            y = F.dropout3d(y, self.dropout)
        y = self.bn3(self.conv3(x))
        return self.relu(x + y)


class ConvU(nn.Module):
    def __init__(self, planes, norm='bn', first=False, padding = 0):
        super(ConvU, self).__init__()

        self.first = first

        if not self.first:
            self.conv1 = nn.Conv3d(2*planes, planes, 3, 1, 0, bias=False)
            self.bn1   = normalization(planes, norm)

        self.conv2 = nn.Conv3d(planes//2, planes//2, 3, 1, 1, bias=False)
        self.bn2   = normalization(planes//2, norm)

        self.conv3 = nn.Conv3d(planes, planes//2, 3, 1, 1, bias=False)
        self.bn3   = normalization(planes//2, norm)

        self.upsampling = nn.ConvTranspose3d(planes, planes//2,
                                      kernel_size=2,
                                      stride=2,padding=padding)
        
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(0.2,inplace=False)  
    def forward(self, x, prev):
        # final output is the localization layer
        y = self.upsampling(x)
        if not self.first:
#             y = self.relu(self.bn1(self.conv1(y)))
            pass
        y = self.relu(self.bn2(self.conv2(y)))
        y = torch.cat([prev, y], 1)
        y = self.relu(self.bn3(self.conv3(y)))

        return y
    

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:new_points = points[batch_indices, idx, :]
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
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):

    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N - 1
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points
    
class IG_fusion(nn.Module):
    def __init__(self, inC,outC,outS, expS):
        #  eg:（114,114,114,16）-> (1024,64),  inS = 114, inC=16, outS=16, outC=64, expS = 1024
        super(IG_fusion, self).__init__()
               
        self.conv = nn.Conv3d(inC, outC, 7, 7, 1, groups=inC, bias=True)
        self.ln = nn.LayerNorm((outC,outS, outS, outS))
        self.mlp_1 = nn.Conv3d(outC, outC, 1, 1, 0, bias=True)
        self.norm = nn.GELU()
        self.mlp_2 = nn.Conv3d(outC, outC, 1, 1, 0, bias=True)
        
        self.outC= outC
        self.flat1 = nn.Conv1d(outS*outS*outS,expS,1,1,0)
        self.flat2 = nn.Conv1d(expS,expS,1,1,0)
        
        self.map1 = nn.Conv1d(outC*2,outC,1,1,0)
        self.map2 = nn.Conv1d(outC,outC,1,1,0)    
        
    def forward(self, x, y):
        batch_size = x.shape[0]
        #   process pc 
        x = self.conv(x) 
        x = self.ln(x) 
        x = self.mlp_1(x) 
        x = self.norm(x) 
        x = self.mlp_2(x) 

        #   flatten
        # x = x.view(1,-1,self.outC)
        x = x.view(batch_size, -1, self.outC)
        x = self.flat1(x)
        x = self.norm(self.flat2(x))
        x = x.permute(0,2,1)

        #   Concat and fusion

        z = torch.cat([y,x],1)  #   (expS,outC*2)      

        z = self.map1(z)
        z = self.norm(self.map2(z))      
        
        return z
    
class I2G(nn.Module):
    def __init__(self,n_layers=4, padding_list =[0, 1, 1, 1],c=1, n=8, dropout=0.1, norm='bn',n_pc=1024):
        super(I2G, self).__init__()
        'Original PointNet++'
        self.sa1 = PointNetSetAbstraction(n_pc, 0.1, 32, 3 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        # self.conv2 = nn.Conv1d(128, 3+3+1, 1)
        self.conv2 = nn.Conv1d(128, 3, 1)
  
        
        'Image_encoder'
        self.middle_channel = 2**(n_layers)*n
        self.dropout =dropout
        self.padding_list = padding_list
        self.n_layers = n_layers  

        self.convd_list = []
        for i in range(n_layers+1):
            if i ==0:
                self.convd_list.append(ConvD(c,     n, self.dropout, norm, first=True)) 
            else:
                self.convd_list.append(ConvD(2**(i-1)*n,   2**(i)*n, self.dropout, norm ,padding=self.padding_list[i-1]))
        self.convd_list = nn.ModuleList(self.convd_list)
        
        'Image PC fusions'  
        #  Img: inC, Img: outC,  Img: outS, PC: expS 
   
        self.IG1 = IG_fusion(16,64,14,n_pc)
        self.IG2 = IG_fusion(32,128,7,256)
        self.IG3 = IG_fusion(64,256,3,64)
        self.IG4 = IG_fusion(128,512,2,16)
        
        
    def forward(self, xyz,img):

        imgs = []
        for i in range(len(self.convd_list)):
            img = self.convd_list[i](img)        
            if i>0:
                imgs.append(img) 
       
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l1_points = self.IG1(imgs[0] ,l1_points)
        
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = self.IG2(imgs[1] ,l2_points)

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points = self.IG3(imgs[2] ,l3_points)
        
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l4_points = self.IG4(imgs[3] ,l4_points)
        

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points)), inplace=True))
        x = self.conv2(x)
        x = F.softplus(x)

        # x = x.permute(0, 2, 1)
        return x, l4_points
    
    
class I2C(nn.Module):
    def __init__(self,n_layers=4, padding_list =[0, 1, 1, 1],c=1, n=8, dropout=0.1, norm='bn',n_pc=1024):
        super(I2C, self).__init__()
        'Original PointNet++'
        self.sa1 = PointNetSetAbstraction(n_pc, 0.1, 32, 3 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        # self.conv2 = nn.Conv1d(128, 3+3+1, 1)
        self.conv2 = nn.Conv1d(128, 3, 1)
  
        
        'Image_encoder'
        self.middle_channel = 2**(n_layers)*n
        self.dropout =dropout
        self.padding_list = padding_list
        self.n_layers = n_layers  

        self.convd_list = []
        for i in range(n_layers+1):
            if i ==0:
                self.convd_list.append(ConvD(c,     n, self.dropout, norm, first=True)) 
            else:
                self.convd_list.append(ConvD(2**(i-1)*n,   2**(i)*n, self.dropout, norm ,padding=self.padding_list[i-1]))
        self.convd_list = nn.ModuleList(self.convd_list)
        
        'Image PC fusions'  
        #  Img: inC, Img: outC,  Img: outS, PC: expS 
   
        self.IG1 = IG_fusion(16,64,14,n_pc)
        self.IG2 = IG_fusion(32,128,7,256)
        self.IG3 = IG_fusion(64,256,3,64)
        self.IG4 = IG_fusion(128,512,2,16)
        
        
    def forward(self, xyz,img):
        batch = xyz.shape[0]
        print('batch',batch)
        imgs = []
        for i in range(len(self.convd_list)):
            img = self.convd_list[i](img)        
            if i>0:
                imgs.append(img) 
       
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l1_points = self.IG1(imgs[0] ,l1_points)
        
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = self.IG2(imgs[1] ,l2_points)

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points = self.IG3(imgs[2] ,l3_points)
        
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l4_points = self.IG4(imgs[3] ,l4_points)
        

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        x = torch.max(l0_points, 2, keepdim=True)[0]
        x = self.conv1(x)
        if batch>1:
            x = self.bn1(x)
        x = self.drop1(F.relu(x, inplace=True))
        x = self.conv2(x)
        # Reshape the output to the desired shape [batch_size, 3]
        x = x.view(-1, 3)  # Flatten to get a single (x, y, z) per batch item
        return x
class ConditionalBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        if input.size(0) > 1:
            # Apply batch normalization when batch size > 1
            return super().forward(input)
        else:
            # Bypass batch normalization when batch size = 1
            # Need to manually apply the affine transformation (scale and shift)
            if self.affine:
                return input * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
            else:
                return input

class I2CR(nn.Module):
    def __init__(self,n_layers=4, padding_list =[0, 1, 1, 1],c=1, n=8, dropout=0.1, norm='bn',n_pc=1024):
        super(I2CR, self).__init__()
        'Original PointNet++'
        self.sa1 = PointNetSetAbstraction(n_pc, 0.1, 32, 3 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])

        # Center prediction
        self.center_pre = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            ConditionalBatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Conv1d(128, 3, 1)
        )

        # Radius prediction
        self.radius_pre = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            ConditionalBatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Conv1d(128, 1, 1)
        )

        'Image_encoder'
        self.middle_channel = 2**(n_layers)*n
        self.dropout =dropout
        self.padding_list = padding_list
        self.n_layers = n_layers  

        self.convd_list = []
        for i in range(n_layers+1):
            if i ==0:
                self.convd_list.append(ConvD(c,     n, self.dropout, norm, first=True)) 
            else:
                self.convd_list.append(ConvD(2**(i-1)*n,   2**(i)*n, self.dropout, norm ,padding=self.padding_list[i-1]))
        self.convd_list = nn.ModuleList(self.convd_list)
        
        'Image PC fusions'  
        #  Img: inC, Img: outC,  Img: outS, PC: expS 
   
        self.IG1 = IG_fusion(16,64,14,n_pc)
        self.IG2 = IG_fusion(32,128,7,256)
        self.IG3 = IG_fusion(64,256,3,64)
        self.IG4 = IG_fusion(128,512,2,16)
        
        
    def forward(self, xyz,img):


        imgs = []
        for i in range(len(self.convd_list)):
            img = self.convd_list[i](img)        
            if i>0:
                imgs.append(img) 
       
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l1_points = self.IG1(imgs[0] ,l1_points)
        
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = self.IG2(imgs[1] ,l2_points)

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points = self.IG3(imgs[2] ,l3_points)
        
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l4_points = self.IG4(imgs[3] ,l4_points)
        

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        r = self.radius_pre(l0_points)
        c = torch.max(l0_points, 2, keepdim=True)[0]
        c = self.center_pre(c)
        c = c.view(-1, 3)

        return c,r
    
class SinglePathModule(nn.Module):
    def __init__(self, inC, outC, outS, expS):
        super(SinglePathModule, self).__init__()
        self.conv = nn.Conv3d(inC, outC, 7, 7, 1, groups=inC, bias=True)
        self.ln = nn.LayerNorm((outC, outS, outS, outS))
        self.mlp = nn.Sequential(
            nn.Conv3d(outC, outC, 1, 1, 0, bias=True),
            nn.GELU(),
            nn.Conv3d(outC, outC, 1, 1, 0, bias=True)
        )
        self.flat = nn.Sequential(
            nn.Conv1d(outS * outS * outS, expS, 1, 1, 0),
            nn.GELU(),
            nn.Conv1d(expS, expS, 1, 1, 0)
        )

    def forward(self, x):
    
        x = self.conv(x)
        x = self.ln(x)
        x = self.mlp(x)
        # Adjusted for batch processing
        batch_size = x.size(0)
        x = x.view(batch_size, -1, x.size(1))
        
        x = self.flat(x)
        return x.permute(0, 2, 1)


class IG_fusion_tem(nn.Module):
    def __init__(self, inC, outC, outS, expS):
        super(IG_fusion_tem, self).__init__()
        self.path_x1 = SinglePathModule(inC, outC, outS, expS)
        self.path_x2 = SinglePathModule(inC, outC, outS, expS)
        self.map = nn.Sequential(
            nn.Conv1d(outC * 3, outC, 1, 1, 0),
            nn.GELU(),
            nn.Conv1d(outC, outC, 1, 1, 0)
        )

    def forward(self, x1, x2, y):
        x1 = self.path_x1(x1)
        x2 = self.path_x2(x2)

        z = torch.cat([y, x1, x2], 1)
       
        return self.map(z)

class I2G_tem(nn.Module):
    def __init__(self,n_layers=4, padding_list =[0, 1, 1, 1],c=1, n=8, dropout=0.1, norm='bn',n_pc=1024):
        super(I2G_tem, self).__init__()
        'Original PointNet++'
        self.sa1 = PointNetSetAbstraction(n_pc, 0.1, 32, 3 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        # self.conv2 = nn.Conv1d(128, 3+3+1, 1)
        self.conv2 = nn.Conv1d(128, 3, 1)
  
        
        'Image_encoder'
        self.middle_channel = 2**(n_layers)*n
        self.dropout =dropout
        self.padding_list = padding_list
        self.n_layers = n_layers  

        self.convd_list = []
        for i in range(n_layers+1):
            if i ==0:
                self.convd_list.append(ConvD(c,     n, self.dropout, norm, first=True)) 
            else:
                self.convd_list.append(ConvD(2**(i-1)*n,   2**(i)*n, self.dropout, norm ,padding=self.padding_list[i-1]))
        self.convd_list = nn.ModuleList(self.convd_list)
        
        'Image PC fusions'  
        #  Img: inC, Img: outC,  Img: outS, PC: expS 
   
        self.IG1 = IG_fusion_tem(16,64,14,n_pc)
        self.IG2 = IG_fusion_tem(32,128,7,256)
        self.IG3 = IG_fusion_tem(64,256,3,64)
        self.IG4 = IG_fusion_tem(128,512,2,16)
        
        
    def forward(self, xyz,img_tem,img):


        imgs = []
        for i in range(len(self.convd_list)):
            img = self.convd_list[i](img)        
            if i>0:
                imgs.append(img) 
       
        imgs_tem = []
        for i in range(len(self.convd_list)):
            img_tem = self.convd_list[i](img_tem)        
            if i>0:
                imgs_tem.append(img_tem) 


        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l1_points = self.IG1(imgs_tem[0],imgs[0],l1_points)
        
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = self.IG2(imgs_tem[1],imgs[1]  ,l2_points)

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points = self.IG3(imgs_tem[2],imgs[2] ,l3_points)
        
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l4_points = self.IG4(imgs_tem[3],imgs[3] ,l4_points)
        

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points)), inplace=True))
        x = self.conv2(x)
        x = F.softplus(x)

        # x = x.permute(0, 2, 1)
        return x, l4_points
    

def spherical_to_cartesian_torch(r, theta, phi):
    x = r * torch.sin(phi) * torch.cos(theta)
    y = r * torch.sin(phi) * torch.sin(theta)
    z = r * torch.cos(phi)
    return x, y, z

def reverse_normal(r,min_,max_):
    return (max_-min_)*r +min_



# Compute the Laplacian loss for a point cloud using Annoy for approximate nearest neighbor search
def laplacian_loss(points, k):
    batch_size = points.size(0)
    total_loss = 0.0

    for b in range(batch_size):
        single_points = points[b, :, :]
        n, d = single_points.shape

        # Build Annoy index for each point cloud in the batch
        index = AnnoyIndex(d, 'euclidean')
        for i in range(n):
            index.add_item(i, single_points[i].tolist())
        index.build(10)

        # Find the approximate k nearest neighbors for each point
        indices = [index.get_nns_by_item(i, k+1)[1:] for i in range(n)]
        indices = torch.tensor(indices, dtype=torch.long)

        # Calculate the average of the k nearest neighbors for each point
        neighbors = single_points[indices]
        mean_neighbors = torch.mean(neighbors, dim=1)

        # Compute the difference and the loss
        diff = single_points - mean_neighbors
        loss = torch.mean(torch.norm(diff, dim=1) ** 2)
        total_loss += loss

    return total_loss / batch_size

class ChamferDistance(nn.Module):
    def __init__(self):
        super(ChamferDistance, self).__init__()

    def forward(self, pc1, pc2):
        batch_size = pc1.size(0)
        total_loss = 0.0

        for b in range(batch_size):
            single_pc1 = pc1[b, :, :]
            single_pc2 = pc2[b, :, :]
            N, D = single_pc1.size()

            single_pc1 = single_pc1.view(N, 1, D).repeat(1, N, 1)
            single_pc2 = single_pc2.view(1, N, D).repeat(N, 1, 1)

            norm = torch.norm(single_pc1 - single_pc2, dim=2)

            min_pc1_pc2, _ = torch.min(norm, dim=1)
            min_pc2_pc1, _ = torch.min(norm, dim=0)

            loss = (min_pc1_pc2.sum() + min_pc2_pc1.sum()) / (2.0 * N)
            total_loss += loss

        return total_loss / batch_size
    
import torch

def sample_from_batched_grid_with_values_and_labels(batched_point_cloud, batched_image_grid, batched_label_grid, grid_size=(192, 192, 192)):
    """
    Efficiently samples points from a grid around each point in each point cloud in a batch by considering the 8 neighboring points forming a cube. Retrieves corresponding pixel values and labels from batched image and label grids.

    Parameters:
    batched_point_cloud (Tensor): A tensor representing the batched point cloud.
    batched_image_grid (Tensor): The batched image grid from which pixel values are extracted.
    batched_label_grid (Tensor): The batched label grid from which labels are extracted.
    grid_size (tuple): The size of the grid in each dimension.

    Returns:
    tuple of Tensors: Two tensors, one containing sampled points with their pixel values, and another with their labels, for each batch.
    """
    batched_sampled_points_with_values = []
    batched_sampled_labels = []

    # Ensure cube_offsets are on the same device as the input tensors
    device = batched_point_cloud.device
    cube_offsets = torch.tensor([[x, y, z] for x in [-1, 0] for y in [-1, 0] for z in [-1, 0]], device=device)

    for batch_index in range(batched_point_cloud.shape[0]):
        sampled_points_with_values = []
        sampled_labels = []
        point_cloud = batched_point_cloud[batch_index]
        image_grid = batched_image_grid[batch_index, 0]
        label_grid = batched_label_grid[batch_index, 0]

        for point in point_cloud:
            closest_grid_point = torch.round(point).long()
            closest_grid_point = torch.stack([closest_grid_point[i].clamp(min=0, max=grid_size[i] - 1) for i in range(3)])

            # Move closest_grid_point to the same device as offset
            closest_grid_point = closest_grid_point.to(device)

            for offset in cube_offsets:
                neighbor_point = closest_grid_point + offset

                neighbor_point_clamped = torch.stack([neighbor_point[i].clamp(min=0, max=grid_size[i] - 1) for i in range(3)])
                pixel_value = image_grid[neighbor_point_clamped[0], neighbor_point_clamped[1], neighbor_point_clamped[2]]
                label_value = label_grid[neighbor_point_clamped[0], neighbor_point_clamped[1], neighbor_point_clamped[2]]
                sampled_points_with_values.append(torch.cat((neighbor_point_clamped, torch.tensor([pixel_value], device=device))))
                sampled_labels.append(label_value)

        batched_sampled_points_with_values.append(torch.stack(sampled_points_with_values))
        batched_sampled_labels.append(torch.stack(sampled_labels))

    # Stack all batched points into single tensors
    return (torch.stack(batched_sampled_points_with_values), torch.stack(batched_sampled_labels))


# class GC(nn.Module):
#     def __init__(self,n_layers=4, padding_list =[0, 1, 1, 1],c=1, n=8, dropout=0.1, norm='bn',n_pc=4096):
#         super(GC, self).__init__()
#         'Original PointNet++'
#         self.sa1 = PointNetSetAbstraction(n_pc*8, 0.1, 32, 4 + 3, [32, 32, 64], False)
#         self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
#         self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
#         self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
#         self.fp4 = PointNetFeaturePropagation(768, [256, 256])
#         self.fp3 = PointNetFeaturePropagation(384, [256, 256])
#         self.fp2 = PointNetFeaturePropagation(320, [256, 128])
#         self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
#         self.conv1 = nn.Conv1d(128, 128, 1)
#         self.bn1 = nn.BatchNorm1d(128)
#         self.drop1 = nn.Dropout(0.5)
#         # self.conv2 = nn.Conv1d(128, 3+3+1, 1)
#         self.conv2 = nn.Conv1d(128, 1, 1)
  
    
#     def forward(self, xyzv):

#         l0_points = xyzv
#         l0_xyz = xyzv[:,:3,:]

#         l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
#         l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
#         l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
#         l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        

#         l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
#         l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
#         l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
#         l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

#         x = self.drop1(F.relu(self.bn1(self.conv1(l0_points)), inplace=True))
#         x = self.conv2(x)
# #         x = F.sigmoid(x)
#         return x
from torch.autograd import Variable
class STNkd(nn.Module):
    def __init__(self, k=64,batch_size = 1):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, k*k, 1)
        self.conv2 = torch.nn.Conv1d(k*k, k*k*2, 1)
        self.conv3 = torch.nn.Conv1d(k*k*2, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(k*k)
        self.bn2 = nn.BatchNorm1d(k*k*2)
        self.bn3 = nn.BatchNorm1d(1024)
        self.batch_size = batch_size
        if self.batch_size >1:
            self.bn4 = nn.BatchNorm1d(512)
            self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        if self.batch_size>1:
            x = F.relu(self.bn4(self.fc1(x)))
            x = F.relu(self.bn5(self.fc2(x)))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x
    
class PointNetfeat(nn.Module):
    def __init__(self, f_dim=3,K=16, feature_transform = True):
        super(PointNetfeat, self).__init__()
        self.stn = STNkd(k=f_dim)
        self.conv1 = torch.nn.Conv1d(f_dim, K, 1)
        self.conv2 = torch.nn.Conv1d(K, K*K, 1)
        self.conv3 = torch.nn.Conv1d(K*K, K, 1)


        self.bn1 = nn.BatchNorm1d(K)
        self.bn2 = nn.BatchNorm1d(K*K)
        self.bn3 = nn.BatchNorm1d(K)
    

        self.feature_transform = feature_transform
        self.K = K
        if self.feature_transform:
            self.fstn = STNkd(k=K)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1,  self.K)

        x = x.view(-1, self.K, 1).repeat(1, 1, n_pts)
       
        return torch.cat([x, pointfeat], 1)

class PosE_Initial(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta

    def forward(self, xyz):
        B, _, N = xyz.shape
        feat_dim = self.out_dim // (self.in_dim * 2)

        # Detect the device from the input tensor
        device = xyz.device

        # Ensure all newly created tensors are on the same device
        feat_range = torch.arange(feat_dim).float().to(device)
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim).to(device)
        div_embed = torch.div(self.beta * xyz.unsqueeze(-1), dim_embed)

        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        position_embed = torch.stack([sin_embed, cos_embed], dim=4).flatten(3)
        position_embed = position_embed.permute(0, 1, 3, 2).reshape(B, self.out_dim, N)

        return position_embed
class GC(nn.Module):
    def __init__(self, num_classes=1, f_dim=4, K=16, feature_transform=True,poe=False):
        super(GC, self).__init__()
        self.poe =poe
        self.poe_module = PosE_Initial(3,6,0.5,2.5)
        if self.poe==False:
            self.feat = PointNetfeat(f_dim=f_dim, K=K, feature_transform=feature_transform)
        else:
            self.feat = PointNetfeat(f_dim=7, K=K, feature_transform=feature_transform)
        self.conv1 = nn.Conv1d(2 * K, K, 1)  # Assuming point features are concatenated
        self.conv2 = nn.Conv1d(K, K // 2, 1)
        self.conv3 = nn.Conv1d(K // 2, num_classes, 1)  # Output channels = num_classes
        self.bn1 = nn.BatchNorm1d(K)
        self.bn2 = nn.BatchNorm1d(K // 2)
        self.dropout = nn.Dropout(p=0.4)
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.poe == True:
            x_poe = self.poe_module(x[:,:3,:])
            x = torch.cat([x_poe,x[:,3:,:]],axis=1)

        x = self.feat(x)  # Output from PointNetfeat

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.dropout(self.conv2(x))))
        x = self.conv3(x)  # No activation here, raw scores
        return x
    
import torch
import torch.nn as nn
from utils.Gsupport import ConvD, ConvU  # Assuming these are your custom convolution modules.

class BaseModel(nn.Module):
    def __init__(self, n_layers=5, c=1, n=8, padding_list=[0, 0, 0, 0, 0, 0, 0], norm='bn', dropout=0.5):
        super(BaseModel, self).__init__()
        self.middle_channel = 2 ** (n_layers) * n
        self.dropout = dropout
        self.padding_list = padding_list
        self.n_layers = n_layers

        # Down sampling
        self.convd_list = nn.ModuleList([
            ConvD(c, n, self.dropout, norm, first=True) if i == 0 else
            ConvD(2 ** (i - 1) * n, 2 ** i * n, self.dropout, norm, padding=self.padding_list[i - 1])
            for i in range(n_layers + 1)
        ])

        # Up sampling
        self.convu_list = nn.ModuleList([
            ConvU(2 ** (i + 1) * n, self.dropout, norm, first=True if i == n_layers - 1 else False, padding=self.padding_list[i])
            for i in reversed(range(n_layers))
        ])

        self.seg = nn.Conv3d(n, 1, 1)
        self.sig = nn.Sigmoid()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        xs = []
        for i, conv in enumerate(self.convd_list):
           
            x = conv(x)
            if i != len(self.convd_list) - 1:
                xs.append(x)
        y = x
        
        ys = []
        for i, convu in enumerate(self.convu_list):
            y = convu(y, xs[::-1][i])
            ys.append(y)
        
        y = self.seg(y)
        y = self.sig(y)
        return y,ys[::-1]

class IG_fusion(nn.Module):
    def __init__(self, inC,outC,outS, expS):
        #  eg:（114,114,114,16）-> (1024,64),  inS = 114, inC=16, outS=16, outC=64, expS = 1024
        super(IG_fusion, self).__init__()
               
        self.conv = nn.Conv3d(inC, outC, 7, 7, 1, groups=inC, bias=True)
        self.ln = nn.LayerNorm((outC,outS, outS, outS))
        self.mlp_1 = nn.Conv3d(outC, outC, 1, 1, 0, bias=True)
        self.norm = nn.GELU()
        self.mlp_2 = nn.Conv3d(outC, outC, 1, 1, 0, bias=True)
        
        self.outC= outC
        self.flat1 = nn.Conv1d(outS*outS*outS,expS,1,1,0)
        self.flat2 = nn.Conv1d(expS,expS,1,1,0)
        
        self.map1 = nn.Conv1d(outC*2,outC,1,1,0)
        self.map2 = nn.Conv1d(outC,outC,1,1,0)    
        
    def forward(self, x, y):
        batch_size = x.shape[0]
        #   process img

        x = self.conv(x) 
    
        x = self.ln(x) 
        x = self.mlp_1(x) 
        x = self.norm(x) 
        x = self.mlp_2(x) 

        #   flatten
        # x = x.view(1,-1,self.outC)
        x = x.view(batch_size, -1, self.outC)

        x = self.flat1(x)
        x = self.norm(self.flat2(x))
        x = x.permute(0,2,1)

        #   Concat and fusion

        z = torch.cat([y,x],1)  #   (expS,outC*2)      

        z = self.map1(z)
        z = self.norm(self.map2(z))      
        
        return z
    
class PCModel(nn.Module):
    def __init__(self,n_layers=5,c=1, n=8, dropout=0.1, norm='bn',n_pc=4096):
        super(PCModel, self).__init__()
        self.sa1 = PointNetSetAbstraction(n_pc, 0.05, 32, 4 + 3, [4, 4, 8], False)
        self.sa2 = PointNetSetAbstraction(1024, 0.1, 16, 8 + 3, [8, 8, 16], False)
        self.sa3 = PointNetSetAbstraction(256, 0.2, 8,   16 + 3, [16, 16, 32], False)
        self.sa4 = PointNetSetAbstraction(64, 0.4, 4,    32 + 3, [32, 32, 64], False)
        self.sa5 = PointNetSetAbstraction(16, 0.8, 2,    64 + 3, [64, 64, 128], False)
        

        self.fp5 = PointNetFeaturePropagation(64+128, [128, 64])  # Correct as is
        self.fp4 = PointNetFeaturePropagation(32+64, [64, 32])  # Adjusted to match the new combined input
        self.fp3 = PointNetFeaturePropagation(16+32, [32, 16])    # Previous output 128 + SA2 output 32 = 160, corrected to match actual flow
        self.fp2 = PointNetFeaturePropagation(8+16, [16, 8])     # Previous output 64 + SA1 output 16 = 80, corrected
        self.fp1 = PointNetFeaturePropagation(8, [4, 3])     # Takes output from FP2

        self.IG1 = IG_fusion(8,8,27,n_pc)
        self.IG2 = IG_fusion(16,16,14,1024)
        self.IG3 = IG_fusion(32,32,7,256)
        self.IG4 = IG_fusion(64,64,3,64)
        self.IG5 = IG_fusion(128,128,2,16)
#         Radius prediction
        self.radius_pre = nn.Sequential(
            nn.Conv1d(3, 1, 1)

        )
#  
    def forward(self, xyz, ys):

        l0_points = xyz
        l0_xyz = xyz[:,:3,:]
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l1_points = self.IG1(ys[0] ,l1_points)
            
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = self.IG2(ys[1] ,l2_points)
            
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points = self.IG3(ys[2] ,l3_points)
            
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l4_points = self.IG4(ys[3] ,l4_points)
        
        l5_xyz, l5_points = self.sa5(l4_xyz, l4_points)
        l5_points = self.IG5(ys[4] ,l5_points)

        l4_points = self.fp5(l4_xyz, l5_xyz, l4_points, l5_points)            
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        r = self.radius_pre(l0_points)        
        return r 

class PCModel_wr(nn.Module):
    def __init__(self,n_layers=5,c=1, n=8, dropout=0.1, norm='bn',n_pc=4096):
        super(PCModel_wr, self).__init__()
        self.sa1 = PointNetSetAbstraction(n_pc, 0.05, 32, 4 + 3, [4, 4, 8], False)
        self.sa2 = PointNetSetAbstraction(1024, 0.1, 16, 8 + 3, [8, 8, 16], False)
        self.sa3 = PointNetSetAbstraction(256, 0.2, 8,   16 + 3, [16, 16, 32], False)
        self.sa4 = PointNetSetAbstraction(64, 0.4, 4,    32 + 3, [32, 32, 64], False)
        self.sa5 = PointNetSetAbstraction(16, 0.8, 2,    64 + 3, [64, 64, 128], False)
        

        self.fp5 = PointNetFeaturePropagation(64+128, [128, 64])  # Correct as is
        self.fp4 = PointNetFeaturePropagation(32+64, [64, 32])  # Adjusted to match the new combined input
        self.fp3 = PointNetFeaturePropagation(16+32, [32, 16])    # Previous output 128 + SA2 output 32 = 160, corrected to match actual flow
        self.fp2 = PointNetFeaturePropagation(8+16, [16, 8])     # Previous output 64 + SA1 output 16 = 80, corrected
        self.fp1 = PointNetFeaturePropagation(8, [4, 3])     # Takes output from FP2

        self.IG1 = IG_fusion(8,8,27,n_pc)
        self.IG2 = IG_fusion(16,16,14,1024)
        self.IG3 = IG_fusion(32,32,7,256)
        self.IG4 = IG_fusion(64,64,3,64)
        self.IG5 = IG_fusion(128,128,2,16)
#         Radius prediction
        self.radius_pre = nn.Sequential(
            nn.Conv1d(4, 16, 1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(dropout),
            nn.Conv1d(16, 1, 1)
        )
#  
    def forward(self, xyz, ys):

        l0_points = xyz
        l0_xyz = xyz[:,:3,:]
        l0_r = xyz[:, 3:4, :]  # Radius
        
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l1_points = self.IG1(ys[0] ,l1_points)
            
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = self.IG2(ys[1] ,l2_points)
            
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points = self.IG3(ys[2] ,l3_points)
            
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l4_points = self.IG4(ys[3] ,l4_points)
        
        l5_xyz, l5_points = self.sa5(l4_xyz, l4_points)
        l5_points = self.IG5(ys[4] ,l5_points)

        l4_points = self.fp5(l4_xyz, l5_xyz, l4_points, l5_points)            
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        
        delta_r  = self.radius_pre(torch.cat([l0_points, l0_r], dim=1))  
        r = l0_r + delta_r 
        return r 
    

class PCModel_wr_v3(nn.Module):
    def __init__(self,n_layers=5,c=1, n=8, dropout=0.1, norm='bn',n_pc=4096):
        super(PCModel_wr_v3, self).__init__()
        self.sa1 = PointNetSetAbstraction(n_pc, 0.05, 32, 4 + 3, [4, 4, 8], False)
        self.sa2 = PointNetSetAbstraction(1024, 0.1, 16, 8 + 3, [8, 8, 16], False)
        self.sa3 = PointNetSetAbstraction(256, 0.2, 8,   16 + 3, [16, 16, 32], False)
        self.sa4 = PointNetSetAbstraction(64, 0.4, 4,    32 + 3, [32, 32, 64], False)
        self.sa5 = PointNetSetAbstraction(16, 0.8, 2,    64 + 3, [64, 64, 128], False)
        

        self.fp5 = PointNetFeaturePropagation(64+128, [128, 64])  # Correct as is
        self.fp4 = PointNetFeaturePropagation(32+64, [64, 32])  # Adjusted to match the new combined input
        self.fp3 = PointNetFeaturePropagation(16+32, [32, 16])    # Previous output 128 + SA2 output 32 = 160, corrected to match actual flow
        self.fp2 = PointNetFeaturePropagation(8+16, [16, 8])     # Previous output 64 + SA1 output 16 = 80, corrected
        self.fp1 = PointNetFeaturePropagation(8, [4, 3])     # Takes output from FP2

        self.IG1 = IG_fusion(8,8,27,n_pc)
        self.IG2 = IG_fusion(16,16,14,1024)
        self.IG3 = IG_fusion(32,32,7,256)
        self.IG4 = IG_fusion(64,64,3,64)
        self.IG5 = IG_fusion(128,128,2,16)

        self.feature_process = nn.Sequential(
            nn.Conv1d(4, 16, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 1, 1),
            nn.BatchNorm1d(1),
            nn.ReLU()
        )
        self.sig = nn.Sigmoid()
        self.gate_conv = nn.Conv1d(4, 1, 1, 1, 0)
#         Radius prediction
        self.radius_pre = nn.Sequential(
            nn.Conv1d(1, 4, 1),
            nn.ReLU(),
            nn.BatchNorm1d(4),
            nn.Conv1d(4, 1, 1)
        )
#          g = self.relu(self.bn2(self.conv_xg(xg)))
    def forward(self, xyz, ys):
 
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]
        l0_r = xyz[:, 3:4, :]  # Radius

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l1_points = self.IG1(ys[0] ,l1_points)
            
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = self.IG2(ys[1] ,l2_points)
            
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points = self.IG3(ys[2] ,l3_points)
            
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l4_points = self.IG4(ys[3] ,l4_points)
        
        l5_xyz, l5_points = self.sa5(l4_xyz, l4_points)
        l5_points = self.IG5(ys[4] ,l5_points)

        l4_points = self.fp5(l4_xyz, l5_xyz, l4_points, l5_points)            
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        
        # # processed point-wise feature  cat original radius information
        feature = torch.cat([l0_points, l0_r], dim=1)
        delta_r = self.feature_process(feature)


        r =  self.radius_pre(l0_r +  self.sig(self.gate_conv(feature)) * delta_r)
        return r 
    
# # Create model instance
# basemodel = BaseModel()

# # Load the model
# saved_model_path = 'saved_models_2024/Unet4096.pth'
# saved_state_dict = torch.load(saved_model_path, map_location='cpu')
# basemodel.load_state_dict(saved_state_dict)

# # Freeze all existing parameters
# for param in basemodel.parameters():
#     param.requires_grad = False

# Add new modules
class ExtendedModel_pc(nn.Module):
    def __init__(self, base_model):
        super(ExtendedModel_pc, self).__init__()
        self.base_model = base_model
        self.pc_evolve = PCModel()

    def forward(self, x,coors_tem):
        x,ys = self.base_model(x)
        r_out = self.pc_evolve(coors_tem,ys)
        return r_out

# # Create extended model instance with base model
# extended_model = ExtendedModel(basemodel)

# # Now `extended_model` has frozen pre-trained parts and trainable new layers
# a,b = extended_model(img,coors_tem)
class ExtendedModel_pcr(nn.Module):
    def __init__(self, base_model):
        super(ExtendedModel_pcr, self).__init__()
        self.base_model = base_model
        self.pc_evolve = PCModel_wr()

    def forward(self, x,coors_tem):
        x,ys = self.base_model(x)
        r_out = self.pc_evolve(coors_tem,ys)
        return r_out

class ExtendedModel_pcr_v3(nn.Module):
    def __init__(self, base_model):
        super(ExtendedModel_pcr_v3, self).__init__()
        self.base_model = base_model
        self.pc_evolve = PCModel_wr_v3()

    def forward(self, x,coors_tem):
        x,ys = self.base_model(x)
        r_out = self.pc_evolve(coors_tem,ys)
        return r_out
