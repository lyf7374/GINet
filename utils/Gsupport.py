import numpy as np
import torch.nn.init as init
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.spatial.distance import directed_hausdorff
import os 

dropout =0.1
 
def sub_sampling_blockwise(n_regions_phi=128,
                           n_regions_theta=128,
                           sample_regions_phi=64,
                           sample_regions_theta=64,
                           samples_per_block=1,
                           random_seed=None):
    """
    Divides a grid of size n_regions_phi * n_regions_theta into 
    sample_regions_phi * sample_regions_theta blocks and picks 
    `samples_per_block` points from each block.

    Parameters
    ----------
    samples_per_block : int
        Number of samples to take per (phi, theta) block.
    random_seed : int or None
        Random seed for reproducibility (if >1 sample per block).

    Returns
    -------
    sampled_indices : list of int
        Indices into the flattened full grid that were kept.
    sampled_points : list of (phi, theta)
        The subsampled (phi, theta) coordinates.
    """
    assert n_regions_phi  % sample_regions_phi  == 0, "phi blocks must divide evenly"
    assert n_regions_theta % sample_regions_theta == 0, "theta blocks must divide evenly"

    if random_seed is not None:
        np.random.seed(random_seed)

    # bin edges and centers
    phi_bins   = np.linspace(0, np.pi,     n_regions_phi   + 1)
    theta_bins = np.linspace(-np.pi, np.pi, n_regions_theta + 1)

    full_points = [((phi_bins[i] + phi_bins[i+1]) / 2,
                    (theta_bins[j] + theta_bins[j+1]) / 2)
                   for i in range(n_regions_phi)
                   for j in range(n_regions_theta)]

    step_phi   = n_regions_phi  // sample_regions_phi
    step_theta = n_regions_theta // sample_regions_theta

    sampled_indices = []
    sampled_points  = []

    for bi in range(sample_regions_phi):
        for bj in range(sample_regions_theta):
            # grid indices of the block
            i_start = bi * step_phi
            j_start = bj * step_theta
            block_indices = [
                (i, j)
                for i in range(i_start, i_start + step_phi)
                for j in range(j_start, j_start + step_theta)
            ]

            # choose sample(s) from block
            chosen = np.random.choice(len(block_indices), size=min(samples_per_block, len(block_indices)), replace=False)
            for c in chosen:
                i, j = block_indices[c]
                idx = i * n_regions_theta + j
                sampled_indices.append(idx)
                sampled_points.append(full_points[idx])

    print(f"Coarsened {n_regions_phi*n_regions_theta} → {len(sampled_indices)} points")
    return sampled_indices, sampled_points


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs_, targets, smooth=1,sig=False):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = inputs_.clone()  
        if sig==True:
            inputs = torch.sigmoid(inputs) 

        # inputs[inputs>=0.5] = 1
        # inputs[inputs<0.5] = 0
        #flatten label and prediction tensors
    
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class DiceLoss_batch(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss_batch, self).__init__()

    def forward(self, inputs_, targets, smooth=1, sig=False):
        # Apply sigmoid if specified
        if sig:
            inputs_ = torch.sigmoid(inputs_)

        # Initialize dice score list
        dice_scores = []

        # Loop through the batch
        for i in range(inputs_.size(0)):
            inputs = inputs_[i].view(-1)
            targets_batch = targets[i].view(-1)
            
            intersection = (inputs * targets_batch).sum()
            dice = (2. * intersection + smooth) / (inputs.sum() + targets_batch.sum() + smooth)
            
            dice_scores.append(1 - dice)

        # Convert list to tensor and return mean dice loss
        dice_scores = torch.tensor(dice_scores)
        
        return dice_scores.mean()
    
# from scipy.spatial.distance import directed_hausdorff
class HausdorffDistance(nn.Module):
    def __init__(self):
        super(HausdorffDistance, self).__init__()

    def forward(self, input, target):
        # Ensure input is binary
        input = (input >= 0.5).float()
        target = (target >= 0.5).float()

        # Convert PyTorch tensors to numpy arrays for scipy compatibility
        input_np = input.cpu().detach().numpy()
        target_np = target.cpu().detach().numpy()

        # Find indices of non-zero elements, representing coordinates of the points
        input_points = np.argwhere(input_np)
        target_points = np.argwhere(target_np)

        # Compute directed Hausdorff distances and then take the maximum for true Hausdorff distance
        hd1 = directed_hausdorff(input_points, target_points)[0]
        hd2 = directed_hausdorff(target_points, input_points)[0]
        
        hausdorff_distance = max(hd1, hd2)

        return hausdorff_distance


class Loss_all(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Loss_all, self).__init__()

    def forward(self, inputs_, targets, smooth=1):
       
        inputs = inputs_.clone()  
        # inputs = F.sigmoid(inputs)   
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        inputs[inputs>=0.5] = 1
        inputs[inputs<0.5] = 0
        
        
        T_ = targets[inputs==targets]
        F_ = targets[inputs!=targets]
        
        TP = T_.sum()  #  TP
        TN = len(T_) - T_.sum()  #  TN
        
        FP = F_.sum() # FP
        FN = len(F_)-F_.sum() # FN
        
        precision= TP/(TP+FP)
        recall =  TP/(TP+FN)
        FPR = FP/(FP+TN)
        FNR = FN/(FN+TP)
  
        return precision,recall,FPR,FNR

def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / r)
    return r, theta, phi
def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z


def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f
def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / r)
    return r, theta, phi
def spherical_to_cartesian_np(r, theta, phi):
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z


def convert2GI(GI,n_patch,relocate=True):
    phi_bins = np.linspace(0, np.pi, n_patch + 1)
    theta_bins = np.linspace(-np.pi, np.pi, n_patch + 1)
    angles = np.zeros((n_patch, n_patch, 2))
    
    n_regions_phi = n_patch
    n_regions_theta = n_patch
    for i in range(n_regions_phi):
        for j in range(n_regions_theta):
            phi_center = (phi_bins[i] + phi_bins[i + 1]) / 2
            theta_center = (theta_bins[j] + theta_bins[j + 1]) / 2

            angles[i, j] = [theta_center, phi_center]

    angles = angles.reshape(n_patch*n_patch, 2)
    GI_data = np.zeros((GI.shape[0], GI.shape[1], 4))
    
    for patient_idx, patient_data in enumerate(GI):
        for point_idx, point in enumerate(patient_data):
            x, y, z = point
            r,_,_ =  cartesian_to_spherical(x, y, z)
            if relocate ==True:
                theta,phi = angles[point_idx]
                x,y,z = spherical_to_cartesian_np(r,theta,phi)
                GI_data[patient_idx, point_idx] = np.array([x,y,z,r])
            else:
                GI_data[patient_idx, point_idx] = np.array([x,y,z,r])
    return GI_data

def normalize_radius(GI_cartesian,g_tem_in=None, tem=False):
    
    min_r = np.min(GI_cartesian[:, :, 3])
    max_r = np.max(GI_cartesian[:, :, 3])

    if tem==False:
        normalized_data = np.copy(GI_cartesian)
        normalized_data[:, :, 3] = (GI_cartesian[:, :, 3] - min_r) / (max_r - min_r)
    else:
        normalized_data = np.copy(g_tem_in)
        normalized_data[:, :, 3] = (g_tem_in[:, :, 3] - min_r) / (max_r - min_r)
    return normalized_data

def spherical_to_cartesian_torch(r, theta, phi):
    x = r * torch.sin(phi) * torch.cos(theta)
    y = r * torch.sin(phi) * torch.sin(theta)
    z = r * torch.cos(phi)
    return x, y, z

def reverse_normal(r,min_,max_):
    return (max_-min_)*r +min_


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
    def __init__(self, inplanes, planes, dropout=0, norm='bn', first=False, padding = 0):
        super(ConvD, self).__init__()

        self.first = first
        # if self.first==True:
        #     group = inplanes
        # else:
        #     group = 1
        group = 1
        self.maxpool = nn.MaxPool3d(2, 2,padding = padding)

        self.dropout = dropout

        self.relu = nn.LeakyReLU(0.2,inplace=False)
        self.conv1 = nn.Conv3d(inplanes, planes, 3, 1, 1, bias=False,groups=group)
        self.bn1   = normalization(planes, norm)

        self.conv2 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False,groups=group)
        self.bn2   = normalization(planes, norm)

        self.conv3 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False,groups=group)
        self.bn3   = normalization(planes, norm)

    def forward(self, x):
        if not self.first:
            x = self.maxpool(x)
        x = self.bn1(self.conv1(x))
        if self.dropout > 0:
            x = F.dropout3d(x, self.dropout)
        y = self.relu(self.bn2(self.conv2(x)))
        y = self.bn3(self.conv3(x))
        return self.relu(x + y)


class ConvU(nn.Module):
    def __init__(self, planes,dropout=0, norm='bn', first=False, padding = 0):
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
        self.dropout = dropout
     
        self.relu = nn.LeakyReLU(0.2,inplace=False)  
    def forward(self, x, prev):
        # final output is the localization layer
        y = self.upsampling(x)
        if self.dropout > 0:
            x = F.dropout3d(x, self.dropout)
        y = self.relu(self.bn2(self.conv2(y)))
        y = torch.cat([prev, y], 1)
        y = self.relu(self.bn3(self.conv3(y)))

        return y
    
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
    def __init__(self, f_dim=4,K=16, feature_transform = True):
        super(PointNetfeat, self).__init__()
        self.stn = STNkd(k=f_dim)
        self.conv1 = torch.nn.Conv1d(f_dim, K, 1)
        # self.conv2 = torch.nn.Conv1d(K, K*K, 1)
        # self.conv3 = torch.nn.Conv1d(K*K, K, 1)


        self.bn1 = nn.BatchNorm1d(K)
        # self.bn2 = nn.BatchNorm1d(K*K)
        # self.bn3 = nn.BatchNorm1d(K)
    

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

        return pointfeat, trans, trans_feat 

class Fusion(nn.Module):
    def __init__(self, inplanes, planes, norm='bn', first=False, padding = 0):
        super(Fusion, self).__init__()

        self.relu = nn.LeakyReLU(0.2,inplace=True)
        
        self.conv1 = nn.Conv3d(inplanes, planes, 3, 1, 1)
        self.bn1   = normalization(planes, norm)
        self.conv2 = nn.Conv3d(planes, planes, 3, 1, 1)
        self.bn2   = normalization(planes, norm)
        self.conv3 = nn.Conv3d(planes, planes, 3, 1, 1)
        self.bn3   = normalization(planes, norm)
    def forward(self, x):
        
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.relu(self.bn2(self.conv2(y)))
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

    # Ensure nsample is not greater than N
    if nsample > N:
        raise ValueError("nsample should not be greater than the number of points in xyz")

    # Initialize group_idx with indices of all points
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])

    # Calculate squared distances and set indices for out-of-radius points to N-1
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N - 1

    # Sort indices based on distances and select the first `nsample` for each query point
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

    # Handle cases where there are fewer than `nsample` points within the radius
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == (N - 1)
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
    
def model_load(model, dic_pth):
    state_dict = torch.load(dic_pth, map_location=torch.device('cpu'))
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # remove `module.` prefix if it exists
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model


def sub_sampling(n_regions_phi=64,n_regions_theta=64,angle=20,step_h=1,step_l=8):
    # Assuming 'selected_points' is your list of points (phi, theta) as provided earlier

    # Define the thresholds for very high and very low phi regions
    # We assume that the very high phi region is the top 10% and the very low phi region is the bottom 10% of the phi range
    phi_bins = np.linspace(0, np.pi, n_regions_phi + 1)
    theta_bins = np.linspace(-np.pi, np.pi, n_regions_theta + 1)

    selected_points = []
    for i in range(n_regions_phi):
        for j in range(n_regions_theta):
            phi_center = (phi_bins[i] + phi_bins[i + 1]) / 2
            theta_center = (theta_bins[j] + theta_bins[j + 1]) / 2
            selected_points.append(( phi_center,theta_center))
    phi_bins = np.linspace(0, np.pi, n_regions_phi + 1)  # Recreating phi_bins as they were used in the original point creation
    phi_threshold_low = np.percentile(phi_bins, angle)
    phi_threshold_high = np.percentile(phi_bins, 100-angle)

    # Function to categorize the phi value into three aspects: very high, very low, and middle phi
    def categorize_phi_fixed_selection(phi, phi_threshold_low, phi_threshold_high):
        if phi <= phi_threshold_low:
            return 'very low'
        elif phi >= phi_threshold_high:
            return 'very high'
        else:
            return 'middle'

    # Categorizing each point and storing them in separate lists
    fixed_selection_very_high_phi_points = []
    fixed_selection_very_low_phi_points = []
    fixed_selection_middle_phi_points = []

    angles_mid  = []
    angles_low  = []
    angles_high = []
    for i,point in enumerate(selected_points):
        category = categorize_phi_fixed_selection(point[0], phi_threshold_low, phi_threshold_high)
        if category == 'very high':
            fixed_selection_very_high_phi_points.append(i)
            angles_high.append(point)
        elif category == 'very low':
            fixed_selection_very_low_phi_points.append(i)
            angles_low.append(point)
        elif category == 'middle':
            fixed_selection_middle_phi_points.append(i)
            angles_mid.append(point)



    step = 8
    angles_high = angles_high[::step_h]
    fixed_selection_very_high_phi_points = fixed_selection_very_high_phi_points[::step_h]
    angles_low = angles_low[::step_l]
    fixed_selection_very_low_phi_points = fixed_selection_very_low_phi_points[::step_l]
    # Combining the points from all categories
    fixed_selection_points = fixed_selection_very_high_phi_points + fixed_selection_very_low_phi_points + fixed_selection_middle_phi_points
    # Output the number of points in each category and total
    print(f"Very High Phi: {len(fixed_selection_very_high_phi_points)}")
    print(f"Very Low Phi: {len(fixed_selection_very_low_phi_points)}")
    print(f"Middle Phi: {len(fixed_selection_middle_phi_points)}")
    print(f"Total Points: {len(fixed_selection_points)}")
    angles = angles_low + angles_mid + angles_high
    
    # edges = extract_edge(angles,n_regions_phi,n_regions_theta)
    return fixed_selection_points ,None