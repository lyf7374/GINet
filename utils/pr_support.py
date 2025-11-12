import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

device = "cuda" if torch.cuda.is_available() else "cpu"

class PosE_Initial(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta

    def forward(self, xyz):
        B, _, N = xyz.shape    
        feat_dim = self.out_dim // (self.in_dim * 2)
        
        feat_range = torch.arange(feat_dim).float()   
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim).to(device)
        div_embed = torch.div(self.beta * xyz.unsqueeze(-1), dim_embed)

        sin_embed = torch.sin(div_embed).squeeze(-1)
        cos_embed = torch.cos(div_embed).squeeze(-1)

        position_embed = torch.cat([sin_embed, cos_embed], dim=2)
        position_embed = position_embed.permute(0, 2, 1)
        
        return position_embed


def interpolate_pc(point_cloud,
    n_regions_phi_new = 96,n_regions_theta_new = 96,n_regions_phi_original = 64,n_regions_theta_original = 64):
    # Original grid
    if (n_regions_phi_new==64) and (n_regions_theta_new==64):
        return point_cloud
    phi_original = np.linspace(0, np.pi, n_regions_phi_original, endpoint=False)
    theta_original = np.linspace(-np.pi, np.pi, n_regions_theta_original, endpoint=False)
    phi_grid_original, theta_grid_original = np.meshgrid(phi_original, theta_original)

    # New grid
    phi_new = np.linspace(0, np.pi, n_regions_phi_new, endpoint=False)
    theta_new = np.linspace(-np.pi, np.pi, n_regions_theta_new, endpoint=False)
    phi_grid_new, theta_grid_new = np.meshgrid(phi_new, theta_new)

    # Flatten the original grid for griddata input
    points_original = np.vstack((phi_grid_original.flatten(), theta_grid_original.flatten())).T

    # Convert spherical to Cartesian coordinates for original points - assuming this is needed
    # Example conversion placeholder - replace with actual conversion if your data is in spherical coordinates
    # x, y, z = sph_to_cart(r, phi, theta) # Implement this based on your specific case

    # Interpolate in Cartesian space
    points_new = np.vstack((phi_grid_new.flatten(), theta_grid_new.flatten())).T
    cartesian_coordinates_new = griddata(points_original, point_cloud, points_new, method='linear')

    # Reshape the interpolated Cartesian coordinates back to grid format if needed
#     interpolated_point_cloud = cartesian_coordinates_new.reshape((n_regions_phi_new, n_regions_theta_new, 3))

    # Assume cartesian_coordinates_new is the output array from griddata, which may contain NaNs
    nan_mask = np.isnan(cartesian_coordinates_new[:, 0])  # Assuming NaNs are in the x-coordinate

    # Coordinates of the valid (non-NaN) and invalid (NaN) points
    valid_coords = np.argwhere(~nan_mask)
    invalid_coords = np.argwhere(nan_mask)

    # Build a KDTree for efficient nearest-neighbor queries
    tree = cKDTree(valid_coords)

    # For each invalid coordinate, find its nearest valid coordinate
    _, nearest_indices = tree.query(invalid_coords)

    # Fill NaN values with the nearest non-NaN values
    # Note: This example assumes a simple structure where cartesian_coordinates_new is directly indexable by the flat indices
    for invalid, nearest in zip(invalid_coords.flatten(), nearest_indices):
        cartesian_coordinates_new[invalid] = cartesian_coordinates_new[nearest]
    return cartesian_coordinates_new



def sample_from_3d_volume(input_, pc):
    """
    Sample values from a 3D volume at specified 3D points for each batch element.
    
    Args:
        input_ (torch.Tensor): The input tensor with shape (N, C, D, H, W).
                               N is batch size, C is number of channels,
                               D, H, W are depths, heights, and widths of the volume.
        pc (torch.Tensor): The tensor containing the 3D points,
                           expected shape (N, num_points, 3) with integer coordinates.
    
    Returns:
        torch.Tensor: The values at the specified points in the volume for each batch and channel.
                      Shape (N, C, num_points).
    """
    # Move the point coordinates to the same device as the input tensor
    pc = pc.to(input_.device)

    # Ensure pc is of type long for indexing and clamp the values to avoid out-of-bounds indexing
    pc_indices = pc.long()
    D, H, W = input_.shape[2], input_.shape[3], input_.shape[4]

    # Clamp the indices to ensure they are within the valid range
    z_indices = pc_indices[..., 0].clamp(0, D - 1)
    y_indices = pc_indices[..., 1].clamp(0, H - 1)
    x_indices = pc_indices[..., 2].clamp(0, W - 1)

    # Use advanced indexing to gather the values
    # No need to expand the indices since each batch should correspond to its own set of points
    sampled_values = torch.stack([
        input_[b, :, z_indices[b], y_indices[b], x_indices[b]] for b in range(input_.shape[0])
    ], dim=0)

    return sampled_values



class PointNetfeat(nn.Module):
    def __init__(self, f_dim=3,K=256, feature_transform = True):
        super(PointNetfeat, self).__init__()
        '''
        
        Potential modules:  
                            1. Position embedding.
                            2. Layer-wise feature fusion
                            
        '''
  
        self.conv1 = torch.nn.Conv1d(f_dim, K, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = torch.nn.Conv1d(K, 2*K,  kernel_size=1, stride=1, padding=0, bias=True)
        self.conv3 = torch.nn.Conv1d(2*K, 4*K,  kernel_size=1, stride=1, padding=0, bias=True)
        self.conv4 = torch.nn.Conv1d(4*K, 2*K,  kernel_size=1, stride=1, padding=0, bias=True)
        self.conv5 = torch.nn.Conv1d(2*K, K,  kernel_size=1, stride=1, padding=0, bias=True)
        self.predictor = nn.Conv1d(K, 1, kernel_size=1, stride=1, padding=0)
    def forward(self, point_features):
        x = point_features
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.predictor(x)
        
        return x
    
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
        return y,ys[-1]


class PointNetfeat(nn.Module):
    def __init__(self, f_dim=3, feature_transform=True):
        super(PointNetfeat, self).__init__()
        # Set the channel sizes
        self.channels = [f_dim, 64, 128, 256, 512, 256, 128]
        
        # Creating convolutional and batch normalization layers
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        for i in range(len(self.channels) - 1):
            self.conv_layers.append(nn.Conv1d(self.channels[i], self.channels[i + 1], kernel_size=1, bias=False))
            self.bn_layers.append(nn.BatchNorm1d(self.channels[i + 1]))
        
        # Final predictor layer
        self.predictor = nn.Conv1d(self.channels[-1], 1, kernel_size=1)

    def forward(self, x):
        # Process through all but the last layer with residual connections
        for i in range(0, len(self.conv_layers) - 1, 2):
            residual = x
            x = self.bn_layers[i](self.conv_layers[i](x))
            x = F.leaky_relu(x, negative_slope=0.01)
            x = self.bn_layers[i + 1](self.conv_layers[i + 1](x))
            x = F.leaky_relu(x, negative_slope=0.01)
            
            # Apply residual connection
            if x.size() == residual.size():
                x += residual  # Only add if dimensions match
            else:
                # Handling different dimensions, potentially using a projection
                x += self.projection(residual, self.channels[i + 2])

        # Output predictor layer
        x = self.predictor(x)
        return x

    def projection(self, x, out_channels):
        # Adjust the number of channels with a 1x1 convolution if necessary
        proj = nn.Conv1d(x.size(1), out_channels, kernel_size=1, bias=False).to(x.device)
        return proj(x)

class PointRend(nn.Module):
    def __init__(self, base_model, f_dim_img=8, use_POE=True,final=False):
        super(PointRend, self).__init__()
        self.base_model = base_model
        self.use_POE = use_POE
        self.final_tune = final
        # Initialize Positional Encoding (POE) if enabled
        if self.use_POE:
            self.poe = PosE_Initial(3, 6, 0.5, 1.0)
        else:
            self.poe = None  # Explicitly set to None if not using POE

        # Determine the dimension for the point features
        f_dim_point = 1 + f_dim_img
        if self.use_POE:
            f_dim_point += 6  # Add dimensions for POE outputs

        self.point_refine = PointNetfeat(f_dim_point)
        self.final_refine =  nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
                                 )
    def forward(self, x, g):
        pre_seg, feature_map = self.base_model(x)
        
        # Sample features from the 3D volume and pre-segmentation
        fea_coor = sample_from_3d_volume(feature_map, g)
        pre_coor = sample_from_3d_volume(pre_seg, g)
       
        # Get positional encodings if POE is enabled
        if self.use_POE:
            coor_poe = self.poe(g)
            point_features = torch.cat((pre_coor, coor_poe, fea_coor), dim=1)
        else:
            point_features = torch.cat((pre_coor, fea_coor), dim=1)
       
        # Refine the point features to get the output logits
        out = self.point_refine(point_features)
        if self.final_tune==True:
            out = update_volume_with_predictions(pre_seg,g,out)
            out = self.final_refine(out)
        else:
            out = F.sigmoid(out)
        return out


def update_volume_with_predictions(original_volume, pc, refined_predictions):
    """
    Update a 3D volume with refined predictions at specified 3D points for each batch.
    
    Args:
        original_volume (torch.Tensor): The original prediction volume with shape (N, 1, D, H, W).
                                        N is batch size, D, H, W are depths, heights, and widths of the volume.
        pc (torch.Tensor): The tensor containing the 3D points for each batch,
                           expected shape (N, num_points, 3) with integer coordinates.
        refined_predictions (torch.Tensor): The refined predictions for the points for each batch,
                                            shape (N, num_points, 1).
    
    Returns:
        torch.Tensor: The updated volume.
    """
    # Ensure pc is of type long for indexing and clamp the values to avoid out-of-bounds indexing
    pc_indices = pc.long()
    N, D, H, W = original_volume.shape[0], original_volume.shape[2], original_volume.shape[3], original_volume.shape[4]

    # Clamp the indices to ensure they are within the valid range
    z_indices = pc_indices[..., 0].clamp(0, D-1)
    y_indices = pc_indices[..., 1].clamp(0, H-1)
    x_indices = pc_indices[..., 2].clamp(0, W-1)
    
    # Update the volume for each batch
    for b in range(N):
        for i in range(pc_indices.shape[1]):  # Iterate over the number of points
            z, y, x = z_indices[b, i], y_indices[b, i], x_indices[b, i]
            original_volume[b, 0, z, y, x] = refined_predictions[b, 0, i]

    return original_volume
