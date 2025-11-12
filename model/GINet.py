import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.Gsupport import ConvD,ConvU,normalization
from scipy.spatial import cKDTree
from torch.autograd import Variable
from utils.pc_est_support import  PCModel_wr_v3
from scipy.ndimage import binary_fill_holes, binary_closing, generate_binary_structure,binary_dilation
device = "cuda" if torch.cuda.is_available() else "cpu"
dropout= 0.1
image_shape = 192


class PosE_Initial(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super(PosE_Initial, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha = alpha
        self.beta = beta

    def forward(self, xyz):
        B, _, N = xyz.shape

         # Normal
        xyz = xyz/191
        # Calculate the number of frequency pairs per input dimension
        feat_dim = self.out_dim // (self.in_dim * 2)

        # Generate feature range for the exponential decay based on alpha
        feat_range = torch.arange(feat_dim).float().to(xyz.device)
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)

        # Calculate the embeddings
#         dim_embed = dim_webatch size, input dimension, number of points, number of frequencies)
        div_embed = self.beta * xyz.unsqueeze(-1) / dim_embed

        # Apply sin and cos functions
        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)

        # Reshape sin and cos embeddings to combine the last two dimensions
        sin_embed = sin_embed.reshape(B, self.in_dim * feat_dim, N)
        cos_embed = cos_embed.reshape(B, self.in_dim * feat_dim, N)

        # Concatenate along the feature dimension
        position_embed = torch.cat([sin_embed, cos_embed], dim=1)  # Results in (B, in_dim * feat_dim * 2, N)

        return position_embed

def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(np.sqrt(x**2 + y**2), z)
    phi = np.arctan2(y, x)
    return r, theta, phi

def interpolate_spherical_points(vertices, center, num_points=4,r_in = False):
    interpolated_points = []
    spherical_data = []

    for point in vertices:
        x, y, z = point
        a, b, c = center
        r, theta, phi = cartesian_to_spherical(x-a, y-b, z-c)
        spherical_data.append((r, theta, phi))
        
        # Generate interpolated points for this (r, theta, phi)
        rs = np.linspace(0, r, num_points)
        for ri in rs:
            xi = a + ri * np.sin(theta) * np.cos(phi)
            yi = b + ri * np.sin(theta) * np.sin(phi)
            zi = c + ri * np.cos(theta)
            if r_in==False:
                interpolated_points.append([xi, yi, zi])
            else:
                interpolated_points.append([xi, yi, zi,ri])

#     return np.array(interpolated_points), np.array(spherical_data)
    return np.array(interpolated_points)


def cartesian_to_spherical_torch(x, y, z):
    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.atan2(torch.sqrt(x**2 + y**2), z)
    phi = torch.atan2(y, x)
    return r, theta, phi
def interpolate_spherical_points_torch(vertices, center, num_points=4):
    # Expect center to have the shape (Batch, 1, 3)
    # Expand center to match vertices for broadcasting
    center = center.unsqueeze(1)
    
    # Calculate relative positions and spherical coordinates
    relative_positions = vertices - center
    x, y, z = relative_positions.unbind(-1)
    
    r, theta, phi = cartesian_to_spherical_torch(x, y, z)

    # Prepare to interpolate
    interpolated_points = []
    
    # Linear interpolation of radii
    rs = torch.linspace(0, 1, num_points, device=vertices.device).unsqueeze(-1) * r.unsqueeze(1)
    
    # Calculating spherical to cartesian for interpolated points
    sin_theta = torch.sin(theta).unsqueeze(1)
    cos_theta = torch.cos(theta).unsqueeze(1)
    sin_phi = torch.sin(phi).unsqueeze(1)
    cos_phi = torch.cos(phi).unsqueeze(1)
    
    xi = center[..., 0].unsqueeze(1) + rs * sin_theta * cos_phi
    yi = center[..., 1].unsqueeze(1) + rs * sin_theta * sin_phi
    zi = center[..., 2].unsqueeze(1) + rs * cos_theta
    
    # Stack interpolated points
    interpolated_points = torch.stack([xi, yi, zi], dim=-1).reshape(vertices.shape[0], -1, 3)
    
    return interpolated_points


def field_boundary(point_cloud, geometric_center, grid_shape = (192, 192, 192), sigma_inner=5, sigma_outer=0.1):
    # Assuming interpolate_pc() interpolates the point cloud
  
    
    # Define a 3D grid
   
    grid = np.indices(grid_shape).reshape(3, -1).T

    # Create a KD-tree for efficient nearest-neighbor search
    tree = cKDTree(point_cloud)

    # Find the distance to the closest point for each voxel
    nearest_neighbor_distances, nearest_neighbor_indices = tree.query(grid)

    # Calculate radial distances from each point in the grid to the geometric center
    radial_distances = np.linalg.norm(grid - geometric_center, axis=1)

    # Get the radial distances from the geometric center to each nearest point in the point cloud
    nearest_point_radial_distances = np.linalg.norm(point_cloud[nearest_neighbor_indices] - geometric_center, axis=1)

    # Initialize Gaussian decay based on distance comparison
    gaussian_weights = np.exp(-nearest_neighbor_distances**2 / (2 * sigma_inner**2))
    outside_weights = np.exp(-nearest_neighbor_distances**2 / (2 * sigma_outer**2))

    # Determine where the grid points are outside the nearest point radial distances
    outside_mask = radial_distances > nearest_point_radial_distances

    # Combine weights: Apply sharper decay (outside_weights) where points are outside the boundary
    gaussian_weights[outside_mask] = outside_weights[outside_mask]

    # Convert the Gaussian weights into a 3D array
    weighted_distance_field = gaussian_weights.reshape(grid_shape)

    # Normalize the weighted distance field if necessary
    normalized_weighted_distance_field = weighted_distance_field / np.max(weighted_distance_field)

    return normalized_weighted_distance_field


def calculate_normals(points, n_phi=64, n_theta=64):
    normals = np.zeros((n_phi, n_theta, 3), dtype=np.float64)  # Explicit dtype declaration
    # Reshape points for easier indexing
    points = points.reshape((n_phi, n_theta, 3))
    
    for i in range(n_phi):
        for j in range(n_theta):
            # Calculate indices of the neighbors using modulo for wrap-around at boundaries
            i_prev = (i - 1) % n_phi
            i_next = (i + 1) % n_phi
            j_prev = (j - 1) % n_theta
            j_next = (j + 1) % n_theta
            
            # Get adjacent points to form vectors
            p = points[i, j]
            p_phi_prev = points[i_prev, j]
            p_phi_next = points[i_next, j]
            p_theta_prev = points[i, j_prev]
            p_theta_next = points[i, j_next]
            
            # Vectors in the phi and theta directions
            v_phi = p_phi_next - p_phi_prev
            v_theta = p_theta_next - p_theta_prev
            
            # Cross product of vectors
            normal = np.cross(v_phi, v_theta)
            normal = normal / np.linalg.norm(normal)  # Normalize to prevent division by zero or NaN values
            
            normals[i, j] = normal
    
    return normals.reshape((-1, 3))

# Modify the function to use multiple nearest points
def calculate_field_map(point_cloud, geometric_center, grid_shape=(192, 192, 192), k=7):
    normals = calculate_normals(point_cloud)
    
    grid = np.indices(grid_shape).reshape(3, -1).T + geometric_center - np.array(grid_shape) // 2
    tree = cKDTree(point_cloud)
    distances, indices = tree.query(grid, k=k)  # Query multiple nearest points
    nearest_points = point_cloud[indices]
    nearest_normals = normals[indices]

    # Calculate weighted average of normals
    weights = 1 / (distances + 1e-10)  # Avoid division by zero
    weighted_normals = np.sum(nearest_normals * weights[:,:,None], axis=1) / np.sum(weights, axis=1)[:,None]
    
    vectors_to_nearest = nearest_points[:,0,:] - grid  # Use only the closest for vector calculation
    vectors_to_nearest_normalized = vectors_to_nearest / np.linalg.norm(vectors_to_nearest, axis=1, keepdims=True)
    dot_products = np.einsum('ij,ij->i', weighted_normals, vectors_to_nearest_normalized)

#     field_strength = np.abs(dot_products)
    field_strength = dot_products
#     Apply a sigmoid or similar function to scale dot products
#     field_strength = 1 / (1 + np.exp(-dot_products))  # Sigmoid function for smoothing effect

    field_map = field_strength / np.max(field_strength)
    field_map = field_map.reshape(grid_shape)
    
    return field_map

def map_points_to_nearest_grid_by_distance_torch(coor, point_cloud, grid_size=(192, 192, 192)):
    """
    Map a batched point cloud to the nearest grid points in a 3D grid based on Euclidean distance using PyTorch.
    """
    batch_size, _, num_points = coor.shape
    num_features = point_cloud.shape[1]
    device = coor.device

    # Create an empty grid of the specified size for each batch and feature
    grid = torch.zeros((batch_size, num_features) + grid_size, dtype=torch.float32, device=device)
    grid_counts = torch.zeros((batch_size, num_features) + grid_size, dtype=torch.float32, device=device)

    # Transpose coordinates to have them in shape (batch_size, num_points, 3)
    coords = coor.permute(0, 2, 1)

    # Create a 3x3x3 grid of offsets for the x, y, z dimensions around a central point
    offsets = torch.stack(torch.meshgrid([torch.tensor([-1, 0, 1]) for _ in range(3)], indexing='ij')).to(device)
    offsets = offsets.reshape(3, 27).permute(1, 0)  # Reshape and permute to get shape (27, 3)

    # Generate a tensor for clamping within grid bounds
    min_val = torch.tensor([0, 0, 0], device=device)
    max_val = torch.tensor([grid_size[0] - 1, grid_size[1] - 1, grid_size[2] - 1], device=device)

    # Generate neighborhood points by adding offsets
    rounded_points = coords.round().long()
#     neighborhood = rounded_detailly ensure they lie within the tracked points by adding offsets
    neighborhood = rounded_points[:, :, None, :] + offsets[None, None, :, :]  # Shape (batch_size, num_points, 27, 3)

    # Clamp the points and their neighborhoods to ensure they lie within the grid
    neighborhood = torch.clamp(neighborhood, min_val, max_val)
    distances = torch.sqrt(((coords[:, :, None, :] - neighborhood.float()) ** 2).sum(dim=-1))

    # Get the indices of the closest points
    min_distance_indices = distances.argmin(dim=2)

    # Get the actual grid positions of the closest points
    closest_points = neighborhood.gather(2, min_distance_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 3))
    closest_points = closest_points.squeeze(2)
 
    # assign values to the grid based on the point cloud
    for b in range(batch_size):
        for n in range(num_points):
            # Using all features for each closest point found
            grid[b, :, closest_points[b, n, 0], closest_points[b, n, 1], closest_points[b, n, 2]] = point_cloud[b, :, n]
            grid_counts[b, :, closest_points[b, n, 0], closest_points[b, n, 1], closest_points[b, n, 2]] += 1
    grid_counts[grid_counts == 0] = 1
    grid_avg = grid / grid_counts
    return grid_avg


class STNkd(nn.Module):
    def __init__(self, k=6, output_size=1):
        super(STNkd, self).__init__()
        self.conv1 = nn.Conv1d(k, k*k, 1)
        self.conv2 = nn.Conv1d(k*k, 256, 1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, k*k)
        self.relu = nn.ReLU()


        self.k = k
        
        # Define an adaptive pooling layer
        self.adaptive_pool = nn.AdaptiveMaxPool1d(output_size)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Use adaptive pooling to ensure a fixed size output
        x = self.adaptive_pool(x)  # This replaces the torch.max operation
        x = x.view(-1, 256)  # Adjust the flattening based on the output of adaptive pooling
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k*self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x
class PointNetfeat(nn.Module):
    def __init__(self, f_dim=6, K=64, feature_transform = True):
        super(PointNetfeat, self).__init__()
        self.stn = STNkd(k=f_dim)
        self.conv1 = torch.nn.Conv1d(f_dim, K, 1)
        self.conv2 = torch.nn.Conv1d(K, K*2, 1)
        self.conv3 = torch.nn.Conv1d(K*2, K, 1)

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
        x = F.relu(self.conv1(x))
     
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)

        return x


def spherical_to_cartesian_torch(r, theta, phi):
    x = r * torch.sin(phi) * torch.cos(theta)
    y = r * torch.sin(phi) * torch.sin(theta)
    z = r * torch.cos(phi)
    return x, y, z
def reverse_normal(r,min_,max_):
    return (max_-min_)*r +min_

def radius_to_cartesian_torch_batch_m(radii, g_center,normal_min_r=12.849446398332546, normal_max_r=108.08587944724654, n_pc=4096, n_regions_phi=64, n_regions_theta=64):
    device = radii.device
    
    batch_size = radii.size(0)
    # Reshape radii to [batch_size, n_regions_phi, n_regions_theta]
    radii = radii.view(batch_size, n_regions_phi, n_regions_theta)
    g_center = g_center.view(batch_size,1, 3).to(device)
    
    phi_bins = torch.linspace(0, np.pi, n_regions_phi + 1)
    theta_bins = torch.linspace(-np.pi, np.pi, n_regions_theta + 1)

    cartesian_coords_batch = torch.zeros((batch_size, n_regions_phi, n_regions_theta, 3))

    for b in range(batch_size):
        for i in range(n_regions_phi):
            for j in range(n_regions_theta):
                phi_center = (phi_bins[i] + phi_bins[i + 1]) / 2
                theta_center = (theta_bins[j] + theta_bins[j + 1]) / 2
                
                radius = reverse_normal(radii[b, i, j], normal_min_r, normal_max_r)
              
                x, y, z = spherical_to_cartesian_torch(radius, theta_center, phi_center)
                cartesian_coords_batch[b, i, j] = torch.tensor([x, y, z])
    
    coor_est = cartesian_coords_batch.view(batch_size, n_pc, 3).to(device)
    coor_est +=  g_center
    coor_est = coor_est
    return coor_est


def pre_mask(point_cloud, geometric_center, sigma_inner=5, sigma_outer=0.1):
    # Assuming interpolate_pc() interpolates the point cloud
  
    
    # Define a 3D grid
    grid_shape = (192, 192, 192)
    grid = np.indices(grid_shape).reshape(3, -1).T

    # Create a KD-tree for efficient nearest-neighbor search
    tree = cKDTree(point_cloud)

    # Find the distance to the closest point for each voxel
    nearest_neighbor_distances, nearest_neighbor_indices = tree.query(grid)

    # Calculate radial distances from each point in the grid to the geometric center
    radial_distances = np.linalg.norm(grid - geometric_center, axis=1)

    # Get the radial distances from the geometric center to each nearest point in the point cloud
    nearest_point_radial_distances = np.linalg.norm(point_cloud[nearest_neighbor_indices] - geometric_center, axis=1)

    # Initialize Gaussian decay based on distance comparison
    gaussian_weights = np.exp(-nearest_neighbor_distances**2 / (2 * sigma_inner**2))
    outside_weights = np.exp(-nearest_neighbor_distances**2 / (2 * sigma_outer**2))

    # Determine where the grid points are outside the nearest point radial distances
    outside_mask = radial_distances > nearest_point_radial_distances

    # Combine weights: Apply sharper decay (outside_weights) where points are outside the boundary
    gaussian_weights[outside_mask] = outside_weights[outside_mask]

    # Convert the Gaussian weights into a 3D array
    weighted_distance_field = gaussian_weights.reshape(grid_shape)

    # Normalize the weighted distance field if necessary
    nwdf = weighted_distance_field / np.max(weighted_distance_field)
    nwdf[nwdf<0.5] = 0
    nwdf[nwdf>=0.5] = 1

    structure = generate_binary_structure(3, 2)  # Define the structure for closing
    connected_binary_mask = binary_dilation(nwdf, structure=structure, iterations=3).astype(int)
    filled_binary_mask = binary_fill_holes(connected_binary_mask).astype(int)
    filled_binary_mask = binary_closing(filled_binary_mask, structure=structure).astype(int)

    return filled_binary_mask


def pre_insert(point_cloud, geometric_center, grid_shape=(192, 192, 192)):
    # Generate grid coordinates
    x = np.arange(grid_shape[0])
    y = np.arange(grid_shape[1])
    z = np.arange(grid_shape[2])
    grid_coords = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1).reshape(-1, 3)

    # Create a KD-tree for the grid
    grid_tree = cKDTree(grid_coords)

    # Query the grid tree to find the indices of the closest grid points for each point in the point cloud
    distances, indices = grid_tree.query(point_cloud)

    # Get the grid indices
    closest_grid_points = grid_coords[indices].astype(int)

    # Initialize the binary mask
    mask = np.zeros(grid_shape, dtype=int)

    # Remove any indices that are out of bounds (shouldn't be necessary but added for safety)
    valid_mask = np.all((closest_grid_points >= 0) & (closest_grid_points < np.array(grid_shape)), axis=1)
    closest_grid_points = closest_grid_points[valid_mask]

    # Set the corresponding grid points in the mask to 1
    mask[closest_grid_points[:, 0], closest_grid_points[:, 1], closest_grid_points[:, 2]] = 1

    return mask



def pre_field(point_cloud, geometric_center, sigma_inner=5, sigma_outer=0.1):
    # Assuming interpolate_pc() interpolates the point cloud
  
    
    # Define a 3D grid
    grid_shape = (192, 192, 192)
    grid = np.indices(grid_shape).reshape(3, -1).T

    # Create a KD-tree for efficient nearest-neighbor search
    tree = cKDTree(point_cloud)

    # Find the distance to the closest point for each voxel
    nearest_neighbor_distances, nearest_neighbor_indices = tree.query(grid)

    # Calculate radial distances from each point in the grid to the geometric center
    radial_distances = np.linalg.norm(grid - geometric_center, axis=1)

    # Get the radial distances from the geometric center to each nearest point in the point cloud
    nearest_point_radial_distances = np.linalg.norm(point_cloud[nearest_neighbor_indices] - geometric_center, axis=1)

    # Initialize Gaussian decay based on distance comparison
    gaussian_weights = np.exp(-nearest_neighbor_distances**2 / (2 * sigma_inner**2))
    outside_weights = np.exp(-nearest_neighbor_distances**2 / (2 * sigma_outer**2))

    # Determine where the grid points are outside the nearest point radial distances
    outside_mask = radial_distances > nearest_point_radial_distances

    # Combine weights: Apply sharper decay (outside_weights) where points are outside the boundary
    gaussian_weights[outside_mask] = outside_weights[outside_mask]

    # Convert the Gaussian weights into a 3D array
    weighted_distance_field = gaussian_weights.reshape(grid_shape)

    # Normalize the weighted distance field if necessary
    nwdf = weighted_distance_field / np.max(weighted_distance_field)
    nwdf[nwdf<0.5] = 0
    nwdf[nwdf>=0.5] = 1


    return nwdf


class Layer_refine_light(nn.Module):
    def __init__(self, planes, dropout=0.1, norm='bn', padding=0, deep_fuse=False, method='add'):
        super(Layer_refine_light, self).__init__()

        self.maxpool = nn.MaxPool3d(2, 2, padding=padding)
        self.dropout = dropout

        self.relu = nn.LeakyReLU(0.2, inplace=False)
        self.conv_x1 = nn.Conv3d(planes, planes//4, 3, 1, 1, bias=False)
        self.bn1 = normalization(planes//4, norm)

        self.conv_x2 = nn.Conv3d(planes//4, planes//8, 1, 1, 0, bias=False)
        self.bn2 = normalization(planes//8, norm)

        self.conv_g = nn.Conv3d(1, planes//8, 1, 1, 0, bias=False)
        self.map_back = nn.Conv3d(planes//8, planes,1, 1,0)
        self.layer_seg = nn.Conv3d(planes, 1, 1)
        self.sig = nn.Sigmoid()
        self.method = method

       
        self.deep_fuse = deep_fuse
        if self.deep_fuse:
            self.bn_deep = normalization(planes//8, norm)
            self.conv_xg = nn.Conv3d(planes//4, planes//8, 3, 1, 1, bias=False)
        if (self.method =='gate') or (self.method =='gate_v2'):
             self.gate_conv = nn.Conv3d(planes//8, planes//8, 1, 1, 0)

    def adain(self, content, style):
        size = content.size()
        content_mean, content_std = content.view(size[0], -1).mean(dim=1), content.view(size[0], -1).std(dim=1)
        style_mean, style_std = style.view(size[0], -1).mean(dim=1), style.view(size[0], -1).std(dim=1)
        normalized_content = (content - content_mean.view(size[0], 1, 1, 1, 1)) / content_std.view(size[0], 1, 1, 1, 1)
        styled_content = normalized_content * style_std.view(size[0], 1, 1, 1, 1) + style_mean.view(size[0], 1, 1, 1, 1)
        return styled_content

    def forward(self, x, g):

        x = self.relu(self.bn1(self.conv_x1(x)))
        x = self.relu(self.bn2(self.conv_x2(x)))
        if self.dropout > 0:
            g = F.dropout3d(g, self.dropout)
        g = self.relu(self.bn2(self.conv_g(g)))

        if self.deep_fuse:
            xg = torch.cat([x, g], dim=1)
            g = self.relu(self.bn_deep(self.conv_xg(xg)))
        
        g = self.adain(g, x)    

        if self.method =='add':
            f_map = self.relu(g + x)
        elif self.method =='mul':
            f_map = self.relu(g * x)
        elif self.method =='hybrid':
            f_map = self.relu(g + g*x)
        elif self.method == 'gate':
            f_map = self.relu(g + self.sig(self.gate_conv(x)))
        elif self.method == 'gate_v2':
            f_map = self.relu(g + self.sig(self.gate_conv(x))*x)

        f_map = self.map_back(f_map)
        f_seg = self.layer_seg(f_map)
        f_seg = self.sig(f_seg)
        return f_map, f_seg


class GINet_(nn.Module):
    def __init__(self, n_layers=5, c=1, n=8, padding_list=[0, 0, 0, 0, 0, 0, 0], norm='bn', dropout=0.1,method ='gate_v2',insert_position=[0,1,2,3,4],set_sdf=False,n_pc = 4096):
        super(GINet_, self).__init__()
        self.deep_fuse = True
        self.method = method
        self.set_sdf = set_sdf
        self.middle_channel = 2 ** (n_layers) * n
        self.dropout = dropout
        self.padding_list = padding_list
        self.n_layers = n_layers
     
        self.pooling = nn.MaxPool3d(2, 2, 0)
        self.insert_positions = insert_position
        self.n_pc = n_pc
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
        # Define layers for the refined pass
        self.convd_list_refined = nn.ModuleList([
            ConvD(c, n, dropout, norm, first=True) if i == 0 else
            ConvD(2 ** (i - 1) * n, 2 ** i * n, dropout, norm, padding=padding_list[i - 1])
            for i in range(n_layers + 1)
        ])
        self.convu_list_refined = nn.ModuleList([
            ConvU(2 ** (i + 1) * n, dropout, norm, first=(i == n_layers - 1), padding=padding_list[i])
            for i in reversed(range(n_layers))
        ])


        self.seg = nn.Conv3d(n, 1, 1)
        self.sig = nn.Sigmoid()


        self.refine_list = nn.ModuleList([
            Layer_refine_light(2 ** (i ) * n, self.dropout, norm, deep_fuse = self.deep_fuse,method=self.method)
            for i in reversed(range(n_layers))
        ])
        self.middle_refine = Layer_refine_light(2 ** (5 ) * n, self.dropout, norm, deep_fuse = self.deep_fuse,method=self.method)
            
        
        self.pc_evolve = PCModel_wr_v3(n_pc = self.n_pc)
       

        # Initialize the weights for both sets of layers
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def load_pretrained_weights(self, path):
            # Load the model
            saved_state_dict = torch.load(path, map_location='cpu')

            # Prepare the current model's state dictionary
            model_state_dict = self.state_dict()

            # Track which parameters are loaded
            loaded_params = set()

            # Filter out unnecessary keys from the loaded state dictionary
            for name in list(saved_state_dict.keys()):  # Use list to copy keys
                if name in model_state_dict:
                    if model_state_dict[name].size() == saved_state_dict[name].size():
                        loaded_params.add(name)
                    else:
                        print(f"Skipping {name} due to size mismatch: Model expects {model_state_dict[name].size()}, loaded state provides {saved_state_dict[name].size()}.")
                        saved_state_dict.pop(name)
                else:
    #                 print(f"Skipping {name} as it's not in the current model's state dictionary.")
                    pass

            # Update current model's state dictionary with the filtered loaded state dictionary
            model_state_dict.update(saved_state_dict)

            # Load the updated state dictionary back to the model
            self.load_state_dict(model_state_dict, strict=False)

            return loaded_params

    def forward(self, x, coors_tem, g_center):
        # start_time = time.time()
        device = x.device
        g_center = g_center.detach()  # Ensure no gradients are expected for g_center in subsequent operations
     
        # First pass using initial layers, handling the entire batch
        y_initial = self.process_layers(self.convd_list, self.convu_list, x)
        
        # Generate point clouds for the whole batch
        r_out = self.pc_evolve(coors_tem, y_initial)
      
        g = radius_to_cartesian_torch_batch_m(r_out, g_center,n_pc=self.n_pc, n_regions_phi= int(self.n_pc**0.5) , n_regions_theta=int(self.n_pc**0.5))

        # Placeholder for processed g_in tensors
        g_in_list = []

        # Process each point cloud in the batch using pre_mask
        for i in range(g.shape[0]):  # Iterate over the batch dimension
            g_np = g[i].cpu().detach().numpy()  # Convert to NumPy
            g_center_np = g_center[i].cpu().numpy()  # Convert to NumPy

            if  self.set_sdf:
                g_in_np = pre_field(g_np, g_center_np) 
            else:
                g_in_np = pre_mask(g_np, g_center_np)  # Apply the pre_mask function
            # Convert result back to PyTorch tensor and store in list
            g_in_torch = torch.from_numpy(g_in_np).to(device).float()
            g_in_torch.requires_grad_()  # Ensure that gradients are tracked
            g_in_list.append(g_in_torch[None, :, :, :])


        # Stack processed tensors back into a batch
        g_in = torch.stack(g_in_list)
        g_in_detached = g_in.detach() 
        # Second pass using refined layers with g_in, handling the entire batch
        y_refined, y_seg = self.process_layers(self.convd_list_refined, self.convu_list_refined, x, g_in_detached)
        # elapsed_time = time.time() - start_time
        # print(f"Point cloud generation time for n_pc = {self.n_pc}: {elapsed_time:.4f} seconds")
        return r_out, y_refined, y_seg


    def process_layers(self, convd_list, convu_list,x, g=None):
        xs = []
        gs = []
        segs = []
        ys = []
        
        for i, conv in enumerate(convd_list):
            x = conv(x)
            if i != len(convd_list) - 1:
                xs.append(x)
                if g!=None:
                    gs.append(g)  
                    g  = self.pooling(g)
        y = x
        if g!=None:
            y,seg = self.middle_refine(y,g)
            segs.append(seg)
        for i, (convu, refine) in enumerate(zip(convu_list, self.refine_list)):
            y = convu(y, xs[::-1][i])
            if g!=None:
                if i in self.insert_positions:
                    y,seg = refine(y,gs[::-1][i])
                    if i !=4:
                        segs.append(seg)
                    else:
                        y = seg
            ys.append(y)
            
        if g is not None:
            return y,segs
        else:
            return ys[::-1]
        