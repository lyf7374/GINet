
import nibabel as nib
import numpy as np


def read_nifti_file(filepath,re_ori = True, tumor=False):
    """Read and load volume"""
    scan = nib.load(filepath)
    if re_ori == True:
        scan = nib.as_closest_canonical(scan)
    scan = scan.get_fdata()    
    return scan

def normalize(volume, method = 'mm'):
    """Normalize the volume
       method: "zs": z-std normalization
               "mm": min max normalization
    
    """
    if method == 'zs':
        mean = np.mean(volume)
        std = np.std(volume)
        volume = (volume - mean)/std
        volume = volume.astype("float32")
        
    elif method =='mm':
    #     min = -1000
    #     max = 400
        min = np.min(volume)
        max = np.max(volume)
    #     volume[volume < min] = min
    #     volume[volume > max] = max
        volume = (volume - min) / (max - min)
        volume = volume.astype("float32")
    elif method =='special':
        ana = volume[volume!=0]
        min = np.min(ana)
        max = np.max(ana)
        volume[volume!=0] = (volume[volume!=0] - min) / (max - min)

        
        
    return volume

def resize_volume(path, center=None, output_shape=(228, 228, 228), tumor=False,re_ori = True):
    img = read_nifti_file(path, tumor=tumor,re_ori = re_ori)
    # Ensure center is within the bounds of the image
    if (center ==None).any():
        center = (output_shape[0]//2,output_shape[1]//2,output_shape[2]//2)
    else:
        center = [max(0, min(img.shape[i], center[i])) for i in range(3)]

    # Calculate start and end indices for cropping
    half_shape = [os // 2 for os in output_shape]
    start = [max(0, center[i] - half_shape[i]) for i in range(3)]
    end = [min(img.shape[i], center[i] + half_shape[i]) for i in range(3)]

    # Adjust start and end indices if they go beyond the image boundaries
    for i in range(3):
        if end[i] - start[i] < output_shape[i]:
            if start[i] == 0:
                end[i] = min(img.shape[i], start[i] + output_shape[i])
            elif end[i] == img.shape[i]:
                start[i] = max(0, end[i] - output_shape[i])

    # Crop the image
    img = img[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

    # Calculate padding if necessary
    diff = [output_shape[i] - (end[i] - start[i]) for i in range(3)]
    padding = [(diff[i] // 2, diff[i] - diff[i] // 2) for i in range(3)]

    # Apply padding
    img = np.pad(img, padding)

    return img

def process_scan(path, mask = False, resize = True,norm_method = 'mm',output_shape=(228,228,228),center=None,re_ori = True):
    """Read and resize volume"""
    # Read scan
#     volume = read_nifti_file(path)
    if resize == True:
        volume = resize_volume(path,output_shape =output_shape,center=center,re_ori = re_ori)
    else:
        volume = read_nifti_file(path)
    
    # Resize width, height and depth      
    # Normalize
    if mask == False:
        volume = normalize(volume, method = norm_method)
    else:
#         volume[volume!=1] = 0
        pass

    if mask == True:
        volume[volume>0.5] = 1
        volume[volume<=0.5] = 0

    return volume



def resize_applymask(path_skull,path_mask, center=None, output_shape=(228, 228, 228), tumor=False):
    img = read_nifti_file(path_skull, tumor=tumor)
    mask = read_nifti_file(path_mask, tumor=tumor)
    img = img*mask

    # Ensure center is within the bounds of the image
    if (center ==None).any():
        center = (output_shape[0]//2,output_shape[1]//2,output_shape[2]//2)
    else:
        center = [max(0, min(img.shape[i], center[i])) for i in range(3)]

    # Calculate start and end indices for cropping
    half_shape = [os // 2 for os in output_shape]
    start = [max(0, center[i] - half_shape[i]) for i in range(3)]
    end = [min(img.shape[i], center[i] + half_shape[i]) for i in range(3)]

    # Adjust start and end indices if they go beyond the image boundaries
    for i in range(3):
        if end[i] - start[i] < output_shape[i]:
            if start[i] == 0:
                end[i] = min(img.shape[i], start[i] + output_shape[i])
            elif end[i] == img.shape[i]:
                start[i] = max(0, end[i] - output_shape[i])

    # Crop the image
    img = img[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

    # Calculate padding if necessary
    diff = [output_shape[i] - (end[i] - start[i]) for i in range(3)]
    padding = [(diff[i] // 2, diff[i] - diff[i] // 2) for i in range(3)]

    # Apply padding
    img = np.pad(img, padding)

    return img

def process_scan_applymask(path_skull,path_mask,norm_method = 'mm',output_shape=(228,228,228),center=np.array([98,109,81])):
    """Read and resize volume"""
    # Read scan
#     volume = read_nifti_file(path)

    volume = resize_applymask(path_skull,path_mask,output_shape =output_shape,center=center)
    volume = normalize(volume, method = norm_method)
    return volume