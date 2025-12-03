"""
Module to perform brain MRI segmentation using a pre-trained network.

This module contains utility functions to load and preprocess NIfTI volumes,
resize volumes to a fixed shape, estimate brain center via image registration,
prepare geometric templates for the network, run the segmentation model, undo
the resize operation to restore original volume dimensions, and finally save
the segmentation mask back to disk.

The functions avoid Python 3.10-style union type hints (e.g. ``np.ndarray | None``)
to maintain compatibility with earlier Python versions. Where necessary,
``typing.Optional`` and friends are used instead.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import nibabel as nib
import torch
import ants

from utils.Gsupport import (
    sub_sampling_blockwise,
    convert2GI,
    normalize_radius,
    model_load,
)
from utils.GGsupport import seq_Gnet_v5


def read_nifti_file(filepath: str, re_ori: bool = True) -> np.ndarray:
    """Read and load a NIfTI volume.

    Args:
        filepath: Path to the NIfTI file.
        re_ori: If True, reorient the image to canonical orientation.

    Returns:
        The image data as a NumPy array.
    """
    scan = nib.load(filepath)
    if re_ori:
        scan = nib.as_closest_canonical(scan)
    return scan.get_fdata()


def normalize(volume: np.ndarray, method: str = "mm") -> np.ndarray:
    """Normalize a volume using z-score or min-max normalization.

    Args:
        volume: The input volume as a NumPy array.
        method: Normalization method: "zs" for z-score or "mm" for min-max.

    Returns:
        The normalized volume as a NumPy array.
    """
    volume = volume.astype(np.float32)
    if method == "zs":
        mean = float(np.mean(volume))
        std = float(np.std(volume))
        if std > 0:
            volume = (volume - mean) / std
    elif method == "mm":
        vmin = float(np.min(volume))
        vmax = float(np.max(volume))
        if vmax > vmin:
            volume = (volume - vmin) / (vmax - vmin)
    return volume.astype(np.float32)


def resize_volume_new(
    img_pth: str,
    center: Optional[np.ndarray] = None,
    output_shape: Tuple[int, int, int] = (192, 192, 192),
    re_ori: bool = True,
    return_transform: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Crop and pad a 3D image to a target shape.

    The image is cropped around a specified center and padded as necessary
    to reach the desired ``output_shape``. When ``return_transform`` is True,
    the function also returns a dictionary containing parameters needed to
    undo the resize operation later.

    Args:
        img_pth: Path to the NIfTI file.
        center: Optional (z, y, x) center. If None, uses the image center.
        output_shape: Desired output shape as a (z, y, x) tuple.
        re_ori: If True, reorient the image to canonical orientation.
        return_transform: If True, return transformation info alongside the volume.

    Returns:
        A tuple ``(img_padded, transform)`` where ``img_padded`` is the resized
        volume and ``transform`` is a dictionary describing how to undo the
        resize. If ``return_transform`` is False, only ``img_padded`` is
        returned and ``transform`` is an empty dict.
    """
    img = read_nifti_file(img_pth, re_ori=re_ori)
    orig_shape = img.shape
    target_shape = list(output_shape)

    # Determine cropping center
    if center is None:
        crop_center = [s // 2 for s in orig_shape]
    else:
        center_arr = np.asarray(center, dtype=int).ravel()
        if center_arr.size < 3:
            raise ValueError(
                f"center must have at least 3 elements (z, y, x); got {center_arr.size}"
            )
        crop_center = center_arr[:3].tolist()

    half_shape = [s // 2 for s in target_shape]

    # Compute tentative start/end indices (can be negative or exceed dims)
    start: List[int] = [crop_center[i] - half_shape[i] for i in range(3)]
    end: List[int] = [start[i] + target_shape[i] for i in range(3)]

    # Amount of padding required before/after in each dimension
    pad_before: List[int] = [max(0, -start[i]) for i in range(3)]
    pad_after: List[int] = [max(0, end[i] - orig_shape[i]) for i in range(3)]

    # Clip cropping indices to valid ranges
    clipped_start: List[int] = [max(0, start[i]) for i in range(3)]
    clipped_end: List[int] = [min(orig_shape[i], end[i]) for i in range(3)]

    # Crop the image
    img_cropped = img[
        clipped_start[0] : clipped_end[0],
        clipped_start[1] : clipped_end[1],
        clipped_start[2] : clipped_end[2],
    ]

    # Pad to target_shape
    padding: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]] = (
        (pad_before[0], pad_after[0]),
        (pad_before[1], pad_after[1]),
        (pad_before[2], pad_after[2]),
    )
    img_padded = np.pad(img_cropped, padding, mode="constant", constant_values=0)

    if not return_transform:
        return img_padded.astype(np.float32), {}

    transform: Dict[str, Any] = {
        "orig_shape": orig_shape,
        "pad_before": pad_before,
        "pad_after": pad_after,
        "clipped_start": clipped_start,
        "clipped_end": clipped_end,
        "output_shape": target_shape,
        "center": crop_center,
    }
    return img_padded.astype(np.float32), transform


def process_scan_with_transform(
    path: str,
    mask: bool = False,
    resize: bool = True,
    norm_method: str = "mm",
    output_shape: Tuple[int, int, int] = (192, 192, 192),
    center: Optional[np.ndarray] = None,
    re_ori: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load, normalize, and optionally resize a NIfTI volume.

    Args:
        path: Path to the NIfTI file to read.
        mask: If True, treat the volume as a binary mask and skip normalization.
        resize: If True, crop/pad the volume to ``output_shape``.
        norm_method: Normalization method: 'zs' or 'mm'.
        output_shape: Target shape for resizing when ``resize`` is True.
        center: Optional center for cropping; defaults to image center.
        re_ori: If True, reorient the image to canonical orientation.

    Returns:
        A tuple ``(volume, transform)``. ``volume`` is the processed volume and
        ``transform`` contains parameters to undo the resize (even when
        ``resize`` is False).
    """
    if resize:
        volume, transform = resize_volume_new(
            path,
            center=center,
            output_shape=output_shape,
            re_ori=re_ori,
            return_transform=True,
        )
    else:
        volume = read_nifti_file(path, re_ori=re_ori)
        transform = {
            "orig_shape": volume.shape,
            "pad_before": [0, 0, 0],
            "pad_after": [0, 0, 0],
            "clipped_start": [0, 0, 0],
            "clipped_end": list(volume.shape),
            "output_shape": list(volume.shape),
            "center": None,
        }

    # Normalize unless this is a binary mask
    if not mask:
        volume = normalize(volume, method=norm_method)
    else:
        volume = (volume > 0.5).astype(np.float32)

    return volume, transform


def undo_resize(mask_resized: np.ndarray, transform: Dict[str, Any]) -> np.ndarray:
    """Undo the crop/pad operation and restore the original image shape.

    Args:
        mask_resized: The resized volume of shape ``output_shape``.
        transform: The transform dictionary returned by :func:`resize_volume_new`.

    Returns:
        A volume of shape ``orig_shape``, with the resized region placed
        back into its original location.
    """
    orig_shape = transform["orig_shape"]
    pad_before = transform["pad_before"]
    pad_after = transform["pad_after"]
    clipped_start = transform["clipped_start"]
    clipped_end = transform["clipped_end"]
    output_shape = transform["output_shape"]

    # Convert to numpy array in case it's a tensor
    mask_resized = np.asarray(mask_resized)

    # Remove the padding applied during resize
    z0 = pad_before[0]
    z1 = output_shape[0] - pad_after[0]
    y0 = pad_before[1]
    y1 = output_shape[1] - pad_after[1]
    x0 = pad_before[2]
    x1 = output_shape[2] - pad_after[2]

    mask_cropped = mask_resized[z0:z1, y0:y1, x0:x1]

    # Create an empty array and insert the cropped region back
    mask_orig = np.zeros(orig_shape, dtype=mask_resized.dtype)
    mask_orig[
        clipped_start[0] : clipped_end[0],
        clipped_start[1] : clipped_end[1],
        clipped_start[2] : clipped_end[2],
    ] = mask_cropped

    return mask_orig


def find_center(
    img_pth: str,
    template_img_path: str,
    mask_template_img_path: str,
) -> np.ndarray:
    """Estimate the brain center via image registration.

    The input image is registered to a standard template using ANTsPy,
    and the median of the transformed brain mask voxels is returned as
    the center.

    Args:
        img_pth: Path to the input NIfTI image.
        template_img_path: Path to the template image used for registration.
        mask_template_img_path: Path to the template brain mask.

    Returns:
        The estimated center as a (z, y, x) integer NumPy array.
    """
    template_img_ants = ants.image_read(template_img_path, reorient="LPI")
    mask_template_img_ants = ants.image_read(mask_template_img_path, reorient="LPI")

    raw_img_ants = ants.image_read(img_pth, reorient="LPI")
    transformation = ants.registration(
        fixed=raw_img_ants,
        moving=template_img_ants,
        type_of_transform="SyN",
        verbose=False,
    )

    # Warp the mask to the input image space
    brain_mask = ants.apply_transforms(
        fixed=transformation["warpedmovout"],
        moving=mask_template_img_ants,
        transformlist=transformation["fwdtransforms"],
        interpolator="nearestNeighbor",
        verbose=False,
    )

    mask_np = brain_mask.numpy()
    center = np.median(np.argwhere(mask_np > 0.5), axis=0).astype(int)
    return center


def prepare_gi_template(
    n_patch: int,
    t_patch: int,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """Load the geometric template and prepare it for the model.

    Args:
        n_patch: Number of patches along the polar angle dimension.
        t_patch: Number of patches selected when sampling the spherical grid.

    Returns:
        A tuple ``(GI_tem, center_g_t, indices, center_g_np)`` where
        ``GI_tem`` is the geometry tensor of shape (1, 4, n_selected),
        ``center_g_t`` is the center tensor of shape (1, 3, 1),
        ``indices`` are the sampled point indices, and ``center_g_np``
        is the center of the spherical template in NumPy format.
    """
    n_pc = 4096
    g_tem = np.load(f"data_utils/cGI_tem_{n_pc}rpt.npy")  # (n_pc, 3)
    center_g_np = np.load(f"data_utils/gcenter_tem_{n_pc}rpt.npy")  # (3,)

    # Add batch dimension
    g_tem = np.expand_dims(g_tem, axis=0)  # (1, n_pc, 3)
    g_tem_nor = g_tem - center_g_np[np.newaxis, np.newaxis, :]
    GI_tem = convert2GI(g_tem_nor, n_patch)  # (1, n_pc, 4)
    GI_tem[:, :, :3] += center_g_np[np.newaxis, np.newaxis, :]

    # Normalize radius
    min_r = 12.849446398332546
    max_r = 108.08587944724654
    GI_tem = normalize_radius(GI_tem, min_r=min_r, max_r=max_r)

    # Determine indices to sample from the point cloud
    indices, _ = sub_sampling_blockwise(
        n_regions_phi=n_patch,
        n_regions_theta=n_patch,
        sample_regions_phi=t_patch,
        sample_regions_theta=n_patch,
        samples_per_block=1,
    )

    # Convert to torch tensors
    GI_tem = torch.tensor(GI_tem, dtype=torch.float32)
    GI_tem = GI_tem.transpose(1, 2)[:, :, indices]
    center_g_t = torch.tensor(center_g_np, dtype=torch.float32).view(1, 1, 3)

    return GI_tem, center_g_t, indices, center_g_np


def segment_volume(
    input_path: str,
    output_path: str,
    model_weights: str,
    template_img_path: str,
    mask_template_img_path: str,
    shapes: Tuple[int, int, int] = (192, 192, 192),
    norm_method: str = "zs",
    n_patch: int = 64,
    t_patch: int = 64,
    device: Optional[str] = None,
) -> None:
    """Run the full segmentation pipeline and save the result.

    The function performs the following steps:
        1. Estimate the brain center via registration to a template.
        2. Load the volume, normalize it, and resize to ``shapes``.
        3. Build the geometric template for the network.
        4. Load the pre-trained model and run inference.
        5. Undo the resize to restore the original volume dimensions.
        6. Save the segmentation mask to ``output_path`` using the original
           affine matrix and header from the input image.

    Args:
        input_path: Path to the input NIfTI file.
        output_path: Destination path for the segmentation mask.
        model_weights: Path to the pre-trained model weights (.pth file).
        template_img_path: Path to the template image used for registration.
        mask_template_img_path: Path to the brain mask in template space.
        shapes: Target shape for resizing (z, y, x). Default is (192, 192, 192).
        norm_method: Normalization method ('zs' or 'mm'). Default is 'zs'.
        n_patch: Number of patches along polar angle dimension for GI conversion.
        t_patch: Number of sampled patches along theta dimension.
        device: Torch device ("cpu" or "cuda"). If None, auto-detect.

    Returns:
        None. The function writes the result to ``output_path``.
    """
    # Choose device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Compute brain center using registration
    center = find_center(input_path, template_img_path, mask_template_img_path)

    # Preprocess input volume (normalize and resize)
    volume_np, transform = process_scan_with_transform(
        input_path,
        mask=False,
        resize=True,
        norm_method=norm_method,
        output_shape=shapes,
        center=center,
        re_ori=True,
    )

    volume_torch = torch.tensor(volume_np, dtype=torch.float32)
    volume_torch = volume_torch.to(device)
    volume_torch = volume_torch.unsqueeze(0).unsqueeze(0)  # shape: (1,1,D,H,W)

    # Prepare geometric template
    GI_tem, center_g_t, _, _ = prepare_gi_template(n_patch, t_patch)
    GI_tem = GI_tem.to(device)
    center_g_t = center_g_t.to(device)

    # Load segmentation model
    model = seq_Gnet_v5(method="gate_v2", set_sdf=False, n_pc=int(t_patch**2))
    model_load(model, model_weights)
    model = model.to(device)
    model.eval()

    # Scale coordinates for the model: divide first 3 components by 191
    coors_tem_in = GI_tem.clone()
    coors_tem_in[:, :3, :] = coors_tem_in[:, :3, :] / 191.0
    coors_tem_in = coors_tem_in.repeat(volume_torch.shape[0], 1, 1)

    # Inference
    with torch.no_grad():
        print("volume_torch,  coors_tem_in, center_g_t shape:", volume_torch.shape,  coors_tem_in.shape, center_g_t.shape)
        _, y_pred, _ = model(volume_torch, coors_tem_in, center_g_t)

    # Convert model output to numpy
    mask_resized = y_pred[0, 0].cpu().numpy()

    # Undo resize to original shape
    mask_orig = undo_resize(mask_resized, transform)

    # Save the result as a NIfTI file
    orig_img = nib.load(input_path)
    mask_img = nib.Nifti1Image(mask_orig.astype(np.float32), orig_img.affine, orig_img.header)
    nib.save(mask_img, output_path)


__all__ = [
    "segment_volume",
    "read_nifti_file",
    "normalize",
    "resize_volume_new",
    "process_scan_with_transform",
    "undo_resize",
    "find_center",
]