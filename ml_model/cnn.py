# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

"""
This model is a U-Net style architecture for thermal field prediction.
The input is a 3-channel image (sdf, time field, gradient magnitude)
The output is a 1-channel image (temperature field)

U-Net Features:
- Encoder-decoder structure with skip connections
- Multi-scale feature extraction
- Preserves spatial details while capturing global context
- Optimized for image-to-image regression tasks
"""

class DoubleConv(nn.Module):
    """(convolution => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        padding = (kernel_size - 1) // 2  # Maintain spatial dimensions
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, kernel_size=kernel_size)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True, kernel_size=5):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # For bilinear, we need to handle the concatenation properly
            # in_channels will be the sum of upsampled channels + skip connection channels
            self.conv = DoubleConv(in_channels, out_channels, kernel_size=kernel_size)
        else:
            # For transpose convolution, use larger kernel with appropriate padding
            transpose_kernel = 6  # Larger kernel for better upsampling
            transpose_padding = 2  # Adjusted padding for proper 2x upsampling
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 
                                       kernel_size=transpose_kernel, stride=2, padding=transpose_padding)
            self.conv = DoubleConv(in_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class myModel(nn.Module):
    """
    U-Net for thermal field prediction
    - in_channels: number of input channels (2 for SDF + time)
    - out_channels: number of output channels (1 for temperature)
    - bilinear: use bilinear upsampling instead of transpose convolution
    - kernel_size: size of convolutional kernels (default 5 for larger spatial features)
    """
    def __init__(self, in_channels=3, out_channels=1, bilinear=True, kernel_size=7):
        super(myModel, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.kernel_size = kernel_size

        # Parametric encoder/decoder depth
        self.depth = 3  # Default value, can be set as an argument if desired

        # Example: to make it configurable, add a depth argument to __init__ and set self.depth = depth

        # Channel progression (can be customized as needed)
        base_channels = 16
        channels = [base_channels * (2 ** i) for i in range(self.depth + 1)]  # e.g. [64, 128, 256, 512, 1024] for depth=4

        # Encoder (downsampling path)
        self.inc = DoubleConv(in_channels, channels[0], kernel_size=kernel_size)
        self.downs = nn.ModuleList()
        for i in range(self.depth):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            # For the last down, apply factor for bilinear upsampling
            if i == self.depth - 1:
                factor = 2 if bilinear else 1
                out_ch = out_ch // factor
            self.downs.append(Down(in_ch, out_ch, kernel_size=kernel_size))

        # Decoder (upsampling path)
        self.ups = nn.ModuleList()
        for i in range(self.depth, 0, -1):
            # For bilinear upsampling, in_channels will be the sum of:
            # - upsampled feature map channels (same as input)
            # - skip connection channels
            if bilinear:
                # The upsampled feature map has the same channels as the input
                upsampled_channels = channels[i] if i != self.depth else channels[i] // 2
                # Skip connection has channels from the corresponding encoder level
                skip_channels = channels[i - 1]
                # Total input channels after concatenation
                in_ch = upsampled_channels + skip_channels
            else:
                # For transpose convolution, the upsampling reduces channels
                in_ch = channels[i] if i != self.depth else channels[i] // 2
            out_ch = channels[i - 1]
            self.ups.append(Up(in_ch, out_ch, bilinear, kernel_size=kernel_size))
        self.outc = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        # Encoder
        downs_outputs = [x1]
        for down in self.downs:
            downs_outputs.append(down(downs_outputs[-1]))
        # Decoder with skip connections
        x = downs_outputs[-1]
        for i, up in enumerate(self.ups):
            skip = downs_outputs[-(i + 2)]
            x = up(x, skip)
        logits = self.outc(x)
        return logits

class myLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, predictions, targets, masks):
        predictions = predictions.squeeze(1)
        mse = (predictions - targets) ** 2
        masked_mse = mse * masks
        return masked_mse.sum() / masks.sum() 

# %%
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
from scipy.ndimage import rotate, laplace
from scipy.interpolate import griddata

import json
from shapely import wkt

import sys
import os



class myDataset(Dataset):
    def __init__(self, data_dirs, N_max=None, normalize=True, 
    sdf_max=0.001, 
    time_max=0.08, 
    temp_min=1000.0, 
    temp_max=2000.0,
    grad_max=0.002,
    grad_min=0.0,
    align_toolpath=False,
    ):
        """
        Args:
            data_dirs: List of data directories to load from
            N_max: Maximum grid size for padding
            normalize: Whether to apply normalization to the fields
            sdf_max: Maximum SDF value for normalization (m)
            time_max: Maximum time value for normalization (s)
            temp_min: Minimum temperature value for normalization (K)
            temp_max: Maximum temperature value for normalization (K)
            grad_max: Maximum gradient value for normalization
            grad_min: Minimum gradient value for normalization
            align_toolpath: Whether to rotate fields to align raster angle to 0 degrees
        """
        self.data_dirs = data_dirs if isinstance(data_dirs, list) else [data_dirs]
        self.N_max = N_max
        self.normalize = normalize
        self.align_toolpath = align_toolpath
        # Normalization parameters as lists (input channels first, then output)
        # Order: [SDF, time_field, grad_mag, temp_field]
        self.min_values = [0.0, 0.0, grad_min, temp_min]  # [sdf_min, time_min, grad_min, temp_min]
        self.max_values = [sdf_max, time_max, grad_max, temp_max]  # [sdf_max, time_max, grad_max, temp_max]
        # Cache for raster angles to avoid repeated file reads
        self.raster_angle_cache = {}

        # Build file index (one entry per directory)
        self.file_index = []
        for data_dir in self.data_dirs:
            self.file_index.append({
                'data_dir': data_dir,
            })

    def __len__(self):
        return len(self.file_index)

    def _normalize_fields(self, fields):
        """Normalize a list of fields using the stored min/max values"""
        if not self.normalize:
            return fields
        normalized_fields = []
        for i, field in enumerate(fields):
            min_val = self.min_values[i]
            max_val = self.max_values[i]
            normalized_fields.append((field - min_val) / (max_val - min_val))
        return normalized_fields
    
    def denormalize_fields(self, fields, minmax: List[float] = None):
        """Denormalize a list of fields using the stored min/max values"""
        denormalized_fields = []
        for i, field in enumerate(fields):
            if minmax is None:
                min_val = self.min_values[i]
                max_val = self.max_values[i]
            else:
                min_val = minmax[0]
                max_val = minmax[1]
            denormalized_fields.append(field * (max_val - min_val) + min_val)
        return denormalized_fields
    
    def get_normalization_info(self):
        return {
            'normalize': self.normalize,
            'min_values': self.min_values,
            'max_values': self.max_values,
            'field_names': ['sdf', 'time', 'grad', 'temp']
        }

    def _get_raster_angle(self, data_dir):
        if data_dir in self.raster_angle_cache:
            return self.raster_angle_cache[data_dir]
        metadata_file = os.path.join(data_dir, 'metadata_01.json')
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Raster angle metadata file not found: {metadata_file}")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        raster_angle = metadata['config_data']['laser_parameters']['raster_angle_degrees']
        self.raster_angle_cache[data_dir] = raster_angle
        return raster_angle

    def _rotate_fields(self, fields: np.ndarray, angle: float):
        rotated_fields = []
        for field in fields:
            rotated_field = rotate(field, angle, reshape=False, order=1)
            rotated_fields.append(rotated_field)
        return rotated_fields

    def _pad_fields(self, fields: np.ndarray, grid_size: int):
        pad_size = (self.N_max - grid_size) // 2
        padded_fields = []
        for field in fields:
            padded_field = np.zeros((self.N_max, self.N_max), dtype=field.dtype)
            padded_field[pad_size:pad_size+grid_size, pad_size:pad_size+grid_size] = field
            padded_fields.append(padded_field)
        return padded_fields
    
    def _mask_fields(self, fields: np.ndarray , mask: np.ndarray, else_values: List[float]):
        masked_fields = []
        for field, else_value in zip(fields, else_values):
            # print(field.shape)
            # print(mask.shape) 
            # print(else_value.shape)
            masked_fields.append(np.where(mask, field, else_value))
        return masked_fields
    
    def __getitem__(self, idx):
        file_info = self.file_index[idx]
        data_dir = file_info['data_dir']


        inside_outside_list = np.load(os.path.join(data_dir, 'inside_outside_array.npy'))
        is_inside_values = inside_outside_list[:, 2]
        item_grid_size = int(np.sqrt(len(is_inside_values))) # grid size of the loaded item which is related to the size of the shape
        inside_outside_field = is_inside_values.reshape((item_grid_size, item_grid_size))
        
        if self.N_max is not None:
            if item_grid_size > self.N_max:
                raise ValueError(f"N_max must be greater than or equal to grid_size. N_max: {self.N_max}, grid_size: {item_grid_size}")
            inside_mask = self._pad_fields([inside_outside_field], item_grid_size)[0]
        else:
            inside_mask = inside_outside_field
        
        # Load precomputed fields (raw, unnormalized)
        sdf_field = np.load(os.path.join(data_dir, 'sdf_field.npy'))
        time_field = np.load(os.path.join(data_dir, 'time_field.npy'))
        grad_mag = np.load(os.path.join(data_dir, 'grad_mag.npy'))
        # Handle object arrays that require allow_pickle=True
        try:
            temp_field = np.load(os.path.join(data_dir, 'max_T_tp.npy'))
        except ValueError:
            temp_field = np.load(os.path.join(data_dir, 'max_T_tp.npy'), allow_pickle=True)
        fields = [sdf_field, time_field, grad_mag, temp_field]

        if self.N_max is not None:
            fields = self._pad_fields(fields, item_grid_size)
        # --- Alignment step: rotate all fields so raster angle is 0 (horizontal) ---
        if self.align_toolpath:
            fields = self._rotate_fields(fields, -self._get_raster_angle(data_dir)) # rotate by -raster_angle to align to 0
        # ---
        fields = self._mask_fields(fields, inside_mask, self.min_values) # mask out outside of the shape
        # --- Normalize fields ---
        if self.normalize:
            fields = self._normalize_fields(fields)

        return {
            'input': torch.tensor(np.array(fields[0:3]), dtype=torch.float32),
            'target': torch.tensor(fields[3], dtype=torch.float32),
            'mask': torch.tensor(inside_mask, dtype=torch.bool)
        }


class myUnlabeledDataset(myDataset):
    def __getitem__(self, idx):
        file_info = self.file_index[idx]
        data_dir = file_info['data_dir']

        # Load and process inside/outside field
        inside_outside_list = np.load(os.path.join(data_dir, 'inside_outside_array.npy'))
        is_inside_values = inside_outside_list[:, 2]
        item_grid_size = int(np.sqrt(len(is_inside_values)))
        inside_outside_field = is_inside_values.reshape((item_grid_size, item_grid_size))

        if self.N_max is not None:
            if item_grid_size > self.N_max:
                raise ValueError(f"N_max must be greater than or equal to grid_size. N_max: {self.N_max}, grid_size: {item_grid_size}")

        # Load precomputed fields (raw, unnormalized)
        sdf_field = np.load(os.path.join(data_dir, 'sdf_field.npy'))
        time_field = np.load(os.path.join(data_dir, 'time_field.npy'))
        grad_mag = np.load(os.path.join(data_dir, 'grad_mag.npy'))
        fields = [sdf_field, time_field, grad_mag]

        if self.N_max is not None:
            fields = self._pad_fields(fields, item_grid_size)
        
        # --- Alignment step: rotate all fields so raster angle is 0 (horizontal) ---
        if self.align_toolpath:
            fields = self._rotate_fields(fields, -self._get_raster_angle(data_dir)) # rotate by -raster_angle to align to 0
        # ---
        if self.N_max is not None:
            inside_mask = self._pad_fields([inside_outside_field], item_grid_size)[0]
        else:
            inside_mask = inside_outside_field
        fields = self._mask_fields(fields, inside_mask, [0.0, 0.0, 0.0])
        
        # --- Normalize fields ---
        if self.normalize:
            fields = self._normalize_fields(fields)

        return {
            'input': torch.tensor(np.array(fields), dtype=torch.float32),
            'mask': torch.tensor(inside_mask, dtype=torch.bool)
        }

# %%

if __name__ == "__main__":
    import matplotlib as mpl
    mpl.rcParams['font.size'] = 16
    mpl.rcParams['axes.labelsize'] = 16
    mpl.rcParams['legend.fontsize'] = 16
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 12
    
    import matplotlib.pyplot as plt

    # List of data directories to visualize
    data_dirs = [
        "/mnt/c/Users/jamba/sim_data/DATABASE_PA/DATA_017",
        # Add more directories as needed
        "/mnt/c/Users/jamba/sim_data/DATABASE_PA/DATA_040",
        "/mnt/c/Users/jamba/sim_data/DATABASE_PA/DATA_086",
    ]

    dataset = myDataset(
        data_dirs=data_dirs,
        # N_max=180,
        normalize=True
    )
    normalization_info = dataset.get_normalization_info()
    print(normalization_info)
    n_rows = len(data_dirs)
    fig, axs = plt.subplots(n_rows, 4, figsize=(18, 5 * n_rows))
    if n_rows == 1:
        axs = axs[None, :]  # Ensure axs is always 2D for consistent indexing

    for row_idx, sample in enumerate([dataset[i] for i in range(n_rows)]):
        # Denormalize all fields

        sdf = dataset.denormalize_fields([sample['input'][0].cpu().numpy()], minmax=[normalization_info['min_values'][0], normalization_info['max_values'][0]])[0]
        time = dataset.denormalize_fields([sample['input'][1].cpu().numpy()], minmax=[normalization_info['min_values'][1], normalization_info['max_values'][1]])[0]
        grad = dataset.denormalize_fields([sample['input'][2].cpu().numpy()], minmax=[normalization_info['min_values'][2], normalization_info['max_values'][2]])[0]
        temp = dataset.denormalize_fields([sample['target'].cpu().numpy()], minmax=[normalization_info['min_values'][3], normalization_info['max_values'][3]])[0]
        # Apply mask before plotting
        mask = sample['mask'].cpu().numpy()
        masked_sdf = np.where(mask, sdf, np.nan)
        masked_time = np.where(mask, time, np.nan)
        masked_grad = np.where(mask, grad, np.nan)
        masked_temp = np.where(mask, temp, np.nan)

        # SDF
        im0 = axs[row_idx, 0].imshow(masked_sdf, cmap='copper')
        axs[row_idx, 0].axis('off')
        cbar0 = plt.colorbar(im0, ax=axs[row_idx, 0], orientation='horizontal', pad=0.15)
        cbar0.set_label('SDF (m)', fontsize=13)
        # Time Field
        im1 = axs[row_idx, 1].imshow(masked_time)
        axs[row_idx, 1].axis('off')
        cbar1 = plt.colorbar(im1, ax=axs[row_idx, 1], orientation='horizontal', pad=0.15)
        cbar1.set_label('t (s)', fontsize=13)
        # Gradient Magnitude
        im2 = axs[row_idx, 2].imshow(masked_grad, cmap='magma')
        axs[row_idx, 2].axis('off')
        cbar2 = plt.colorbar(im2, ax=axs[row_idx, 2], orientation='horizontal', pad=0.15)
        cbar2.set_label(r'$\left|\nabla\, t\right|$ (K/m)', fontsize=13)
        # Max Temperature
        im3 = axs[row_idx, 3].imshow(masked_temp, cmap='hot', vmin=1400, vmax=1700)
        axs[row_idx, 3].axis('off')
        cbar3 = plt.colorbar(im3, ax=axs[row_idx, 3], orientation='horizontal', pad=0.15)
        cbar3.set_label('max(T) (K)', fontsize=13)

    plt.tight_layout()
    # plt.show()
    fig.savefig('cnn_16_example.svg', dpi=600, format='svg')
# %%
    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # sample = dataset[0]
    # sdf = sample['input'][0].cpu().numpy()
    # time = sample['input'][1].cpu().numpy()
    # grad = sample['input'][2].cpu().numpy()
    # im0 = axs[0].imshow(sdf)
    # axs[0].set_title('SDF')
    # axs[0].set_xticks([])
    # axs[0].set_yticks([])
    # im1 = axs[1].imshow(time)
    # axs[1].set_title('Time')
    # axs[1].set_xticks([])
    # axs[1].set_yticks([])
    # im2 = axs[2].imshow(grad)
    # axs[2].set_title(r'$\left|\nabla\, t\right|$')
    # axs[2].set_xticks([])
    # axs[2].set_yticks([])
    # plt.tight_layout()
    # plt.show()