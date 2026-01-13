# %%
# ============================================================================
# FIXED DATA_LOADER.PY
# ============================================================================

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pathlib import Path

class MultiChannelHARDataset(Dataset):
    """
    HAR dataset with multi-channel images for Pix2Seq encoder.
    
    Handles [N, 9, 3, 224, 224] images by processing each channel separately.
    """
    
    def __init__(self, data_path, mode='flatten', transform=None):
        """
        Args:
            data_path: Path to .pt file
            mode: How to handle 9 channels
                - 'flatten': Process each of 9 channels separately [N*9 samples]
                - 'average': Average across channels [N samples, 3, 224, 224]
                - 'first': Use only first channel [N samples, 3, 224, 224]
            transform: Image transforms (augmentation)
        """
        data = torch.load(data_path)
        self.images = data['images']            # [N, 9, 3, 224, 224]
        self.labels = data['labels'].long()     # [N] - CONVERT TO LONG!
        self.mode = mode
        self.transform = transform
        
        # Store original sizes
        self.num_samples = len(self.labels)
        self.num_channels = 9
        
        print(f"Loaded {data_path}")
        print(f"  Original: {self.images.shape}")
        print(f"  Labels: {self.labels.shape}, dtype: {self.labels.dtype}")
        print(f"  Mode: {mode}")
        
        if mode == 'flatten':
            # Each channel becomes a separate sample
            # [N, 9, 3, 224, 224] -> [N*9, 3, 224, 224]
            self.images_flat = self.images.view(-1, 3, 224, 224)  # [N*9, 3, 224, 224]
            self.labels_flat = self.labels.repeat_interleave(9)    # [N*9]
            self.effective_size = self.num_samples * self.num_channels
            print(f"  Flattened: {self.images_flat.shape}")
            print(f"  Labels: {self.labels_flat.shape}")
            
        elif mode == 'average':
            # Average across 9 channels
            # [N, 9, 3, 224, 224] -> [N, 3, 224, 224]
            self.images_avg = self.images.mean(dim=1)
            self.effective_size = self.num_samples
            print(f"  Averaged: {self.images_avg.shape}")
            
        elif mode == 'first':
            # Use only first channel
            self.images_first = self.images[:, 0]  # [N, 3, 224, 224]
            self.effective_size = self.num_samples
            print(f"  First channel: {self.images_first.shape}")
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def __len__(self):
        return self.effective_size
    
    def __getitem__(self, idx):
        if self.mode == 'flatten':
            image = self.images_flat[idx]
            label = self.labels_flat[idx]
        elif self.mode == 'average':
            image = self.images_avg[idx]
            label = self.labels[idx]
        elif self.mode == 'first':
            image = self.images_first[idx]
            label = self.labels[idx]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_original_sample(self, idx):
        """Get original multi-channel sample (all 9 channels)."""
        return self.images[idx], self.labels[idx]


def create_har_dataloaders(
    data_dir='data/HAR/multichannel_images',
    mode='flatten',
    batch_size=32,
    num_workers=4,
    use_augmentation=True
):
    """
    Create dataloaders for HAR multi-channel images.
    
    Args:
        data_dir: Directory with multichannel_images .pt files
        mode: 'flatten', 'average', or 'first'
        batch_size: Batch size
        num_workers: DataLoader workers
        use_augmentation: Apply data augmentation to training
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Data augmentation for training
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    # No augmentation for val/test
    test_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = MultiChannelHARDataset(
        f'{data_dir}/train_multichannel_images.pt',
        mode=mode,
        transform=train_transform
    )
    
    val_dataset = MultiChannelHARDataset(
        f'{data_dir}/val_multichannel_images.pt',
        mode=mode,
        transform=test_transform
    )
    
    test_dataset = MultiChannelHARDataset(
        f'{data_dir}/test_multichannel_images.pt',
        mode=mode,
        transform=test_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print("\n" + "="*80)
    print("DATALOADERS CREATED")
    print("="*80)
    print(f"Mode: {mode}")
    print(f"Batch size: {batch_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader
