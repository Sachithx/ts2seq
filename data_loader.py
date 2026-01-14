import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
from collections import Counter

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
            # Flatten images but DON'T repeat labels
            self.images_flat = self.images.view(-1, 3, 224, 224)  # [N*9, 3, 224, 224]
            # Labels stay as [N] - we'll handle indexing in __getitem__
            self.effective_size = self.num_samples  # ← Key change!
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def __len__(self):
        return self.effective_size
    
    def __getitem__(self, idx):
        if self.mode == 'flatten':
            start_idx = idx * self.num_channels
            end_idx = start_idx + self.num_channels
            images = self.images_flat[start_idx:end_idx]  # [9, 3, 224, 224]
            label = self.labels[idx]  # Single label
        
        # Apply transforms
        if self.transform:
            images = torch.stack([self.transform(img) for img in images])

        return images, label  # Return [9, 3, 224, 224] and single label

    def get_original_sample(self, idx):
        """Get original multi-channel sample (all 9 channels)."""
        return self.images[idx], self.labels[idx]


def custom_collate_fn(batch):
    """
    Collate function to flatten channel dimension.

    Args:
        batch: List of (images, label) tuples
               where images is [9, 3, 224, 224]

    Returns:
        images: [B*9, 3, 224, 224]
        labels: [B]
    """
    images_list = []
    labels_list = []

    for images, label in batch:
        # images: [9, 3, 224, 224]
        images_list.append(images)
        labels_list.append(label)

    # Stack batch: [B, 9, 3, 224, 224]
    images = torch.stack(images_list, dim=0)
    labels = torch.tensor(labels_list)

    # Flatten channels: [B, 9, 3, 224, 224] -> [B*9, 3, 224, 224]
    batch_size = images.size(0)
    images = images.view(-1, 3, 224, 224)

    return images, labels


def stratified_sample_indices(labels, percentage, seed=42):
    """
    Sample indices with stratified sampling to maintain class distribution.
    
    Args:
        labels: Tensor of labels [N]
        percentage: Percentage of data to sample (1-100)
        seed: Random seed
    
    Returns:
        selected_indices: Array of selected indices
        class_counts: Dictionary of class counts in sample
    """
    np.random.seed(seed)
    
    # Get unique classes
    unique_classes = torch.unique(labels).numpy()
    
    # Group indices by class
    class_to_indices = {}
    for cls in unique_classes:
        class_to_indices[cls] = np.where(labels.numpy() == cls)[0]
    
    # Sample from each class
    selected_indices = []
    class_counts_original = {}
    class_counts_sampled = {}
    
    for cls, indices in class_to_indices.items():
        n_samples = len(indices)
        n_select = max(1, int(n_samples * percentage / 100.0))  # At least 1 per class
        
        # Random sample without replacement
        sampled = np.random.choice(indices, n_select, replace=False)
        selected_indices.extend(sampled)
        
        class_counts_original[cls] = n_samples
        class_counts_sampled[cls] = n_select
    
    # Sort for cache performance
    selected_indices = np.array(sorted(selected_indices))
    
    return selected_indices, class_counts_original, class_counts_sampled


def save_subset_indices(indices, save_path):
    """Save selected indices to file."""
    torch.save({'indices': indices}, save_path)
    print(f"Saved subset indices to: {save_path}")


def load_subset_indices(load_path):
    """Load selected indices from file."""
    data = torch.load(load_path)
    indices = data['indices']
    print(f"Loaded subset indices from: {load_path}")
    return indices


def create_har_dataloaders(
    data_dir='/home/sachithxcviii/ts2seq/data/HAR/multichannel_images',
    mode='flatten',
    batch_size=32,
    num_workers=4,
    use_augmentation=True,
    train_percentage=100.0,
    seed=42,
    subset_indices_path=None,  # ← NEW: Path to save/load indices
    force_resample=False  # ← NEW: Force new sampling even if file exists
):
    """
    Create dataloaders for HAR multi-channel images with stratified sampling.
    
    Args:
        data_dir: Directory with multichannel_images .pt files
        mode: 'flatten', 'average', or 'first'
        batch_size: Batch size
        num_workers: DataLoader workers
        use_augmentation: Apply data augmentation to training
        train_percentage: Percentage of training data to use (1-100)
        seed: Random seed for sampling
        subset_indices_path: Path to save/load subset indices (e.g., 'subset_5pct.pt')
        force_resample: If True, resample even if indices file exists
    
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
    
    # Create full training dataset
    full_train_dataset = MultiChannelHARDataset(
        f'{data_dir}/train_multichannel_images.pt',
        mode=mode,
        transform=train_transform
    )
    
    # ========================================================================
    # STRATIFIED SUBSAMPLING WITH FIXED INDICES
    # ========================================================================
    if train_percentage < 100.0:
        # Check if we should load existing indices
        if subset_indices_path and Path(subset_indices_path).exists() and not force_resample:
            print(f"\n{'='*80}")
            print("LOADING EXISTING SUBSET INDICES")
            print(f"{'='*80}")
            indices = load_subset_indices(subset_indices_path)
            
            # Verify loaded indices
            total_samples = len(full_train_dataset)
            num_samples = len(indices)
            
            # Count classes in subset
            subset_labels = full_train_dataset.labels[indices]
            class_counts = Counter(subset_labels.numpy())
            
        else:
            # Perform stratified sampling
            print(f"\n{'='*80}")
            print("STRATIFIED SUBSAMPLING - CREATING NEW SUBSET")
            print(f"{'='*80}")
            
            indices, class_counts_original, class_counts_sampled = stratified_sample_indices(
                full_train_dataset.labels,
                train_percentage,
                seed
            )
            
            total_samples = len(full_train_dataset)
            num_samples = len(indices)
            
            print(f"Original size: {total_samples}")
            print(f"Percentage: {train_percentage}%")
            print(f"Subsampled size: {num_samples}")
            print(f"\nClass distribution:")
            print(f"{'Class':<10} {'Original':<12} {'Sampled':<12} {'Percentage':<12}")
            print(f"{'-'*50}")
            for cls in sorted(class_counts_original.keys()):
                orig = class_counts_original[cls]
                samp = class_counts_sampled[cls]
                pct = (samp / orig) * 100
                print(f"{cls:<10} {orig:<12} {samp:<12} {pct:<12.2f}%")
            
            # Save indices if path provided
            if subset_indices_path:
                save_subset_indices(indices, subset_indices_path)
            
            class_counts = class_counts_sampled
        
        print(f"{'='*80}\n")
        
        # Create subset
        train_dataset = Subset(full_train_dataset, indices)
        
    else:
        train_dataset = full_train_dataset
    
    # Val and test datasets (no subsampling)
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
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    print("\n" + "="*80)
    print("DATALOADERS CREATED")
    print("="*80)
    print(f"Mode: {mode}")
    print(f"Batch size: {batch_size}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"{'='*80}\n")
    
    return train_loader, val_loader, test_loader