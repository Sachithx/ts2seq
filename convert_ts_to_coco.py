"""
Convert Hierarchical Time Series Annotations to COCO Format for Pix2Seq
"""

import torch
import numpy as np
import json
import tensorflow as tf
from PIL import Image
import io
import matplotlib.pyplot as plt

# Your existing imports and classes
from hierarchical_event_labeling import (
    CompleteHierarchicalEventDataset, 
    HierarchicalAnnotation,
    VOCAB
)


class TimeSeriesAnnotationToCOCO:
    """
    Convert time series hierarchical annotations to COCO format.
    
    Key mappings:
        - Event interval [start, end] → bbox [ymin=0, xmin=start/L, ymax=1, xmax=end/L]
        - Event label → class_id
        - Time series → 2D image visualization
    """
    
    def __init__(self, 
                 image_height: int = 640,
                 image_width: int = 640,
                 viz_type: str = 'line_plot'):
        """
        Args:
            image_height: Target image height
            image_width: Target image width  
            viz_type: 'line_plot', 'heatmap', or 'spectrogram'
        """
        self.image_height = image_height
        self.image_width = image_width
        self.viz_type = viz_type
    
    def create_annotations_json(self, 
                                dataset: CompleteHierarchicalEventDataset,
                                output_path: str = '/tmp/ts_annotations/instances_train.json'):
        """Create COCO-style annotations JSON"""
        
        # Get all unique labels from dataset
        unique_labels = set()
        for annotation in dataset.annotations:
            for event in annotation.all_events:
                unique_labels.add(event.label)
        
        # Create categories (COCO format requires 1-based IDs)
        categories = []
        for label_id in sorted(unique_labels):
            label_name = VOCAB.id_to_label(label_id)
            categories.append({
                "id": int(label_id),  # Use original label ID
                "name": label_name,
                "supercategory": "event"
            })
        
        annotations_dict = {
            "info": {
                "description": "Time Series Hierarchical Event Detection Dataset",
                "version": "1.0",
                "year": 2026
            },
            "licenses": [],
            "categories": categories,
            "images": [],  # Will be filled during TFRecord creation
            "annotations": []  # Will be filled during TFRecord creation
        }
        
        # Save JSON
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(annotations_dict, f, indent=2)
        
        print(f"✓ Created annotations JSON with {len(categories)} categories")
        return annotations_dict
    
    def timeseries_to_image(self, 
                           x: torch.Tensor,
                           annotation: HierarchicalAnnotation = None) -> np.ndarray:
        """
        Convert time series to image.
        
        Args:
            x: Time series [L]
            annotation: Optional annotation for visualization
            
        Returns:
            RGB image array [H, W, 3]
        """
        x_np = x.cpu().numpy()
        L = len(x_np)
        
        if self.viz_type == 'line_plot':
            return self._create_line_plot(x_np, annotation)
        elif self.viz_type == 'heatmap':
            return self._create_heatmap(x_np)
        elif self.viz_type == 'spectrogram':
            return self._create_spectrogram(x_np)
        else:
            raise ValueError(f"Unknown viz_type: {self.viz_type}")
    
    def _create_line_plot(self, x: np.ndarray, annotation=None) -> np.ndarray:
        """Create line plot visualization"""
        fig, ax = plt.subplots(figsize=(self.image_width/100, self.image_height/100), dpi=80)
        
        # Plot time series
        ax.plot(x, linewidth=1, color='blue', alpha=0.8)
        
        # Optionally overlay events
        if annotation is not None:
            colors = plt.cm.Set3(np.linspace(0, 1, 12))
            for i, event in enumerate(annotation.all_events[:20]):  # Limit to first 20
                color = colors[i % len(colors)]
                ax.axvspan(event.start, event.end, alpha=0.2, color=color)
        
        ax.set_xlim(0, len(x)-1)
        ax.set_ylim(x.min() - 0.1*np.abs(x.min()), x.max() + 0.1*np.abs(x.max()))
        ax.axis('off')
        plt.tight_layout(pad=0)
        
        # ✅ FIX: Use buffer_rgba() instead of tostring_rgb()
        fig.canvas.draw()
        
        # Get RGBA buffer and convert to RGB
        buf = fig.canvas.buffer_rgba()
        img = np.frombuffer(buf, dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = img[:, :, :3]  # Drop alpha channel, keep RGB only
        
        plt.close(fig)
        
        # Resize to target size
        img_pil = Image.fromarray(img).resize((self.image_width, self.image_height))
        return np.array(img_pil)
    
    def _create_heatmap(self, x: np.ndarray) -> np.ndarray:
        """Create heatmap by reshaping time series"""
        L = len(x)
        
        # Reshape to 2D
        rows = int(np.sqrt(L))
        cols = L // rows
        x_2d = x[:rows*cols].reshape(rows, cols)
        
        # Normalize
        x_norm = (x_2d - x_2d.min()) / (x_2d.max() - x_2d.min() + 1e-8)
        
        # Convert to RGB using colormap
        cmap = plt.cm.viridis
        img = (cmap(x_norm)[:, :, :3] * 255).astype(np.uint8)
        
        # Resize
        img_pil = Image.fromarray(img).resize((self.image_width, self.image_height))
        return np.array(img_pil)
    
    def _create_spectrogram(self, x: np.ndarray) -> np.ndarray:
        """Create spectrogram visualization"""
        from scipy import signal
        
        f, t, Sxx = signal.spectrogram(x, fs=1.0, nperseg=min(64, len(x)//4))
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        
        # Normalize
        Sxx_norm = (Sxx_db - Sxx_db.min()) / (Sxx_db.max() - Sxx_db.min() + 1e-8)
        
        # Convert to RGB
        cmap = plt.cm.inferno
        img = (cmap(Sxx_norm)[:, :, :3] * 255).astype(np.uint8)
        
        # Resize
        img_pil = Image.fromarray(img).resize((self.image_width, self.image_height))
        return np.array(img_pil)
    
    def create_tfrecords(self,
                        dataset: CompleteHierarchicalEventDataset,
                        output_path: str = '/tmp/ts_tfrecords/train-00000-of-00001.tfrecord'):
        """
        Convert dataset to TFRecord format matching COCO structure.
        
        Each event becomes a bounding box where:
            - xmin = start_time / sequence_length
            - xmax = end_time / sequence_length  
            - ymin = 0.0 (full height)
            - ymax = 1.0 (full height)
            - class = event.label
        """
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        num_examples = len(dataset)
        print(f"\nCreating TFRecords: {output_path}")
        print(f"Processing {num_examples} sequences...")
        
        with tf.io.TFRecordWriter(output_path) as writer:
            for idx in range(num_examples):
                if idx % 50 == 0:
                    print(f"  Progress: {idx}/{num_examples}")
                
                annotation = dataset[idx]
                x = annotation.sequence
                L = len(x)
                
                # Convert time series to image
                img_array = self.timeseries_to_image(x, annotation)
                
                # Encode image
                img_pil = Image.fromarray(img_array)
                buffer = io.BytesIO()
                img_pil.save(buffer, format='PNG')
                image_encoded = buffer.getvalue()
                
                # Extract events as bounding boxes
                events = annotation.all_events
                # keep only events that are not global “regime”
                events = [e for e in events if e.event_type != "regime"]

                if len(events) == 0:
                    # Skip sequences with no events
                    continue
                
                bboxes_ymin = []
                bboxes_xmin = []
                bboxes_ymax = []
                bboxes_xmax = []
                labels = []
                areas = []
                is_crowd = []
                
                for event in events:
                    # Map time interval to normalized x coordinates
                    xmin = event.start / L
                    xmax = (event.end + 1) / L  # ✅ inclusive end fix
                    xmax = min(1.0, xmax)
                    
                    # Use full height for y coordinates
                    ymin = 0.0
                    ymax = 1.0
                    
                    bboxes_ymin.append(ymin)
                    bboxes_xmin.append(xmin)
                    bboxes_ymax.append(ymax)
                    bboxes_xmax.append(xmax)
                    labels.append(event.label)
                    
                    # Calculate area in pixels
                    area = (ymax - ymin) * (xmax - xmin) * self.image_height * self.image_width
                    areas.append(area)
                    is_crowd.append(0)
                
                # Create TFRecord feature
                feature = {
                    'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_encoded])),
                    'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[self.image_height])),
                    'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[self.image_width])),
                    'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(idx).encode('utf-8')])),
                    'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[f'ts_{idx}.png'.encode('utf-8')])),
                    'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=bboxes_ymin)),
                    'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=bboxes_xmin)),
                    'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=bboxes_ymax)),
                    'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=bboxes_xmax)),
                    'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=labels)),
                    'image/object/area': tf.train.Feature(float_list=tf.train.FloatList(value=areas)),
                    'image/object/is_crowd': tf.train.Feature(int64_list=tf.train.Int64List(value=is_crowd)),
                }
                
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
        
        print(f"✓ Created TFRecord with {num_examples} sequences")


# ============================================================================
# MAIN CONVERSION PIPELINE
# ============================================================================

def convert_timeseries_dataset_to_coco(
    # x: torch.Tensor,
    train_dataset: CompleteHierarchicalEventDataset = None,
    val_dataset: CompleteHierarchicalEventDataset = None,
    train_split: float = 0.8,
    viz_type: str = 'line_plot',
    output_dir: str = '/tmp/ts_coco'
):
    """
    Complete pipeline to convert time series dataset to COCO format.
    
    Args:
        x: Time series tensor [B, L]
        train_split: Train/val split ratio
        viz_type: Visualization type
        output_dir: Output directory
    """
    print("\n" + "="*80)
    print("TIME SERIES → COCO FORMAT CONVERSION PIPELINE")
    print("="*80)
    
    # # Step 1: Create hierarchical annotations
    # print("\n[1/5] Creating hierarchical annotations...")
    # if dataset is None:
    #     dataset = CompleteHierarchicalEventDataset(
    #         x,
    #         use_spectral=True,
    #         use_entropy=True,
    #         use_wavelets=True,
    #         use_wavelet_peaks=True,
    #         use_changepoint=True,
    #         use_chaotic=False,  # ✅ DISABLED - causes issues with empty entropy
    #         verbose=False
    #     )
    # print(f"✓ Created {len(dataset)} annotated sequences")
    
    # Step 2: Split dataset
    # print("\n[2/5] Splitting dataset...")
    # B = len(dataset)
    # train_size = int(B * train_split)
    # indices = torch.randperm(B)
    # train_indices = indices[:train_size]
    # val_indices = indices[train_size:]
    
    # # Create subset datasets
    # train_x = x[train_indices]
    # val_x = x[val_indices]
    
    # train_dataset = CompleteHierarchicalEventDataset(
    #     train_x,
    #     use_spectral=True,
    #     use_entropy=True,
    #     use_wavelets=True,
    #     use_wavelet_peaks=True,
    #     use_changepoint=True,
    #     use_chaotic=False,  # ✅ DISABLED
    #     verbose=False
    # )
    # val_dataset = CompleteHierarchicalEventDataset(
    #     val_x,
    #     use_spectral=True,
    #     use_entropy=True,
    #     use_wavelets=True,
    #     use_wavelet_peaks=True,
    #     use_changepoint=True,
    #     use_chaotic=False,  # ✅ DISABLED
    #     verbose=False
    # )
    
    print(f"✓ Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Step 3: Create converter
    print("\n[3/5] Initializing converter...")
    converter = TimeSeriesAnnotationToCOCO(
        image_height=128,
        image_width=128,
        viz_type=viz_type
    )
    
    # Step 4: Create annotations JSON
    print("\n[4/5] Creating COCO annotations...")
    converter.create_annotations_json(
        train_dataset,
        output_path=f'{output_dir}/annotations/instances_train.json'
    )
    converter.create_annotations_json(
        val_dataset,
        output_path=f'{output_dir}/annotations/instances_val.json'
    )
    
    # Step 5: Create TFRecords
    print("\n[5/5] Creating TFRecords...")
    converter.create_tfrecords(
        train_dataset,
        output_path=f'{output_dir}/tfrecords/train-00000-of-00001.tfrecord'
    )
    converter.create_tfrecords(
        val_dataset,
        output_path=f'{output_dir}/tfrecords/val-00000-of-00001.tfrecord'
    )
    
    print("\n" + "="*80)
    print("✓ CONVERSION COMPLETE!")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  - Annotations: {output_dir}/annotations/")
    print(f"  - TFRecords: {output_dir}/tfrecords/")
    print(f"\nTrain command:")
    print(f"""
python3 run.py --mode=train --model_dir=/tmp/ts_model \\
  --config=configs/config_det_finetune.py \\
  --config.dataset.train_file_pattern='{output_dir}/tfrecords/train*.tfrecord' \\
  --config.dataset.val_file_pattern='{output_dir}/tfrecords/val*.tfrecord' \\
  --config.dataset.category_names_path='{output_dir}/annotations/instances_val.json' \\
  --config.model.pretrained_ckpt='' \\
  --config.train.batch_size=16 \\
  --config.train.epochs=10 \\
  --config.optimization.learning_rate=3e-5 \\
  --config.model.num_encoder_layers=4 \\
  --config.model.num_decoder_layers=2 \\
  --config.model.dim_att=256 \\
  --config.model.dim_att_dec=128
    """)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Generate your time series data
    # B, L = 32, 128  # 200 sequences, 336 timesteps each
    # torch.manual_seed(42)
    # np.random.seed(42)
    #
    # # Create synthetic time series
    # t = torch.linspace(0, 4*np.pi, L)
    # x = torch.zeros(B, L)

    # for i in range(B):
    #     trend = 0.5 * torch.sin(t / 2) + 0.1 * t
    #     vol_modulator = 0.1 + 0.2 * (torch.sin(3 * t) > 0).float()
    #     noise = torch.randn(L) * vol_modulator
    #     num_spikes = np.random.randint(2, 5)
    #     spike_indices = torch.randint(50, L-50, (num_spikes,))
    #     spikes = torch.zeros(L)
    #     spikes[spike_indices] = torch.randn(num_spikes) * 2
    #     x[i] = trend + noise + spikes

    import torch
    import hierarchical_event_labeling as hel
    # Load pre-annotated HAR event datasets
    torch.serialization.add_safe_globals([hel.CompleteHierarchicalEventDataset])

    train_dataset = torch.load('data/HAR/train_event_dataset.pt', weights_only=False)
    val_dataset = torch.load('data/HAR/val_event_dataset.pt', weights_only=False)

    # Convert to COCO format
    convert_timeseries_dataset_to_coco(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_split=0.8,
        viz_type='line_plot',  # or 'heatmap' or 'spectrogram'
        output_dir='/tmp/ts_coco'
    )