"""
MEMORY-EFFICIENT Convert - CORRECT VERSION
Both COCO JSON and TFRecords use PIXEL coordinates
Pix2Seq normalizes them automatically during data loading
"""

import torch
import numpy as np
import json
import tensorflow as tf
from PIL import Image, ImageDraw
import io
import os
import gc

DIR = "/projects/pix2seqdata"

from hierarchical_event_labeling import (
    CompleteHierarchicalEventDataset, 
    HierarchicalAnnotation,
    VOCAB
)


class TimeSeriesAnnotationToCOCO:
    """Convert time series hierarchical annotations to COCO format."""
    
    def __init__(self, 
                 image_height: int = 224,
                 image_width: int = 224,
                 viz_type: str = 'line_plot'):
        self.image_height = image_height
        self.image_width = image_width
        self.viz_type = viz_type
    
    def create_annotations_json(self, 
                                dataset: CompleteHierarchicalEventDataset,
                                output_path: str):
        """Create COCO-style annotations JSON with images and annotations (PIXEL coordinates)"""
        unique_labels = set()
        for annotation in dataset.annotations:
            for event in annotation.all_events:
                unique_labels.add(event.label)
        
        categories = []
        for label_id in sorted(unique_labels):
            label_name = VOCAB.id_to_label(label_id)
            categories.append({
                "id": int(label_id),
                "name": label_name,
                "supercategory": "event"
            })
        
        # Create images and annotations lists
        images = []
        annotations = []
        annotation_id = 1
        
        print(f"  Building annotations for {len(dataset)} images...")
        for idx in range(len(dataset)):
            annotation_obj = dataset[idx]
            L = len(annotation_obj.sequence)
            
            # Add image entry
            images.append({
                "id": idx,
                "file_name": f"ts_{idx}.png",
                "height": self.image_height,
                "width": self.image_width
            })
            
            # Add annotation entries for this image
            events = [e for e in annotation_obj.all_events if e.event_type != "regime"]
            
            for event in events:
                W = float(self.image_width)
                H = float(self.image_height)
                
                # PIXEL coordinates (standard COCO format)
                xmin = (event.start / L) * W
                xmax = ((event.end + 1) / L) * W
                xmax = min(W, xmax)
                ymin, ymax = 0.0, H
                
                bbox_width = xmax - xmin
                bbox_height = ymax - ymin
                area = bbox_width * bbox_height
                
                annotations.append({
                    "id": annotation_id,
                    "image_id": idx,
                    "category_id": int(event.label),
                    "bbox": [xmin, ymin, bbox_width, bbox_height],  # COCO format: [x, y, w, h] in PIXELS
                    "area": area,
                    "iscrowd": 0
                })
                annotation_id += 1
            
            if (idx + 1) % 5000 == 0:
                print(f"    Processed {idx + 1}/{len(dataset)} images...")
        
        annotations_dict = {
            "info": {
                "description": "Time Series Hierarchical Event Detection Dataset",
                "version": "1.0",
                "year": 2026
            },
            "licenses": [],
            "categories": categories,
            "images": images,
            "annotations": annotations
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(annotations_dict, f, indent=2)
        
        print(f"✓ Created annotations JSON:")
        print(f"    {len(categories)} categories")
        print(f"    {len(images)} images")
        print(f"    {len(annotations)} annotations")
        print(f"    Bbox format: PIXEL coordinates ✓")
        
        return annotations_dict
    
    def timeseries_to_image(self, 
                           x: torch.Tensor,
                           annotation: HierarchicalAnnotation = None) -> np.ndarray:
        """Convert time series to image."""
        x_np = x.cpu().numpy()
        
        if self.viz_type == 'line_plot':
            return self._create_line_plot_fast(x_np)
        elif self.viz_type == 'heatmap':
            return self._create_heatmap(x_np)
        else:
            raise ValueError(f"Unknown viz_type: {self.viz_type}")
    
    def _create_line_plot_fast(self, x: np.ndarray) -> np.ndarray:
        """Fast line plot without matplotlib"""
        img = Image.new('RGB', (self.image_width, self.image_height), color='white')
        draw = ImageDraw.Draw(img)
        
        x_min, x_max = x.min(), x.max()
        if x_max - x_min < 1e-8:
            x_norm = np.ones_like(x) * 0.5
        else:
            x_norm = (x - x_min) / (x_max - x_min)
        
        margin = int(self.image_height * 0.05)
        y_coords = self.image_height - margin - (x_norm * (self.image_height - 2*margin)).astype(int)
        x_coords = np.linspace(0, self.image_width-1, len(x)).astype(int)
        
        points = list(zip(x_coords.tolist(), y_coords.tolist()))
        if len(points) > 1:
            draw.line(points, fill='blue', width=1)
        
        return np.array(img)
    
    def _create_heatmap(self, x: np.ndarray) -> np.ndarray:
        """Create heatmap by reshaping time series"""
        L = len(x)
        rows = int(np.sqrt(L))
        cols = L // rows
        x_2d = x[:rows*cols].reshape(rows, cols)
        x_norm = (x_2d - x_2d.min()) / (x_2d.max() - x_2d.min() + 1e-8)
        img = (x_norm[:, :, None] * np.array([255, 128, 0])).astype(np.uint8)
        img_pil = Image.fromarray(img).resize((self.image_width, self.image_height))
        return np.array(img_pil)


def create_tfrecords_simple_with_progress(converter, dataset, output_path):
    """
    MEMORY-EFFICIENT: Process one sequence at a time with progress tracking
    Stores PIXEL coordinates (Pix2Seq normalizes during loading)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    num_examples = len(dataset)
    print(f"\nCreating TFRecord: {output_path}")
    print(f"Processing {num_examples} sequences...")
    print(f"Progress: ", end='', flush=True)
    
    written_count = 0
    progress_interval = max(1, num_examples // 50)  # Update 50 times
    
    with tf.io.TFRecordWriter(output_path) as writer:
        for idx in range(num_examples):
            try:
                annotation = dataset[idx]
                x = annotation.sequence
                L = len(x)
                
                # Convert to image
                img_array = converter.timeseries_to_image(x, annotation)
                
                # Encode image
                img_pil = Image.fromarray(img_array)
                buffer = io.BytesIO()
                img_pil.save(buffer, format='PNG')
                image_encoded = buffer.getvalue()
                
                # Extract events
                events = [e for e in annotation.all_events if e.event_type != "regime"]
                
                if len(events) == 0:
                    continue
                
                # Prepare bounding boxes (PIXEL coordinates)
                bboxes_ymin, bboxes_xmin, bboxes_ymax, bboxes_xmax = [], [], [], []
                labels, areas, is_crowd = [], [], []
                
                W = float(converter.image_width)
                H = float(converter.image_height)
                
                for event in events:
                    # PIXEL coordinates (Pix2Seq will normalize during loading)
                    xmin = (event.start / L) * W
                    xmax = ((event.end + 1) / L) * W
                    xmax = min(W, xmax)
                    ymin, ymax = 0.0, H

                    bboxes_ymin.append(ymin)  # PIXELS
                    bboxes_xmin.append(xmin)  # PIXELS
                    bboxes_ymax.append(ymax)  # PIXELS
                    bboxes_xmax.append(xmax)  # PIXELS
                    labels.append(event.label)
                    
                    area = (ymax - ymin) * (xmax - xmin)
                    areas.append(area)
                    is_crowd.append(0)
                
                # Create TFRecord feature
                feature = {
                    'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_encoded])),
                    'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[converter.image_height])),
                    'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[converter.image_width])),
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
                written_count += 1
                
                # Progress indicator
                if (idx + 1) % progress_interval == 0:
                    pct = 100 * (idx + 1) / num_examples
                    print(f"{pct:.0f}%...", end='', flush=True)
                
                # Periodic garbage collection
                if (idx + 1) % 1000 == 0:
                    gc.collect()
                    
            except Exception as e:
                print(f"\nError processing sequence {idx}: {e}")
                continue
    
    print(f" Done!")
    print(f"✓ Created TFRecord with {written_count}/{num_examples} sequences")
    print(f"  Bbox format: PIXEL coordinates (Pix2Seq will normalize) ✓")
    
    # Final cleanup
    gc.collect()


def convert_timeseries_dataset_to_coco(
    train_dataset: CompleteHierarchicalEventDataset = None,
    val_dataset: CompleteHierarchicalEventDataset = None,
    viz_type: str = 'line_plot',
    output_dir: str = '/projects/pix2seqdata/ts_coco',
    image_size: int = 64
):
    """Complete pipeline - CORRECT VERSION with PIXEL coordinates"""
    print("\n" + "="*80)
    print("TIME SERIES → COCO FORMAT CONVERSION")
    print("Using PIXEL coordinates (Pix2Seq standard)")
    print("="*80)
    
    print(f"\nConfiguration:")
    print(f"  - Image size: {image_size}x{image_size}")
    print(f"  - Visualization: {viz_type}")
    print(f"  - Output: {output_dir}")
    print(f"  - Coordinate format: PIXEL (Pix2Seq normalizes during loading)")
    
    print(f"\n[1/4] Dataset Information")
    print(f"  - Train sequences: {len(train_dataset)}")
    print(f"  - Val sequences: {len(val_dataset)}")
    
    print(f"\n[2/4] Initializing Converter")
    converter = TimeSeriesAnnotationToCOCO(
        image_height=image_size,
        image_width=image_size,
        viz_type=viz_type
    )
    print(f"  ✓ Converter ready")
    
    print(f"\n[3/4] Creating COCO Annotations (PIXEL coordinates)")
    converter.create_annotations_json(
        train_dataset,
        output_path=f'{output_dir}/annotations/instances_train.json'
    )
    converter.create_annotations_json(
        val_dataset,
        output_path=f'{output_dir}/annotations/instances_val.json'
    )
    
    print(f"\n[4/4] Creating TFRecords (PIXEL coordinates)")
    
    print(f"\n--- Training Data ---")
    create_tfrecords_simple_with_progress(
        converter=converter,
        dataset=train_dataset,
        output_path=f'{output_dir}/tfrecords/train-00000-of-00001.tfrecord'
    )
    
    print(f"\n--- Validation Data ---")
    create_tfrecords_simple_with_progress(
        converter=converter,
        dataset=val_dataset,
        output_path=f'{output_dir}/tfrecords/val-00000-of-00001.tfrecord'
    )
    
    print("\n" + "="*80)
    print("✓ CONVERSION COMPLETE!")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nFiles created:")
    print(f"  - {output_dir}/annotations/instances_train.json")
    print(f"  - {output_dir}/annotations/instances_val.json")
    print(f"  - {output_dir}/tfrecords/train-00000-of-00001.tfrecord")
    print(f"  - {output_dir}/tfrecords/val-00000-of-00001.tfrecord")
    print(f"\n✅ Both JSON and TFRecords use PIXEL coordinates")
    print(f"✅ Pix2Seq will normalize to [0,1] during data loading")


if __name__ == "__main__":
    import torch
    import hierarchical_event_labeling as hel
    
    print("Loading datasets...")
    torch.serialization.add_safe_globals([hel.CompleteHierarchicalEventDataset])
    
    train_dataset = torch.load('data/HAR/train_event_dataset.pt', weights_only=False)
    val_dataset = torch.load('data/HAR/val_event_dataset.pt', weights_only=False)
    
    print(f"Loaded: {len(train_dataset)} train, {len(val_dataset)} val")
    
    convert_timeseries_dataset_to_coco(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        viz_type='line_plot',
        output_dir=f'{DIR}/ts_coco',
        image_size=64
    )