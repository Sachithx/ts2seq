import io
import os
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
import json
import random

TFRECORD = "/tmp/ts_coco/tfrecords/train-00000-of-00001.tfrecord"
ANNOTATIONS = "/tmp/ts_coco/annotations/instances_train.json"
OUTDIR = "/usr1/home/s124mdg54_01/ts2seq/visualizations"
NUM_SHOW = 20
MAX_EVENTS_TO_SHOW = 2  # Show only 2 random EVENTS (box + label) per image

os.makedirs(OUTDIR, exist_ok=True)

# Load category names
with open(ANNOTATIONS, 'r') as f:
    coco_data = json.load(f)
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

feature_desc = {
    "image/encoded": tf.io.FixedLenFeature([], tf.string),
    "image/height": tf.io.FixedLenFeature([], tf.int64),
    "image/width": tf.io.FixedLenFeature([], tf.int64),
    "image/filename": tf.io.FixedLenFeature([], tf.string),
    "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
    "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
    "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
    "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
    "image/object/class/label": tf.io.VarLenFeature(tf.int64),
}

def dense(v):
    return tf.sparse.to_dense(v).numpy()

def infer_fmt(xmin, xmax, ymin, ymax):
    m = max(
        float(np.max(xmin)) if len(xmin) else 0.0,
        float(np.max(xmax)) if len(xmax) else 0.0,
        float(np.max(ymin)) if len(ymin) else 0.0,
        float(np.max(ymax)) if len(ymax) else 0.0,
    )
    return "normalized" if m <= 1.5 else "pixel"

def visualize_example(img, xmin, xmax, ymin, ymax, labels, max_events=MAX_EVENTS_TO_SHOW):
    W, H = img.size
    draw = ImageDraw.Draw(img)
    fmt = infer_fmt(xmin, xmax, ymin, ymax)
    
    colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'orange', 'pink']
    
    num_events = len(labels)
    
    # Randomly select which events to show (BOTH box and label)
    if num_events > max_events:
        show_indices = random.sample(range(num_events), max_events)
        print(f"    Showing {max_events} out of {num_events} total events")
    else:
        show_indices = list(range(num_events))
        print(f"    Showing all {num_events} events")

    shown_labels = []
    
    # ONLY draw selected events
    for idx, i in enumerate(show_indices):
        x1, x2, y1, y2 = float(xmin[i]), float(xmax[i]), float(ymin[i]), float(ymax[i])
        if fmt == "normalized":
            x1, x2 = x1 * W, x2 * W
            y1, y2 = y1 * H, y2 * H

        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        
        label_id = int(labels[i])
        label_name = categories.get(label_id, f"Unknown_{label_id}")
        color = colors[idx % len(colors)]

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label text
        label_text = label_name[:15]
        text_w = len(label_text) * 7
        text_h = 12
        
        # Black background for text
        draw.rectangle([x1, y1, x1 + text_w, y1 + text_h], fill='black')
        draw.text((x1 + 2, y1 + 2), label_text, fill=color)
        
        shown_labels.append(label_name)

    return fmt, num_events, len(shown_labels), shown_labels

ds = tf.data.TFRecordDataset([TFRECORD])

print(f"Event categories in dataset:")
for cat_id, cat_name in sorted(categories.items()):
    print(f"  {cat_id}: {cat_name}")
print()
print(f"Will show max {MAX_EVENTS_TO_SHOW} events (box + label) per image\n")

for idx, raw in enumerate(ds.take(NUM_SHOW)):
    ex = tf.io.parse_single_example(raw, feature_desc)

    img_bytes = ex["image/encoded"].numpy()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    xmin = dense(ex["image/object/bbox/xmin"])
    xmax = dense(ex["image/object/bbox/xmax"])
    ymin = dense(ex["image/object/bbox/ymin"])
    ymax = dense(ex["image/object/bbox/ymax"])
    labels = dense(ex["image/object/class/label"])

    print(f"Example {idx}:")
    fmt, total_events, shown_events, shown_labels = visualize_example(img, xmin, xmax, ymin, ymax, labels)

    out_path = os.path.join(OUTDIR, f"ex_{idx:03d}_showing_{shown_events}_of_{total_events}.png")
    img.save(out_path)
    
    print(f"    Shown: {', '.join(shown_labels)}\n")

print(f"\nSaved {NUM_SHOW} visualizations to: {OUTDIR}")
print(f"Each image shows ONLY {MAX_EVENTS_TO_SHOW} randomly selected events")
print(f"This keeps images clean and readable!")