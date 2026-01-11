#!/bin/bash
#SBATCH --job-name=pix2seq_eval
#SBATCH --gpus=l40:4
#SBATCH --time=4:00:00
#SBATCH --output=/projects/pix2seqdata/outputs/eval-%j.log
#SBATCH --error=/projects/pix2seqdata/outputs/eval-%j.log

# Load CUDA and cuDNN FIRST (before conda)
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1

# Then load Miniconda
module load Miniconda3
source activate
conda activate test

cd ~/pix2seq

# Run eval with ALL necessary config overrides
python3 run.py --mode=eval \
  --model_dir=/projects/pix2seqdata/tmp/ts_model \
  --config=configs/config_det_finetune.py \
  --config.dataset.val_file_pattern='/projects/pix2seqdata/tmp/ts_coco/tfrecords/val*.tfrecord' \
  --config.dataset.category_names_path='/projects/pix2seqdata/tmp/ts_coco/annotations/instances_val.json' \
  --config.dataset.coco_annotations_dir_for_metrics='/projects/pix2seqdata/tmp/ts_coco/annotations' \
  --config.dataset.val_filename_for_metrics='instances_val.json' \
  --config.task.image_size='(224, 224)' \
  --config.model.pretrained_ckpt='' \
  --config.eval.batch_size=8 \
  --config.eval.steps=0 