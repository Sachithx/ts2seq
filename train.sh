#!/bin/bash
#SBATCH --job-name=pix2seq_har
#SBATCH --gpus=l40:4
#SBATCH --time=48:00:00
#SBATCH --output=/projects/pix2seqdata/outputs/train-%j.log
#SBATCH --error=/projects/pix2seqdata/outputs/train-%j.log

# Load CUDA and cuDNN FIRST (before conda)
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1

# Then load Miniconda
module load Miniconda3
source activate
conda activate test

cd ~/ts2seq

python3 run.py --mode=train \
  --model_dir=/projects/pix2seqdata/tmp/ts_model \
  --config=configs/config_det_finetune.py \
  --config.dataset.train_file_pattern='/projects/pix2seqdata/tmp/ts_coco/tfrecords/train*.tfrecord' \
  --config.dataset.val_file_pattern='/projects/pix2seqdata/tmp/ts_coco/tfrecords/val*.tfrecord' \
  --config.dataset.category_names_path='/projects/pix2seqdata/tmp/ts_coco/annotations/instances_val.json' \
  --config.dataset.coco_annotations_dir_for_metrics='/projects/pix2seqdata/tmp/ts_coco/annotations' \
  --config.dataset.train_filename_for_metrics='instances_train.json' \
  --config.dataset.val_filename_for_metrics='instances_val.json' \
  --config.task.image_size='(224, 224)' \
  --config.model.pretrained_ckpt='' \
  --config.train.batch_size=128 \
  --config.train.epochs=10 \
  --config.optimization.learning_rate=3e-5 \
  --config.model.num_encoder_layers=4 \
  --config.model.dim_att=128 \
  --config.model.dim_mlp=512 \
  --config.model.num_heads=8 \
  --config.model.num_decoder_layers=4 \
  --config.model.dim_att_dec=128 \
  --config.model.dim_mlp_dec=512 \
  --config.model.num_heads_dec=8