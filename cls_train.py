
# %%
# ============================================================================
# RUN TRAINING
# ============================================================================

import os
from classification import train_har_classifier


if __name__ == '__main__':
    # Training with FLATTEN mode (recommended)
    print("\n" + "="*80)
    print("TRAINING WITH FLATTEN MODE (9x more samples)")
    print("="*80)
    
    model_flatten, results_flatten = train_har_classifier(
        mode='flatten',
        batch_size=32,
        num_epochs=4,
        learning_rate=1e-3,
        save_dir='checkpoints/har_flatten',
        pretrained_encoder_path='/home/sachithxcviii/ts2seq/data/HAR/extracted_encoder/encoder_weights.pth',
        data_dir='/home/sachithxcviii/ts2seq/data/HAR/multichannel_images'
    )
    