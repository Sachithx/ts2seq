#!/usr/bin/env python3
"""
EXAMPLE TRAINING SCENARIOS
Demonstrates different training configurations
"""

import subprocess
import sys


def run_command(cmd, description):
    """Run a command and print description."""
    print("\n" + "="*80)
    print(f"SCENARIO: {description}")
    print("="*80)
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


def scenario_1_quick_test():
    """Quick test run with minimal epochs."""
    cmd = """python main.py \
        --mode flatten \
        --epochs 5 \
        --batch-size 32 \
        --lr 1e-3 \
        --save-dir checkpoints/quick_test
    """
    return run_command(cmd, "Quick Test (5 epochs)")


def scenario_2_standard_training():
    """Standard training with default settings."""
    cmd = """python main.py \
        --mode flatten \
        --epochs 20 \
        --batch-size 32 \
        --lr 1e-3 \
        --warmup-epochs 2 \
        --save-dir checkpoints/standard
    """
    return run_command(cmd, "Standard Training (20 epochs)")


def scenario_3_high_performance():
    """High performance training with larger batch size."""
    cmd = """python main.py \
        --mode flatten \
        --epochs 30 \
        --batch-size 64 \
        --lr 2e-3 \
        --warmup-epochs 3 \
        --grad-accum-steps 1 \
        --num-workers 8 \
        --save-dir checkpoints/high_perf
    """
    return run_command(cmd, "High Performance (larger batch)")


def scenario_4_gradient_accumulation():
    """Training with gradient accumulation for effective larger batch."""
    cmd = """python main.py \
        --mode flatten \
        --epochs 25 \
        --batch-size 16 \
        --grad-accum-steps 4 \
        --lr 1e-3 \
        --warmup-epochs 3 \
        --save-dir checkpoints/grad_accum
    """
    return run_command(cmd, "Gradient Accumulation (effective batch: 64)")


def scenario_5_regularized():
    """Heavy regularization for better generalization."""
    cmd = """python main.py \
        --mode flatten \
        --epochs 40 \
        --batch-size 32 \
        --lr 5e-4 \
        --weight-decay 5e-4 \
        --dropout 0.2 \
        --early-stopping-patience 10 \
        --save-dir checkpoints/regularized
    """
    return run_command(cmd, "Heavy Regularization")


def scenario_6_fine_tuning():
    """Fine-tune the encoder (unfreeze)."""
    cmd = """python main.py \
        --mode flatten \
        --no-freeze-encoder \
        --epochs 15 \
        --batch-size 16 \
        --lr 5e-5 \
        --warmup-epochs 2 \
        --weight-decay 1e-4 \
        --save-dir checkpoints/fine_tuned
    """
    return run_command(cmd, "Fine-tuning Encoder")


def scenario_7_deep_classifier():
    """Deeper classification head."""
    cmd = """python main.py \
        --mode flatten \
        --epochs 25 \
        --batch-size 32 \
        --lr 1e-3 \
        --hidden-dims 1024 512 256 \
        --dropout 0.15 \
        --save-dir checkpoints/deep_classifier
    """
    return run_command(cmd, "Deep Classifier Head")


def scenario_8_channel_averaging():
    """Train with channel averaging instead of flattening."""
    cmd = """python main.py \
        --mode average \
        --epochs 20 \
        --batch-size 64 \
        --lr 1e-3 \
        --save-dir checkpoints/channel_avg
    """
    return run_command(cmd, "Channel Averaging Mode")


def scenario_9_profile_only():
    """Run profiling without full training."""
    script = """
from profiler import PerformanceProfiler
from models.encoder_cls import EncoderClassifier
from data_loader import create_har_dataloaders
import torch

print("Setting up...")
train_loader, _, _ = create_har_dataloaders(batch_size=32, mode='flatten')

model = EncoderClassifier(
    num_classes=6,
    pretrained_encoder_path='/home/sachithxcviii/ts2seq/data/HAR/extracted_encoder/encoder_weights.pth',
    freeze_encoder=True
)

print("Running profiler...")
profiler = PerformanceProfiler(model, train_loader, device='cuda')
results = profiler.run_full_profile(save_path='profile_results.json')

print("\\nProfiler complete! Results saved to profile_results.json")
"""
    
    print("\n" + "="*80)
    print("SCENARIO: Performance Profiling")
    print("="*80)
    
    with open('_temp_profile.py', 'w') as f:
        f.write(script)
    
    result = subprocess.run([sys.executable, '_temp_profile.py'])
    subprocess.run(['rm', '_temp_profile.py'])
    
    return result.returncode == 0


def scenario_10_batch_benchmark():
    """Benchmark different batch sizes."""
    script = """
from profiler import PerformanceProfiler
from models.encoder_cls import EncoderClassifier
from data_loader import create_har_dataloaders
import torch

print("Setting up...")
train_loader, _, _ = create_har_dataloaders(batch_size=32, mode='flatten')

model = EncoderClassifier(
    num_classes=6,
    pretrained_encoder_path='/home/sachithxcviii/ts2seq/data/HAR/extracted_encoder/encoder_weights.pth',
    freeze_encoder=True
)

print("Benchmarking batch sizes...")
profiler = PerformanceProfiler(model, train_loader, device='cuda')
results = profiler.benchmark_batch_sizes([8, 16, 32, 64, 128, 256])

print("\\nBenchmark complete!")
"""
    
    print("\n" + "="*80)
    print("SCENARIO: Batch Size Benchmarking")
    print("="*80)
    
    with open('_temp_benchmark.py', 'w') as f:
        f.write(script)
    
    result = subprocess.run([sys.executable, '_temp_benchmark.py'])
    subprocess.run(['rm', '_temp_benchmark.py'])
    
    return result.returncode == 0


def print_menu():
    """Print scenario menu."""
    print("\n" + "="*80)
    print("TRAINING SCENARIO EXAMPLES")
    print("="*80)
    print("\n1.  Quick Test (5 epochs)")
    print("2.  Standard Training (20 epochs)")
    print("3.  High Performance (larger batch)")
    print("4.  Gradient Accumulation (effective batch: 64)")
    print("5.  Heavy Regularization (better generalization)")
    print("6.  Fine-tuning Encoder (unfreeze)")
    print("7.  Deep Classifier Head")
    print("8.  Channel Averaging Mode")
    print("9.  Performance Profiling")
    print("10. Batch Size Benchmarking")
    print("\n0.  Exit")
    print("="*80)


def main():
    """Main menu."""
    scenarios = {
        1: scenario_1_quick_test,
        2: scenario_2_standard_training,
        3: scenario_3_high_performance,
        4: scenario_4_gradient_accumulation,
        5: scenario_5_regularized,
        6: scenario_6_fine_tuning,
        7: scenario_7_deep_classifier,
        8: scenario_8_channel_averaging,
        9: scenario_9_profile_only,
        10: scenario_10_batch_benchmark,
    }
    
    while True:
        print_menu()
        
        try:
            choice = int(input("\nSelect scenario (0-10): "))
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue
        
        if choice == 0:
            print("\nExiting...")
            break
        elif choice in scenarios:
            success = scenarios[choice]()
            if success:
                print(f"\n✓ Scenario {choice} completed successfully!")
            else:
                print(f"\n✗ Scenario {choice} failed!")
            
            input("\nPress Enter to continue...")
        else:
            print("Invalid choice. Please select 0-10.")


if __name__ == '__main__':
    # Check if direct scenario number provided as argument
    if len(sys.argv) > 1:
        try:
            scenario_num = int(sys.argv[1])
            scenarios = {
                1: scenario_1_quick_test,
                2: scenario_2_standard_training,
                3: scenario_3_high_performance,
                4: scenario_4_gradient_accumulation,
                5: scenario_5_regularized,
                6: scenario_6_fine_tuning,
                7: scenario_7_deep_classifier,
                8: scenario_8_channel_averaging,
                9: scenario_9_profile_only,
                10: scenario_10_batch_benchmark,
            }
            
            if scenario_num in scenarios:
                scenarios[scenario_num]()
            else:
                print(f"Invalid scenario number: {scenario_num}")
                print("Valid options: 1-10")
        except ValueError:
            print("Invalid argument. Usage: python examples.py [scenario_number]")
    else:
        # Interactive menu
        main()
