"""
PROFILING AND BENCHMARKING UTILITIES
Profile training performance, memory usage, and throughput
"""

import torch
import time
import numpy as np
from pathlib import Path
from contextlib import contextmanager
import json


class PerformanceProfiler:
    """Profile model training performance."""
    
    def __init__(self, model, dataloader, device='cuda'):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.results = {}
    
    @contextmanager
    def timer(self, name):
        """Context manager for timing operations."""
        start = time.time()
        yield
        elapsed = time.time() - start
        self.results[name] = elapsed
    
    def profile_forward_pass(self, num_iterations=100):
        """Profile forward pass speed."""
        print("\n" + "="*80)
        print("PROFILING FORWARD PASS")
        print("="*80)
        
        self.model.eval()
        
        # Warmup
        print("Warming up...")
        for i, (images, _) in enumerate(self.dataloader):
            if i >= 10:
                break
            images = images.to(self.device)
            with torch.no_grad():
                _ = self.model(images)
        
        # Profile
        print(f"Profiling {num_iterations} iterations...")
        times = []
        
        with torch.no_grad():
            for i, (images, _) in enumerate(self.dataloader):
                if i >= num_iterations:
                    break
                
                images = images.to(self.device)
                
                torch.cuda.synchronize() if self.device == 'cuda' else None
                start = time.time()
                
                _ = self.model(images)
                
                torch.cuda.synchronize() if self.device == 'cuda' else None
                elapsed = time.time() - start
                times.append(elapsed)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        throughput = self.dataloader.batch_size / avg_time
        
        print(f"\nResults:")
        print(f"  Average time: {avg_time*1000:.2f} ms")
        print(f"  Std dev: {std_time*1000:.2f} ms")
        print(f"  Throughput: {throughput:.1f} samples/sec")
        print(f"  FPS: {1/avg_time:.1f}")
        
        return {
            'avg_time_ms': avg_time * 1000,
            'std_time_ms': std_time * 1000,
            'throughput': throughput,
            'fps': 1 / avg_time
        }
    
    def profile_backward_pass(self, num_iterations=100):
        """Profile backward pass speed."""
        print("\n" + "="*80)
        print("PROFILING BACKWARD PASS")
        print("="*80)
        
        self.model.train()
        criterion = torch.nn.CrossEntropyLoss()
        
        # Warmup
        print("Warming up...")
        for i, (images, labels) in enumerate(self.dataloader):
            if i >= 10:
                break
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
        
        # Profile
        print(f"Profiling {num_iterations} iterations...")
        times = []
        
        for i, (images, labels) in enumerate(self.dataloader):
            if i >= num_iterations:
                break
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            torch.cuda.synchronize() if self.device == 'cuda' else None
            start = time.time()
            
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.cuda.synchronize() if self.device == 'cuda' else None
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        throughput = self.dataloader.batch_size / avg_time
        
        print(f"\nResults:")
        print(f"  Average time: {avg_time*1000:.2f} ms")
        print(f"  Std dev: {std_time*1000:.2f} ms")
        print(f"  Throughput: {throughput:.1f} samples/sec")
        
        return {
            'avg_time_ms': avg_time * 1000,
            'std_time_ms': std_time * 1000,
            'throughput': throughput
        }
    
    def profile_memory(self):
        """Profile GPU memory usage."""
        if self.device != 'cuda':
            print("\n⚠ Memory profiling only available for CUDA devices")
            return None
        
        print("\n" + "="*80)
        print("PROFILING GPU MEMORY")
        print("="*80)
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        self.model.train()
        criterion = torch.nn.CrossEntropyLoss()
        
        # Get one batch
        images, labels = next(iter(self.dataloader))
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Measure
        torch.cuda.reset_peak_memory_stats()
        
        outputs = self.model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        
        print(f"\nMemory usage:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
        print(f"  Peak: {peak:.2f} GB")
        print(f"  Batch size: {images.size(0)}")
        print(f"  Per sample: {peak * 1000 / images.size(0):.1f} MB")
        
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'peak_gb': peak,
            'batch_size': images.size(0),
            'per_sample_mb': peak * 1000 / images.size(0)
        }
    
    def benchmark_batch_sizes(self, batch_sizes=[8, 16, 32, 64, 128, 256]):
        """Benchmark different batch sizes."""
        print("\n" + "="*80)
        print("BENCHMARKING BATCH SIZES")
        print("="*80)
        
        results = {}
        
        for bs in batch_sizes:
            print(f"\nTesting batch size: {bs}")
            
            # Create new dataloader with this batch size
            from torch.utils.data import DataLoader
            new_loader = DataLoader(
                self.dataloader.dataset,
                batch_size=bs,
                shuffle=False,
                num_workers=self.dataloader.num_workers,
                pin_memory=True
            )
            
            try:
                # Test memory
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                images, labels = next(iter(new_loader))
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                self.model.train()
                outputs = self.model(images)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                loss.backward()
                
                peak_memory = torch.cuda.max_memory_allocated() / 1e9
                
                # Test speed
                times = []
                with torch.no_grad():
                    for i, (imgs, _) in enumerate(new_loader):
                        if i >= 20:
                            break
                        imgs = imgs.to(self.device)
                        
                        torch.cuda.synchronize()
                        start = time.time()
                        _ = self.model(imgs)
                        torch.cuda.synchronize()
                        times.append(time.time() - start)
                
                avg_time = np.mean(times)
                throughput = bs / avg_time
                
                results[bs] = {
                    'peak_memory_gb': peak_memory,
                    'avg_time_ms': avg_time * 1000,
                    'throughput': throughput,
                    'success': True
                }
                
                print(f"  ✓ Peak memory: {peak_memory:.2f} GB")
                print(f"  ✓ Time: {avg_time*1000:.2f} ms")
                print(f"  ✓ Throughput: {throughput:.1f} samples/s")
                
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"  ✗ OOM - Batch size too large")
                    results[bs] = {'success': False, 'error': 'OOM'}
                    torch.cuda.empty_cache()
                    break
                else:
                    raise
        
        # Print summary
        print("\n" + "="*80)
        print("BATCH SIZE SUMMARY")
        print("="*80)
        print(f"{'Batch Size':<12} {'Memory (GB)':<15} {'Time (ms)':<12} {'Throughput':<15}")
        print("-" * 80)
        
        for bs, res in results.items():
            if res['success']:
                print(f"{bs:<12} {res['peak_memory_gb']:<15.2f} {res['avg_time_ms']:<12.2f} {res['throughput']:<15.1f}")
            else:
                print(f"{bs:<12} {'OOM':<15} {'-':<12} {'-':<15}")
        
        return results
    
    def run_full_profile(self, save_path='profile_results.json'):
        """Run complete profiling suite."""
        print("\n" + "="*80)
        print("RUNNING FULL PERFORMANCE PROFILE")
        print("="*80)
        
        all_results = {}
        
        # Forward pass
        all_results['forward_pass'] = self.profile_forward_pass()
        
        # Backward pass
        all_results['backward_pass'] = self.profile_backward_pass()
        
        # Memory
        all_results['memory'] = self.profile_memory()
        
        # Save results
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\n✓ Results saved to: {save_path}")
        
        return all_results


def compare_amp_performance(model, dataloader, device='cuda', num_iterations=50):
    """Compare performance with and without AMP."""
    print("\n" + "="*80)
    print("COMPARING AMP PERFORMANCE")
    print("="*80)
    
    from torch.amp import autocast
    from torch.cuda.amp import GradScaler
    
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    
    results = {}
    
    # Test without AMP
    print("\n[1/2] Testing WITHOUT AMP...")
    model.train()
    times_no_amp = []
    
    for i, (images, labels) in enumerate(dataloader):
        if i >= num_iterations:
            break
        
        images = images.to(device)
        labels = labels.to(device)
        
        torch.cuda.synchronize()
        start = time.time()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        torch.cuda.synchronize()
        times_no_amp.append(time.time() - start)
    
    avg_no_amp = np.mean(times_no_amp)
    throughput_no_amp = dataloader.batch_size / avg_no_amp
    
    print(f"  Average time: {avg_no_amp*1000:.2f} ms")
    print(f"  Throughput: {throughput_no_amp:.1f} samples/s")
    
    # Test with AMP
    print("\n[2/2] Testing WITH AMP...")
    model.train()
    scaler = GradScaler()
    times_amp = []
    
    for i, (images, labels) in enumerate(dataloader):
        if i >= num_iterations:
            break
        
        images = images.to(device)
        labels = labels.to(device)
        
        torch.cuda.synchronize()
        start = time.time()
        
        with autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(torch.optim.SGD(model.parameters(), lr=0.01))
        scaler.update()
        
        torch.cuda.synchronize()
        times_amp.append(time.time() - start)
    
    avg_amp = np.mean(times_amp)
    throughput_amp = dataloader.batch_size / avg_amp
    
    print(f"  Average time: {avg_amp*1000:.2f} ms")
    print(f"  Throughput: {throughput_amp:.1f} samples/s")
    
    # Comparison
    speedup = avg_no_amp / avg_amp
    
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Time reduction: {(1 - avg_amp/avg_no_amp) * 100:.1f}%")
    print(f"  Throughput increase: {(throughput_amp/throughput_no_amp - 1) * 100:.1f}%")
    
    return {
        'no_amp': {'time_ms': avg_no_amp * 1000, 'throughput': throughput_no_amp},
        'with_amp': {'time_ms': avg_amp * 1000, 'throughput': throughput_amp},
        'speedup': speedup
    }


if __name__ == '__main__':
    # Example usage
    print("This is a utility module. Import and use the profiling functions.")
    print("\nExample:")
    print("  from profiler import PerformanceProfiler")
    print("  profiler = PerformanceProfiler(model, dataloader)")
    print("  profiler.run_full_profile()")
