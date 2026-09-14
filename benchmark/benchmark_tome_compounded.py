import numpy as np

def simulate_tome_compounded_benchmark():
    batch_size = 16
    initial_seq_len = 2048
    hidden_dim = 4096
    
    # Day 32: 30% early exit rate
    early_exit_ratio = 0.30
    # Day 33: 20% token merge rate via ToMe
    tome_merge_ratio = 0.20
    
    print("="*60)
    print("Day 33: ToMe + Early Exit Compounded Benchmark")
    print("="*60)
    print(f"Batch Size: {batch_size}")
    print(f"Initial Sequence Length: {initial_seq_len}")
    print(f"Early Exit Ratio (Day 32): {early_exit_ratio*100}%")
    print(f"ToMe Merge Ratio (Day 33): {tome_merge_ratio*100}%")
    print("-" * 60)
    
    active_sequences = int(batch_size * (1.0 - early_exit_ratio))
    effective_seq_len = int(initial_seq_len * (1.0 - tome_merge_ratio))
    
    baseline_attention_flops = batch_size * (initial_seq_len ** 2) * hidden_dim * 2
    optimized_attention_flops = active_sequences * (effective_seq_len ** 2) * hidden_dim * 2
    
    attention_reduction = (baseline_attention_flops - optimized_attention_flops) / baseline_attention_flops * 100
    
    baseline_kv_memory = batch_size * initial_seq_len * hidden_dim * 2 * 2
    optimized_kv_memory = active_sequences * effective_seq_len * hidden_dim * 2 * 2
    
    kv_memory_reduction = (baseline_kv_memory - optimized_kv_memory) / baseline_kv_memory * 100
    
    print("Theoretical Performance Savings:")
    print(f"  - Baseline Active Sequences: {batch_size} -> Optimized: {active_sequences}")
    print(f"  - Baseline Seq Length: {initial_seq_len} -> Optimized: {effective_seq_len}")
    print("-" * 60)
    print(f"  - Baseline Attention FLOPs: {baseline_attention_flops / 1e9:.2f} GFLOPs")
    print(f"  - Optimized Attention FLOPs: {optimized_attention_flops / 1e9:.2f} GFLOPs")
    print(f"  - Attention FLOPs Reduction: {attention_reduction:.1f}%")
    print("-" * 60)
    print(f"  - Baseline KV Cache Memory: {baseline_kv_memory / 1e6:.2f} MB")
    print(f"  - Optimized KV Cache Memory: {optimized_kv_memory / 1e6:.2f} MB")
    print(f"  - KV Cache Memory Reduction: {kv_memory_reduction:.1f}%")
    print("="*60)
    
    if kv_memory_reduction >= 40.0 and attention_reduction >= 40.0:
        print("✅ SUCCESS: Compounded reduction > 40% target achieved for both FLOPs and Memory!")
    else:
        print("⚠️  WARNING: Compounded reduction below 40% target.")

if __name__ == "__main__":
    simulate_tome_compounded_benchmark()