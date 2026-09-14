import numpy as np
import matplotlib.pyplot as plt

def simulate_mixed_workload(enable_chunking=True, chunk_size=256):
    """
    Simulates a mixed workload: 8 concurrent decode requests + 1 large prefill (2048 tokens).
    Measures the queuing delay (latency spike) experienced by decode requests.
    """
    num_decode_requests = 8
    prefill_tokens = 2048
    decode_tokens_per_req = 10
    max_batch_size = 8
    
    # Time constants (mocking hardware execution time)
    # Prefill is O(N^2) due to KV cache expansion and attention, Decode is O(N)
    prefill_time_per_token_sq = 0.00005  # seconds
    decode_time_per_token = 0.010        # seconds (10ms)
    
    decode_latencies = []
    
    prefill_remaining = prefill_tokens
    decode_remaining = [decode_tokens_per_req] * num_decode_requests
    current_time = 0.0
    
    step = 0
    while prefill_remaining > 0 or sum(decode_remaining) > 0:
        step += 1
        
        # Scheduler logic
        if enable_chunking:
            tokens_to_process = min(chunk_size, prefill_remaining)
            prefill_time = (tokens_to_process ** 2) * prefill_time_per_token_sq
            prefill_remaining -= tokens_to_process
        else:
            if prefill_remaining > 0:
                tokens_to_process = prefill_remaining
                prefill_time = (tokens_to_process ** 2) * prefill_time_per_token_sq
                prefill_remaining = 0
            else:
                prefill_time = 0.0
                
        if not enable_chunking and prefill_remaining == 0 and step > 1:
            step_duration = sum(1 for r in decode_remaining if r > 0) * decode_time_per_token
        else:
            if enable_chunking:
                step_duration = prefill_time + (sum(1 for r in decode_remaining if r > 0) * decode_time_per_token)
            else:
                step_duration = prefill_time
                
        current_time += step_duration
        
        can_decode = enable_chunking or (prefill_remaining == 0)
        
        if can_decode:
            for i in range(num_decode_requests):
                if decode_remaining[i] > 0:
                    decode_remaining[i] -= 1
                    decode_latencies.append(step_duration * 1000) # ms
                    
    return decode_latencies

if __name__ == "__main__":
    print("🚀 Running Day 28 Chunked Prefill Benchmark...")
    print("-" * 55)
    
    print("Simulating WITHOUT chunking (Monolithic Prefill)...")
    latencies_no_chunk = simulate_mixed_workload(enable_chunking=False)
    max_lat_no_chunk = max(latencies_no_chunk) if latencies_no_chunk else 0
    
    print("Simulating WITH chunking (Chunk Size = 256)...")
    latencies_chunk = simulate_mixed_workload(enable_chunking=True, chunk_size=256)
    max_lat_chunk = max(latencies_chunk) if latencies_chunk else 0
    
    print("-" * 55)
    print(f"📊 Results:")
    print(f"  Max Decode Latency Spike (No Chunking) : {max_lat_no_chunk:>8.2f} ms")
    print(f"  Max Decode Latency Spike (With Chunking): {max_lat_chunk:>8.2f} ms")
    
    improvement = ((max_lat_no_chunk - max_lat_chunk) / max_lat_no_chunk) * 100 if max_lat_no_chunk > 0 else 0
    print(f"  Latency Spike Reduction                : {improvement:>8.1f} %")
    
    plt.figure(figsize=(10, 5))
    
    max_steps = max(len(latencies_no_chunk), len(latencies_chunk))
    x = np.arange(max_steps)
    
    plt.plot(x[:len(latencies_no_chunk)], latencies_no_chunk, 
             label='Without Chunking (Starvation)', color='red', marker='o', alpha=0.7)
    plt.plot(x[:len(latencies_chunk)], latencies_chunk, 
             label='With Chunking (256 tokens)', color='green', marker='s', alpha=0.7)
    
    plt.axhline(y=50, color='orange', linestyle='--', linewidth=2, label='50ms SLO Target')
    
    plt.title('Day 28: Decode Latency Spike (Mixed Workload: 1x2048 Prefill + 8x Decode)')
    plt.xlabel('Decode Step Completion Event')
    plt.ylabel('Latency per Step (ms)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('benchmark_chunked_prefill_results.png', dpi=150)
    print("\n✅ Saved plot to 'benchmark_chunked_prefill_results.png'")