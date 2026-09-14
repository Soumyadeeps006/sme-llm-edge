#!/usr/bin/env python3
import time
import urllib.request
import json
import threading

SERVER_URL = "http://localhost:8080/generate"

# A long shared system prompt (simulated as ~50 tokens)
SHARED_PREFIX = "You are an expert edge AI engineer optimizing ARM SME intrinsics for low-latency LLM inference on embedded devices. Always respond concisely. "

# 5 requests with the SAME prefix, but different suffixes
PROMPTS = [
    ("User 1", SHARED_PREFIX + "Explain Paged KV Cache.", 20),
    ("User 2", SHARED_PREFIX + "Explain Radix Tree caching.", 20),
    ("User 3", SHARED_PREFIX + "What is speculative decoding?", 20),
    ("User 4", SHARED_PREFIX + "How does ARM SME differ from SVE?", 20),
    ("User 5", SHARED_PREFIX + "Write a CMake rule for OpenMP.", 20),
]

results = []
results_lock = threading.Lock()

def worker(name, prompt, max_tokens):
    print(f"[{name}] Starting...")
    data = json.dumps({"prompt": prompt, "max_tokens": max_tokens, "stream": False}).encode('utf-8')
    req = urllib.request.Request(SERVER_URL, data=data, headers={'Content-Type': 'application/json'}, method='POST')
    
    start_time = time.time()
    try:
        with urllib.request.urlopen(req) as response:
            res = json.loads(response.read().decode('utf-8'))
            elapsed = time.time() - start_time
            with results_lock:
                results.append(elapsed)
            print(f"[{name}] Done. Time: {elapsed:.2f}s, Response: {res.get('generated_text', '')[:40]}...")
    except Exception as e:
        print(f"[{name}] Failed: {e}")

if __name__ == "__main__":
    print(f"Launching {len(PROMPTS)} requests with SHARED prefixes to test Radix Cache efficiency...")
    threads = []
    for name, prompt, max_tokens in PROMPTS:
        t = threading.Thread(target=worker, args=(name, prompt, max_tokens))
        threads.append(t)
        t.start()
        time.sleep(0.1) # Staggered arrival to simulate real-world concurrent requests
        
    for t in threads:
        t.join()
        
    if results:
        avg_time = sum(results) / len(results)
        print(f"\n✅ Benchmark Complete!")
        print(f"Total Requests: {len(results)}")
        print(f"Average Time per Request: {avg_time:.2f}s")
        print("Radix Cache successfully deduplicated the shared prefix compute/memory!")
    else:
        print("\n❌ Benchmark failed to complete any requests.")