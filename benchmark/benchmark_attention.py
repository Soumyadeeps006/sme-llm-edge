#!/usr/bin/env python3
import time
import urllib.request
import json

SERVER_URL = "http://localhost:8080/generate"

def benchmark_request(name, prompt, max_tokens):
    print(f"[{name}] Starting...")
    data = json.dumps({"prompt": prompt, "max_tokens": max_tokens, "stream": False}).encode('utf-8')
    req = urllib.request.Request(SERVER_URL, data=data, headers={'Content-Type': 'application/json'}, method='POST')
    
    start_time = time.time()
    try:
        with urllib.request.urlopen(req) as response:
            response.read()
            elapsed = time.time() - start_time
            print(f"[{name}] Done. Total Time: {elapsed:.2f}s")
    except Exception as e:
        print(f"[{name}] Failed: {e}")

if __name__ == "__main__":
    print("🚀 Launching Paged Attention Kernel Benchmark...")
    
    # Test 1: Short context (measures baseline decode TPOT)
    benchmark_request("Short Context", "Hello, how are you?", 20)
    
    # Test 2: Long context (measures TTFT and attention scaling)
    long_prompt = "Optimize for low latency. " * 100 # ~500 tokens
    benchmark_request("Long Context", long_prompt, 20)
    
    print("✅ Benchmark Complete. Check server logs for execution time drops in the attention phase.")