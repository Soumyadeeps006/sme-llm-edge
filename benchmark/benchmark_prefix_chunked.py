#!/usr/bin/env python3
import time
import urllib.request
import json
import threading

SERVER_URL = "http://localhost:8080/generate"

# ~200 tokens system prompt
SYSTEM_PROMPT = "You are an expert edge AI engineer. Always optimize for low latency, minimal memory footprint, and ARM SME acceleration. " * 10

def worker(name, prompt_suffix, max_tokens, is_decode_test=False):
    print(f"[{name}] Starting...")
    full_prompt = SYSTEM_PROMPT + prompt_suffix
    data = json.dumps({
        "prompt": full_prompt, 
        "max_tokens": max_tokens, 
        "stream": is_decode_test
    }).encode('utf-8')
    
    req = urllib.request.Request(
        SERVER_URL, 
        data=data, 
        headers={'Content-Type': 'application/json'}, 
        method='POST'
    )
    
    start_time = time.time()
    token_count = 0
    
    try:
        with urllib.request.urlopen(req) as response:
            if is_decode_test:
                buffer = ""
                while True:
                    chunk = response.read(1).decode('utf-8')
                    if not chunk: 
                        break
                    buffer += chunk
                    while "\n\n" in buffer:
                        line, buffer = buffer.split("\n\n", 1)
                        if line.startswith("data: ") and line[6:] != "[DONE]":
                            token_count += 1
            else:
                response.read() # Just wait for completion
                
            elapsed = time.time() - start_time
            tps = token_count / elapsed if is_decode_test else 0
            print(f"[{name}] Done. Time: {elapsed:.2f}s | Throughput: {tps:.2f} t/s")
            
    except Exception as e:
        print(f"[{name}] Failed: {e}")

if __name__ == "__main__":
    print("🚀 Launching Chunked Prefill & Prefix Caching Benchmark...")
    print(f"System Prompt Length: ~{len(SYSTEM_PROMPT.split())} words")
    
    # Request 1: Long prefill (should be chunked, allowing interleaving)
    t1 = threading.Thread(target=worker, args=("Long Prefill", "Explain ARM SME in detail. ", 20, False))
    
    # Request 2: Short decode (should experience minimal latency spike due to chunking & prefix hit)
    time.sleep(0.1) # Ensure Request 1 starts first and holds the batch
    t2 = threading.Thread(target=worker, args=("Short Decode", "Hi", 5, True))
    
    # Request 3: Another long prefill (should instantly hit prefix cache for SYSTEM_PROMPT)
    time.sleep(0.1)
    t3 = threading.Thread(target=worker, args=("Prefix Hit", "What is Paged Attention? ", 20, False))
    
    t1.start()
    t2.start()
    t3.start()
    
    t1.join()
    t2.join()
    t3.join()
    
    print("✅ Benchmark Complete. Check server logs for prefix hit rates and chunking interleaving.")