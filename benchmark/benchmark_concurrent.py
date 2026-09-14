#!/usr/bin/env python3
import time
import urllib.request
import json
import threading

SERVER_URL = "http://localhost:8080/generate"
PROMPTS = [
    "The future of edge AI is",
    "ARM SME acceleration provides",
    "Zero-copy memory mapping reduces",
    "OpenMP parallelization unlocks",
    "Paged KV cache prevents"
]
MAX_TOKENS = 20

def worker(prompt, worker_id):
    print(f"[Worker {worker_id}] Starting: {prompt}")
    data = json.dumps({"prompt": prompt, "max_tokens": MAX_TOKENS, "stream": True}).encode('utf-8')
    req = urllib.request.Request(SERVER_URL, data=data, headers={'Content-Type': 'application/json'}, method='POST')
    
    start_time = time.time()
    token_count = 0
    
    try:
        with urllib.request.urlopen(req) as response:
            buffer = ""
            while True:
                chunk = response.read(1).decode('utf-8')
                if not chunk: break
                buffer += chunk
                while "\n\n" in buffer:
                    line, buffer = buffer.split("\n\n", 1)
                    if line.startswith("data: ") and line[6:] != "[DONE]":
                        token_count += 1
            
            elapsed = time.time() - start_time
            print(f"[Worker {worker_id}] Done. Tokens: {token_count}, Time: {elapsed:.2f}s, Throughput: {token_count/elapsed:.2f} t/s")
    except Exception as e:
        print(f"[Worker {worker_id}] Failed: {e}")

if __name__ == "__main__":
    print(f"Launching {len(PROMPTS)} concurrent requests...")
    threads = []
    for i, prompt in enumerate(PROMPTS):
        t = threading.Thread(target=worker, args=(prompt, i))
        threads.append(t)
        t.start()
        time.sleep(0.1) # Slight stagger to simulate real-world arrival
        
    for t in threads:
        t.join()
    print("All concurrent requests completed.")