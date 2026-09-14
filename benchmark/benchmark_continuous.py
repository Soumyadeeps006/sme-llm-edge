#!/usr/bin/env python3
import time
import urllib.request
import json
import threading

SERVER_URL = "http://localhost:8080/generate"

# Mix of short, medium, and long prompts to test in-flight batching
PROMPTS = [
    ("Short 1", "AI is", 10),
    ("Short 2", "Edge computing", 15),
    ("Medium 1", "The future of ARM SME acceleration in", 30),
    ("Medium 2", "Continuous batching improves throughput by", 30),
    ("Long 1", "In a world where edge devices must run large language models efficiently, zero-copy memory mapping and paged KV caches are essential because", 50),
]

def worker(name, prompt, max_tokens):
    print(f"[{name}] Starting...")
    data = json.dumps({"prompt": prompt, "max_tokens": max_tokens, "stream": True}).encode('utf-8')
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
            print(f"[{name}] Done. Tokens: {token_count}, Time: {elapsed:.2f}s, Throughput: {token_count/elapsed:.2f} t/s")
    except Exception as e:
        print(f"[{name}] Failed: {e}")

if __name__ == "__main__":
    print(f"Launching {len(PROMPTS)} continuous requests with varying lengths...")
    threads = []
    for name, prompt, max_tokens in PROMPTS:
        t = threading.Thread(target=worker, args=(name, prompt, max_tokens))
        threads.append(t)
        t.start()
        time.sleep(0.2) # Staggered arrival to simulate real-world continuous batching
        
    for t in threads:
        t.join()
    print("All continuous requests completed.")