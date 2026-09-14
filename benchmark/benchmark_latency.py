#!/usr/bin/env python3
import time
import urllib.request
import json

SERVER_URL = "http://localhost:8080/generate"
PROMPT = "the future of edge ai is"
MAX_TOKENS = 30

def benchmark():
    print(f"Starting streaming benchmark: {MAX_TOKENS} tokens...")
    data = json.dumps({"prompt": PROMPT, "max_tokens": MAX_TOKENS, "stream": True}).encode('utf-8')
    req = urllib.request.Request(SERVER_URL, data=data, headers={'Content-Type': 'application/json'}, method='POST')
    
    start_time = time.time()
    first_token_time = None
    token_count = 0
    
    try:
        with urllib.request.urlopen(req) as response:
            buffer = ""
            while True:
                chunk = response.read(1).decode('utf-8')
                if not chunk:
                    break
                buffer += chunk
                
                # Process complete SSE lines
                while "\n\n" in buffer:
                    line, buffer = buffer.split("\n\n", 1)
                    if line.startswith("data: "):
                        payload = line[6:]
                        if payload == "[DONE]":
                            continue
                        
                        if first_token_time is None:
                            first_token_time = time.time()
                            
                        token_data = json.loads(payload)
                        print(token_data.get("token", ""), end="", flush=True)
                        token_count += 1

            end_time = time.time()
            
            print("\n" + "-" * 40)
            ttft = (first_token_time - start_time) if first_token_time else 0
            total_time = end_time - start_time
            gen_time = end_time - (first_token_time or start_time)
            
            print(f"Time to First Token (TTFT): {ttft:.4f} seconds")
            print(f"Total Generation Time: {gen_time:.4f} seconds")
            print(f"Throughput: {(token_count / gen_time):.2f} tokens/sec")
            print("-" * 40)
            
    except Exception as e:
        print(f"\nBenchmark failed: {e}")

if __name__ == "__main__":
    benchmark()