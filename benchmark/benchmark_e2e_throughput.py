import asyncio
import aiohttp
import time
import statistics

async def generate_text(session, prompt, request_id):
    # Adjust URL to match your server's actual endpoint
    url = "http://127.0.0.1:8080/generate"
    payload = {
        "prompt": prompt,
        "max_tokens": 50,
        "stream": True
    }
    
    start_time = time.time()
    first_token_time = None
    token_count = 0
    
    try:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                print(f"Request {request_id} failed with status {response.status}")
                return None
                
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    
                    if first_token_time is None:
                        first_token_time = time.time()
                    
                    token_count += 1
                    
    except Exception as e:
        print(f"Request {request_id} failed: {e}")
        return None
        
    end_time = time.time()
    total_time = end_time - start_time
    ttft = (first_token_time - start_time) if first_token_time else 0.0
    itl = (total_time - ttft) / token_count if token_count > 0 else 0.0
    
    return {
        "request_id": request_id,
        "ttft": ttft,
        "itl": itl,
        "total_time": total_time,
        "token_count": token_count,
        "tokens_per_sec": token_count / total_time if total_time > 0 else 0.0
    }

async def main():
    num_requests = 8
    # ~500 token prompt simulation (approx 10 words * 50 repetitions)
    base_prompt = "The rapid advancement of artificial intelligence and edge computing requires highly optimized, low-latency inference engines. "
    prompt = base_prompt * 50 
    
    print(f"Starting End-to-End Load Benchmark with {num_requests} concurrent requests...")
    print(f"Prompt length: ~{len(prompt.split())} words (simulating ~500 tokens)")
    print("-" * 60)
    
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        tasks = [generate_text(session, prompt, i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)
        
    end_time = time.time()
    total_benchmark_time = end_time - start_time
    
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        print("No successful requests. Is the server running on http://127.0.0.1:8080?")
        return
        
    avg_ttft = statistics.mean(r["ttft"] for r in valid_results)
    avg_itl = statistics.mean(r["itl"] for r in valid_results)
    total_tokens = sum(r["token_count"] for r in valid_results)
    aggregate_tps = total_tokens / total_benchmark_time
    
    print("Benchmark Results:")
    print(f"  Total Successful Requests: {len(valid_results)}")
    print(f"  Total Tokens Generated:    {total_tokens}")
    print(f"  Avg Time to First Token (TTFT):            {avg_ttft * 1000:.2f} ms")
    print(f"  Avg Inter-Token Latency (ITL):             {avg_itl * 1000:.2f} ms")
    print(f"  Aggregate Throughput:                      {aggregate_tps:.2f} Tokens/sec")
    print(f"  Total Benchmark Time:                      {total_benchmark_time:.2f} sec")
    print("-" * 60)
    
    if aggregate_tps > 100.0 and avg_ttft < 0.1:
        print("✅ SUCCESS: System maintains >100 Tokens/sec aggregate throughput with sub-100ms TTFT!")
    else:
        print("⚠️ WARNING: Performance targets not met. Check server load, kernel optimizations, and network overhead.")

if __name__ == "__main__":
    # Requires: pip install aiohttp
    asyncio.run(main())