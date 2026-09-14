#!/usr/bin/env python3
import time
import urllib.request
import json

SERVER_URL = "http://localhost:8080/generate"

# A prompt that benefits from predictable, repetitive continuation
PROMPT = "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over"
MAX_TOKENS = 50

print(f"🚀 Launching speculative decoding benchmark (Target: {MAX_TOKENS} tokens)...")

# Optional: Warm-up request to ensure the server is loaded and ready
print("🔥 Sending warm-up request...")
try:
    warmup_data = json.dumps({"prompt": "Hello", "max_tokens": 5, "stream": False}).encode('utf-8')
    warmup_req = urllib.request.Request(SERVER_URL, data=warmup_data, headers={'Content-Type': 'application/json'}, method='POST')
    urllib.request.urlopen(warmup_req, timeout=5)
    print("✅ Warm-up complete.")
except Exception as e:
    print(f"⚠️ Warm-up failed (server might be starting): {e}")

data = json.dumps({"prompt": PROMPT, "max_tokens": MAX_TOKENS, "stream": False}).encode('utf-8')
req = urllib.request.Request(SERVER_URL, data=data, headers={'Content-Type': 'application/json'}, method='POST')

start_time = time.time()
try:
    with urllib.request.urlopen(req, timeout=30) as response:
        res = json.loads(response.read().decode('utf-8'))
        elapsed = time.time() - start_time
        generated_text = res.get('generated_text', '')
        
        # Rough token count estimation (1 token ≈ 1 word / 4 chars)
        estimated_tokens = len(generated_text.split()) 
        
        tps = estimated_tokens / elapsed if elapsed > 0 else 0
        print(f"\n✅ Benchmark Complete!")
        print(f"⏱️  Time: {elapsed:.2f}s")
        print(f"🔢 Estimated Tokens: {estimated_tokens}")
        print(f"⚡ Throughput: {tps:.2f} Tokens/sec")
        print(f"📝 Response: {generated_text[:80]}...")
except Exception as e:
    print(f"❌ Benchmark Failed: {e}")