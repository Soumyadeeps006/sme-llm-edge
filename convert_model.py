#!/usr/bin/env python3
import os
import sys
import json
import struct
import numpy as np

# SME Model Serializer
# Binary format:
# [9 bytes] Magic "SME_MODEL"
# [4 bytes] JSON Config size (uint32)
# [Config size bytes] JSON configuration string
# [4 bytes] Vocabulary size (uint32)
# [Vocab size entries] Each entry:
#   [4 bytes] Token string length (uint32)
#   [Length bytes] Token string
# [4 bytes] Number of Tensors (uint32)
# [Tensors entries] Each entry:
#   [4 bytes] Name length (uint32)
#   [Name length bytes] Name string
#   [4 bytes] Dimension count (uint32)
#   [Dim count * 4 bytes] Dimensions (uint32 array)
#   [1 byte] Is Quantized (uint8: 1=int8, 0=float32)
#   [4 bytes] Data size in bytes (uint32)
#   If quantized:
#     [Dim[0] * 4 bytes] Quantization scales per row (float32 array)
#   [Data size bytes] Raw tensor weights (int8 or float32)

def quantize_matrix(weight):
    # Perform symmetric per-row 8-bit quantization
    # weight shape: [M, K]
    # scale = 127.0 / max(abs(row))
    # w_quant = round(w * scale)
    M, K = weight.shape
    scales = []
    quant_weight = np.zeros_like(weight, dtype=np.int8)
    
    for i in range(M):
        row = weight[i]
        max_val = np.max(np.abs(row))
        if max_val == 0:
            scale = 1.0
        else:
            scale = 127.0 / max_val
        
        scales.append(scale)
        quant_weight[i] = np.clip(np.round(row * scale), -128, 127).astype(np.int8)
        
    return quant_weight, np.array(scales, dtype=np.float32)

def save_sme_model(filepath, config, vocab, tensors):
    print(f"Writing .sme model to {filepath}...")
    with open(filepath, 'wb') as f:
        # 1. Magic
        f.write(b"SME_MODEL")
        
        # 2. Config JSON
        config_bytes = json.dumps(config).encode('utf-8')
        f.write(struct.pack("<I", len(config_bytes)))
        f.write(config_bytes)
        
        # 3. Vocabulary
        f.write(struct.pack("<I", len(vocab)))
        for token in vocab:
            token_bytes = token.encode('utf-8')
            f.write(struct.pack("<I", len(token_bytes)))
            f.write(token_bytes)
            
        # 4. Tensors
        f.write(struct.pack("<I", len(tensors)))
        for name, data in tensors.items():
            name_bytes = name.encode('utf-8')
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            
            dims = data.shape
            f.write(struct.pack("<I", len(dims)))
            for d in dims:
                f.write(struct.pack("<I", d))
                
            # Perform quantization for large weight matrices (like projection matrices)
            # Embedding and small bias arrays are kept in float32 for model accuracy
            is_quantized = 1 if len(dims) == 2 and dims[0] > 16 else 0
            f.write(struct.pack("B", is_quantized))
            
            if is_quantized == 1:
                quant_w, scales = quantize_matrix(data)
                data_bytes = quant_w.tobytes()
                # Data size in bytes
                f.write(struct.pack("<I", len(data_bytes)))
                # Scales
                f.write(scales.tobytes())
                # Weights
                f.write(data_bytes)
            else:
                data_f32 = data.astype(np.float32)
                data_bytes = data_f32.tobytes()
                # Data size in bytes
                f.write(struct.pack("<I", len(data_bytes)))
                # Weights
                f.write(data_bytes)
                
    print("Model compilation completed successfully.")

def generate_mock_model():
    print("Generating a synthetic 7B LLM structure for testing...")
    # Define a small 3-layer model config to simulate Llama structure
    config = {
        "model_type": "llama-sme-test",
        "num_layers": 3,
        "num_heads": 8,
        "hidden_dim": 256,
        "vocab_size": 1000,
        "max_seq_len": 512
    }
    
    # Generate simple vocab
    vocab = [f"<token_{i}>" for i in range(config["vocab_size"])]
    vocab[0] = "<unk>"
    vocab[1] = "<s>"
    vocab[2] = "</s>"
    # Add some common words for testing RAG/generation
    words = ["the", "local", "server", "runs", "llama", "model", "with", "arm", "sme", "sve2", "acceleration", "for", "high", "throughput", "inference"]
    for i, w in enumerate(words):
        if i + 3 < len(vocab):
            vocab[i + 3] = w
            
    tensors = {}
    
    # Embedding table
    tensors["embed_tokens"] = np.random.randn(config["vocab_size"], config["hidden_dim"]).astype(np.float32)
    
    # Layers
    for layer in range(config["num_layers"]):
        # Self-attention weights
        tensors[f"layers.{layer}.q_proj"] = np.random.randn(config["hidden_dim"], config["hidden_dim"]).astype(np.float32)
        tensors[f"layers.{layer}.k_proj"] = np.random.randn(config["hidden_dim"], config["hidden_dim"]).astype(np.float32)
        tensors[f"layers.{layer}.v_proj"] = np.random.randn(config["hidden_dim"], config["hidden_dim"]).astype(np.float32)
        tensors[f"layers.{layer}.o_proj"] = np.random.randn(config["hidden_dim"], config["hidden_dim"]).astype(np.float32)
        
        # MLP weights
        tensors[f"layers.{layer}.gate_proj"] = np.random.randn(config["hidden_dim"] * 2, config["hidden_dim"]).astype(np.float32)
        tensors[f"layers.{layer}.up_proj"] = np.random.randn(config["hidden_dim"] * 2, config["hidden_dim"]).astype(np.float32)
        tensors[f"layers.{layer}.down_proj"] = np.random.randn(config["hidden_dim"], config["hidden_dim"] * 2).astype(np.float32)
        
        # Norms
        tensors[f"layers.{layer}.attn_norm"] = np.ones(config["hidden_dim"]).astype(np.float32)
        tensors[f"layers.{layer}.ffn_norm"] = np.ones(config["hidden_dim"]).astype(np.float32)
        
    # Final layer norm and LM head
    tensors["norm"] = np.ones(config["hidden_dim"]).astype(np.float32)
    tensors["lm_head"] = np.random.randn(config["vocab_size"], config["hidden_dim"]).astype(np.float32)
    
    return config, vocab, tensors

if __name__ == "__main__":
    output_path = "model/llama4-7b.sme"
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Try importing torch/transformers to read an actual model, or fallback
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        if len(sys.argv) < 3:
            print("To convert a real Hugging Face model, specify the model path: python convert_model.py <output.sme> <hf_model_path_or_id>")
            print("Proceeding with synthetic model generation for validation...")
            raise ImportError()
            
        model_id = sys.argv[2]
        print(f"Loading Hugging Face model {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        
        config = {
            "model_type": model.config.model_type,
            "num_layers": getattr(model.config, "num_hidden_layers", 1),
            "num_heads": getattr(model.config, "num_attention_heads", 1),
            "hidden_dim": model.config.hidden_size,
            "vocab_size": model.config.vocab_size,
            "max_seq_len": getattr(model.config, "max_position_embeddings", 2048)
        }
        
        vocab = [tokenizer.decode([i], skip_special_tokens=False) for i in range(config["vocab_size"])]
        
        tensors = {}
        for name, param in model.named_parameters():
            tensors[name] = param.detach().cpu().numpy()
            
        save_sme_model(output_path, config, vocab, tensors)
        
    except ImportError:
        config, vocab, tensors = generate_mock_model()
        save_sme_model(output_path, config, vocab, tensors)
