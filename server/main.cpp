#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <random>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")
typedef int socklen_t;
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#define closesocket close
typedef int SOCKET;
#define INVALID_SOCKET -1
#endif

#include "sme_gemm.h"

// Link Rust functions
extern "C" {
void rmsnorm_f32(float* out, const float* x, const float* weight, int size, float eps);
void softmax_f32(float* x, int size);
}

// Config structure
struct ModelConfig {
    std::string model_type;
    int num_layers = 0;
    int num_heads = 0;
    int hidden_dim = 0;
    int vocab_size = 0;
    int max_seq_len = 0;
};

// Tensor structure
struct Tensor {
    std::string name;
    std::vector<int> dims;
    bool is_quantized = false;
    std::vector<float> scales; // row scales for quantized tensors
    std::vector<int8_t> data_i8;
    std::vector<float> data_f32;
};

// Global Model State
ModelConfig g_config;
std::vector<std::string> g_vocab;
std::unordered_map<std::string, Tensor> g_tensors;
std::unordered_map<std::string, int> g_token_to_id;

// Helper to split string
std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        result.push_back(item);
    }
    return result;
}

// Parse a simple yaml config (only key-value pairs)
std::unordered_map<std::string, std::string> parse_yaml(const std::string& filepath) {
    std::unordered_map<std::string, std::string> config;
    std::ifstream file(filepath);
    if (!file.is_open()) return config;
    
    std::string line;
    while (std::getline(file, line)) {
        // Skip comments and empty lines
        size_t comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }
        if (line.empty()) continue;
        
        size_t colon_pos = line.find(':');
        if (colon_pos != std::string::npos) {
            std::string key = line.substr(0, colon_pos);
            std::string value = line.substr(colon_pos + 1);
            
            // Trim whitespace
            key.erase(0, key.find_first_not_of(" \t\r\n"));
            key.erase(key.find_last_not_of(" \t\r\n") + 1);
            value.erase(0, value.find_first_not_of(" \t\r\n"));
            value.erase(value.find_last_not_of(" \t\r\n") + 1);
            
            if (!key.empty()) {
                config[key] = value;
            }
        }
    }
    return config;
}

// Load .sme binary model
bool load_sme_model(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open model file: " << filepath << std::endl;
        return false;
    }
    
    // 1. Magic check
    char magic[9];
    file.read(magic, 9);
    if (std::strncmp(magic, "SME_MODEL", 9) != 0) {
        std::cerr << "Invalid model format. Magic header mismatch." << std::endl;
        return false;
    }
    
    // 2. Config JSON
    uint32_t config_size = 0;
    file.read(reinterpret_cast<char*>(&config_size), 4);
    std::vector<char> config_buf(config_size);
    file.read(config_buf.data(), config_size);
    std::string config_str(config_buf.data(), config_size);
    
    // Parse config (simple key extractor to avoid dependencies)
    auto extract_int = [&](const std::string& key) {
        size_t pos = config_str.find("\"" + key + "\"");
        if (pos == std::string::npos) return 0;
        pos = config_str.find(":", pos);
        if (pos == std::string::npos) return 0;
        return std::stoi(config_str.substr(pos + 1));
    };
    
    size_t type_pos = config_str.find("\"model_type\"");
    if (type_pos != std::string::npos) {
        size_t val_start = config_str.find("\"", config_str.find(":", type_pos));
        size_t val_end = config_str.find("\"", val_start + 1);
        g_config.model_type = config_str.substr(val_start + 1, val_end - val_start - 1);
    }
    
    g_config.num_layers = extract_int("num_layers");
    g_config.num_heads = extract_int("num_heads");
    g_config.hidden_dim = extract_int("hidden_dim");
    g_config.vocab_size = extract_int("vocab_size");
    g_config.max_seq_len = extract_int("max_seq_len");
    
    std::cout << "Loaded config. Type: " << g_config.model_type 
              << ", Layers: " << g_config.num_layers
              << ", Hidden: " << g_config.hidden_dim
              << ", Vocab: " << g_config.vocab_size << std::endl;
              
    // 3. Vocabulary
    uint32_t vocab_size = 0;
    file.read(reinterpret_cast<char*>(&vocab_size), 4);
    g_vocab.resize(vocab_size);
    for (uint32_t i = 0; i < vocab_size; ++i) {
        uint32_t token_len = 0;
        file.read(reinterpret_cast<char*>(&token_len), 4);
        std::vector<char> token_buf(token_len);
        file.read(token_buf.data(), token_len);
        std::string token(token_buf.data(), token_len);
        g_vocab[i] = token;
        g_token_to_id[token] = i;
    }
    
    // 4. Tensors
    uint32_t tensor_count = 0;
    file.read(reinterpret_cast<char*>(&tensor_count), 4);
    for (uint32_t t = 0; t < tensor_count; ++t) {
        uint32_t name_len = 0;
        file.read(reinterpret_cast<char*>(&name_len), 4);
        std::vector<char> name_buf(name_len);
        file.read(name_buf.data(), name_len);
        std::string name(name_buf.data(), name_len);
        
        Tensor tensor;
        tensor.name = name;
        
        uint32_t dim_count = 0;
        file.read(reinterpret_cast<char*>(&dim_count), 4);
        tensor.dims.resize(dim_count);
        uint32_t total_elements = 1;
        for (uint32_t d = 0; d < dim_count; ++d) {
            file.read(reinterpret_cast<char*>(&tensor.dims[d]), 4);
            total_elements *= tensor.dims[d];
        }
        
        uint8_t is_quantized = 0;
        file.read(reinterpret_cast<char*>(&is_quantized), 1);
        tensor.is_quantized = (is_quantized == 1);
        
        uint32_t data_size = 0;
        file.read(reinterpret_cast<char*>(&data_size), 4);
        
        if (tensor.is_quantized) {
            // Read scales
            int rows = tensor.dims[0];
            tensor.scales.resize(rows);
            file.read(reinterpret_cast<char*>(tensor.scales.data()), rows * sizeof(float));
            
            // Read int8 weights
            tensor.data_i8.resize(total_elements);
            file.read(reinterpret_cast<char*>(tensor.data_i8.data()), total_elements * sizeof(int8_t));
        } else {
            // Read float32 weights
            tensor.data_f32.resize(total_elements);
            file.read(reinterpret_cast<char*>(tensor.data_f32.data()), total_elements * sizeof(float));
        }
        
        g_tensors[name] = tensor;
    }
    
    std::cout << "Successfully loaded " << g_tensors.size() << " tensors." << std::endl;
    return true;
}

// Tokenizer helpers
int tokenize(const std::string& word) {
    auto it = g_token_to_id.find(word);
    if (it != g_token_to_id.end()) {
        return it->second;
    }
    return 0; // <unk>
}

// Simple SwiGLU / SiLU activation
void silu(float* out, const float* in, int size) {
    for (int i = 0; i < size; ++i) {
        out[i] = in[i] * (1.0f / (1.0f + std::exp(-in[i])));
    }
}

// Run single inference step
int run_inference_step(int token_id, std::vector<float>& state) {
    int h_dim = g_config.hidden_dim;
    
    // Get token embedding
    Tensor& embed = g_tensors["embed_tokens"];
    std::memcpy(state.data(), &embed.data_f32[token_id * h_dim], h_dim * sizeof(float));
    
    std::vector<float> norm_state(h_dim);
    std::vector<float> q(h_dim), k(h_dim), v(h_dim), o(h_dim);
    std::vector<float> ffn_gate(h_dim * 2), ffn_up(h_dim * 2), ffn_down(h_dim);
    
    for (int l = 0; l < g_config.num_layers; ++l) {
        // 1. Attention RMSNorm (invoking Rust kernel)
        rmsnorm_f32(norm_state.data(), state.data(), g_tensors["layers." + std::to_string(l) + ".attn_norm"].data_f32.data(), h_dim, 1e-5f);
        
        // 2. Q, K, V projections (using C++ SME/SVE2 GEMV)
        Tensor& Wq = g_tensors["layers." + std::to_string(l) + ".q_proj"];
        Tensor& Wk = g_tensors["layers." + std::to_string(l) + ".k_proj"];
        Tensor& Wv = g_tensors["layers." + std::to_string(l) + ".v_proj"];
        
        gemv_s8_f32(Wq.data_i8.data(), norm_state.data(), q.data(), Wq.scales.data(), h_dim, h_dim);
        gemv_s8_f32(Wk.data_i8.data(), norm_state.data(), k.data(), Wk.scales.data(), h_dim, h_dim);
        gemv_s8_f32(Wv.data_i8.data(), norm_state.data(), v.data(), Wv.scales.data(), h_dim, h_dim);
        
        // 3. Simple self-attention mockup
        // Compute scale factor
        float att_scale = 1.0f / std::sqrt(static_cast<float>(h_dim));
        for (int i = 0; i < h_dim; ++i) {
            o[i] = v[i] * (q[i] * k[i] * att_scale); // mockup elementwise dot product attention
        }
        
        // Out projection
        Tensor& Wo = g_tensors["layers." + std::to_string(l) + ".o_proj"];
        std::vector<float> o_proj(h_dim);
        gemv_s8_f32(Wo.data_i8.data(), o.data(), o_proj.data(), Wo.scales.data(), h_dim, h_dim);
        
        // Residual connection
        for (int i = 0; i < h_dim; ++i) {
            state[i] += o_proj[i];
        }
        
        // 4. FFN RMSNorm (invoking Rust kernel)
        rmsnorm_f32(norm_state.data(), state.data(), g_tensors["layers." + std::to_string(l) + ".ffn_norm"].data_f32.data(), h_dim, 1e-5f);
        
        // MLP projections
        Tensor& Wgate = g_tensors["layers." + std::to_string(l) + ".gate_proj"];
        Tensor& Wup = g_tensors["layers." + std::to_string(l) + ".up_proj"];
        Tensor& Wdown = g_tensors["layers." + std::to_string(l) + ".down_proj"];
        
        gemv_s8_f32(Wgate.data_i8.data(), norm_state.data(), ffn_gate.data(), Wgate.scales.data(), h_dim * 2, h_dim);
        gemv_s8_f32(Wup.data_i8.data(), norm_state.data(), ffn_up.data(), Wup.scales.data(), h_dim * 2, h_dim);
        
        // SwiGLU activation
        std::vector<float> silu_gate(h_dim * 2);
        silu(silu_gate.data(), ffn_gate.data(), h_dim * 2);
        for (int i = 0; i < h_dim * 2; ++i) {
            silu_gate[i] *= ffn_up[i];
        }
        
        // FFN Down projection
        gemv_s8_f32(Wdown.data_i8.data(), silu_gate.data(), ffn_down.data(), Wdown.scales.data(), h_dim, h_dim * 2);
        
        // Residual connection
        for (int i = 0; i < h_dim; ++i) {
            state[i] += ffn_down[i];
        }
    }
    
    // Final normalization
    std::vector<float> final_norm(h_dim);
    rmsnorm_f32(final_norm.data(), state.data(), g_tensors["norm"].data_f32.data(), h_dim, 1e-5f);
    
    // LM head projection to logits
    Tensor& lm_head = g_tensors["lm_head"];
    std::vector<float> logits(g_config.vocab_size);
    
    // lm_head is large, so it's quantized
    gemv_s8_f32(lm_head.data_i8.data(), final_norm.data(), logits.data(), lm_head.scales.data(), g_config.vocab_size, h_dim);
    
    // Argmax sampling for stability/testability
    int best_token = 0;
    float max_logit = logits[0];
    for (int i = 1; i < g_config.vocab_size; ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
            best_token = i;
        }
    }
    
    return best_token;
}

// Generate tokens loop
std::string generate_text(const std::string& prompt, int max_tokens) {
    // Basic tokenizer (split words)
    std::vector<std::string> words = split(prompt, ' ');
    if (words.empty()) return "";
    
    std::vector<float> state(g_config.hidden_dim, 0.0f);
    int next_token = 1; // Start with token
    
    // Encode prompt
    for (const auto& w : words) {
        int tid = tokenize(w);
        next_token = run_inference_step(tid, state);
    }
    
    std::string response = "";
    for (int step = 0; step < max_tokens; ++step) {
        if (next_token < 0 || next_token >= g_config.vocab_size) break;
        
        std::string token_str = g_vocab[next_token];
        if (token_str == "</s>") break;
        
        response += token_str + " ";
        next_token = run_inference_step(next_token, state);
    }
    
    return response;
}

int main(int argc, char* argv[]) {
    std::string config_path = "config.yaml";
    if (argc > 1) {
        config_path = argv[1];
    }
    
    std::cout << "Reading config from " << config_path << "..." << std::endl;
    auto yaml_config = parse_yaml(config_path);
    
    std::string model_path = yaml_config.count("model_path") ? yaml_config["model_path"] : "model/llama4-7b.sme";
    std::string host = yaml_config.count("host") ? yaml_config["host"] : "0.0.0.0";
    int port = yaml_config.count("port") ? std::stoi(yaml_config["port"]) : 8080;
    
    std::cout << "Loading model from: " << model_path << std::endl;
    if (!load_sme_model(model_path)) {
        std::cerr << "Fatal error: Failed to load model." << std::endl;
        return 1;
    }
    
    // Socket initialization
#ifdef _WIN32
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "WSAStartup failed." << std::endl;
        return 1;
    }
#endif

    SOCKET server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == INVALID_SOCKET) {
        std::cerr << "Socket creation failed." << std::endl;
        return 1;
    }
    
    int opt = 1;
#ifndef _WIN32
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

    struct sockaddr_in address;
    std::memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY; // Listen on all interfaces
    address.sin_port = htons(port);
    
    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        std::cerr << "Bind failed on port " << port << std::endl;
        closesocket(server_fd);
        return 1;
    }
    
    if (listen(server_fd, 5) < 0) {
        std::cerr << "Listen failed." << std::endl;
        closesocket(server_fd);
        return 1;
    }
    
    std::cout << "HTTP Server running at http://" << host << ":" << port << "/" << std::endl;
    
    while (true) {
        socklen_t addrlen = sizeof(address);
        SOCKET client_fd = accept(server_fd, (struct sockaddr*)&address, &addrlen);
        if (client_fd == INVALID_SOCKET) {
            continue;
        }
        
        std::vector<char> buffer(4096, 0);
        int bytes_received = recv(client_fd, buffer.data(), buffer.size() - 1, 0);
        if (bytes_received <= 0) {
            closesocket(client_fd);
            continue;
        }
        
        std::string request(buffer.data());
        
        // Extract HTTP method and path
        std::size_t first_space = request.find(' ');
        std::size_t second_space = request.find(' ', first_space + 1);
        if (first_space == std::string::npos || second_space == std::string::npos) {
            closesocket(client_fd);
            continue;
        }
        
        std::string method = request.substr(0, first_space);
        std::string path = request.substr(first_space + 1, second_space - first_space - 1);
        
        if (method == "POST" && path == "/generate") {
            // Find content body (after double CRLF)
            std::size_t body_pos = request.find("\r\n\r\n");
            std::string body = "";
            if (body_pos != std::string::npos) {
                body = request.substr(body_pos + 4);
            }
            
            // Basic JSON extraction of prompt and max_tokens
            std::string prompt = "local server";
            int max_tokens = 30;
            
            std::size_t prompt_pos = body.find("\"prompt\"");
            if (prompt_pos != std::string::npos) {
                std::size_t start = body.find("\"", body.find(":", prompt_pos));
                std::size_t end = body.find("\"", start + 1);
                prompt = body.substr(start + 1, end - start - 1);
            }
            
            std::size_t tokens_pos = body.find("\"max_tokens\"");
            if (tokens_pos != std::string::npos) {
                std::size_t start = body.find(":", tokens_pos);
                std::size_t end = body.find_first_of(",}", start);
                max_tokens = std::stoi(body.substr(start + 1, end - start - 1));
            }
            
            std::cout << "Received prompt: \"" << prompt << "\", max_tokens: " << max_tokens << std::endl;
            std::string response_text = generate_text(prompt, max_tokens);
            
            // Format HTTP Response
            std::string json_response = "{\"generated_text\":\"" + response_text + "\"}";
            std::ostringstream response_stream;
            response_stream << "HTTP/1.1 200 OK\r\n"
                            << "Content-Type: application/json\r\n"
                            << "Content-Length: " << json_response.length() << "\r\n"
                            << "Connection: close\r\n\r\n"
                            << json_response;
            
            std::string response_str = response_stream.str();
            send(client_fd, response_str.c_str(), response_str.length(), 0);
        } else {
            // Simple 404 response
            std::string not_found = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            send(client_fd, not_found.c_str(), not_found.length(), 0);
        }
        
        closesocket(client_fd);
    }
    
    closesocket(server_fd);
#ifdef _WIN32
    WSACleanup();
#endif
    return 0;
}
