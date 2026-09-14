#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <thread>
#include <atomic>
#include <chrono>

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
#include "model_loader.h"
#include "http_streaming.h"
#include "paged_kv_cache.h"
#include "continuous_scheduler.h"
#include "speculative_decoder.h"
#include "radix_kv_cache.h"
#include "chunked_prefill.h"
#include "paged_attention.h"       // Day 10: Added

extern "C" {
void rmsnorm_f32(float* out, const float* x, const float* weight, int size, float eps);
void softmax_f32(float* x, int size);
}

struct ModelConfig {
    std::string model_type;
    int num_layers = 0;
    int num_heads = 0;
    int hidden_dim = 0;
    int vocab_size = 0;
    int max_seq_len = 0;
};

struct Tensor {
    std::string name;
    std::vector<int> dims;
    bool is_quantized = false;
    const float* scales = nullptr; 
    const int8_t* data_i8 = nullptr;
    const float* data_f32 = nullptr;
};

struct InferenceResult {
    std::vector<int> next_tokens;
    std::vector<float> batch_logits;
};

ModelConfig g_config;
std::vector<std::string> g_vocab;
std::unordered_map<std::string, int> g_token_to_id;
std::unordered_map<std::string, Tensor> g_tensors;

PagedKVCache* g_paged_cache = nullptr;
RadixKVCache* g_radix_cache = nullptr;
int g_block_size = 16;
SpeculativeDecoder* g_speculative_decoder = nullptr;
std::atomic<int> g_request_counter{0};

std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        result.push_back(item);
    }
    return result;
}

std::unordered_map<std::string, std::string> parse_yaml(const std::string& filepath) {
    std::unordered_map<std::string, std::string> config;
    std::ifstream file(filepath);
    if (!file.is_open()) return config;
    
    std::string line;
    while (std::getline(file, line)) {
        size_t comment_pos = line.find('#');
        if (comment_pos != std::string::npos) line = line.substr(0, comment_pos);
        if (line.empty()) continue;
        
        size_t colon_pos = line.find(':');
        if (colon_pos != std::string::npos) {
            std::string key = line.substr(0, colon_pos);
            std::string value = line.substr(colon_pos + 1);
            key.erase(0, key.find_first_not_of(" \t\r\n"));
            key.erase(key.find_last_not_of(" \t\r\n") + 1);
            value.erase(0, value.find_first_not_of(" \t\r\n"));
            value.erase(value.find_last_not_of(" \t\r\n") + 1);
            if (!key.empty()) config[key] = value;
        }
    }
    return config;
}

bool load_sme_model(const MappedFile& mapping) {
    const char* ptr = static_cast<const char*>(mapping.data);
    const char* end_ptr = ptr + mapping.size;

    if (mapping.size < 9 || std::strncmp(ptr, "SME_MODEL", 9) != 0) {
        std::cerr << "Invalid model format. Magic header mismatch." << std::endl;
        return false;
    }
    ptr += 9;

    if (ptr + 4 > end_ptr) return false;
    uint32_t config_size = *reinterpret_cast<const uint32_t*>(ptr);
    ptr += 4;

    if (ptr + config_size > end_ptr) return false;
    std::string config_str(ptr, config_size);
    ptr += config_size;
    
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
              
    if (ptr + 4 > end_ptr) return false;
    uint32_t vocab_size = *reinterpret_cast<const uint32_t*>(ptr);
    ptr += 4;
    
    g_vocab.resize(vocab_size);
    for (uint32_t i = 0; i < vocab_size; ++i) {
        if (ptr + 4 > end_ptr) return false;
        uint32_t token_len = *reinterpret_cast<const uint32_t*>(ptr);
        ptr += 4;
        if (ptr + token_len > end_ptr) return false;
        std::string token(ptr, token_len);
        ptr += token_len;
        g_vocab[i] = token;
        g_token_to_id[token] = i;
    }
    
    if (ptr + 4 > end_ptr) return false;
    uint32_t tensor_count = *reinterpret_cast<const uint32_t*>(ptr);
    ptr += 4;
    
    for (uint32_t t = 0; t < tensor_count; ++t) {
        if (ptr + 4 > end_ptr) return false;
        uint32_t name_len = *reinterpret_cast<const uint32_t*>(ptr);
        ptr += 4;
        if (ptr + name_len > end_ptr) return false;
        std::string name(ptr, name_len);
        ptr += name_len;
        
        Tensor tensor;
        tensor.name = name;
        
        if (ptr + 4 > end_ptr) return false;
        uint32_t dim_count = *reinterpret_cast<const uint32_t*>(ptr);
        ptr += 4;
        
        tensor.dims.resize(dim_count);
        uint32_t total_elements = 1;
        for (uint32_t d = 0; d < dim_count; ++d) {
            if (ptr + 4 > end_ptr) return false;
            tensor.dims[d] = *reinterpret_cast<const uint32_t*>(ptr);
            ptr += 4;
            total_elements *= tensor.dims[d];
        }
        
        if (ptr + 1 > end_ptr) return false;
        uint8_t is_quantized = *reinterpret_cast<const uint8_t*>(ptr);
        ptr += 1;
        tensor.is_quantized = (is_quantized == 1);
        
        if (ptr + 4 > end_ptr) return false;
        uint32_t data_size = *reinterpret_cast<const uint32_t*>(ptr);
        ptr += 4;
        
        if (tensor.is_quantized) {
            int rows = tensor.dims[0];
            if (ptr + rows * sizeof(float) > end_ptr) return false;
            tensor.scales = reinterpret_cast<const float*>(ptr);
            ptr += rows * sizeof(float);
            
            if (ptr + total_elements * sizeof(int8_t) > end_ptr) return false;
            tensor.data_i8 = reinterpret_cast<const int8_t*>(ptr);
            ptr += total_elements * sizeof(int8_t);
        } else {
            if (ptr + total_elements * sizeof(float) > end_ptr) return false;
            tensor.data_f32 = reinterpret_cast<const float*>(ptr);
            ptr += total_elements * sizeof(float);
        }
        g_tensors[name] = tensor;
    }
    
    std::cout << "Successfully loaded " << g_tensors.size() << " tensors via zero-copy mmap." << std::endl;
    return true;
}

int tokenize_word(const std::string& word) {
    auto it = g_token_to_id.find(word);
    return (it != g_token_to_id.end()) ? it->second : 1;
}

std::vector<int> tokenize_prompt(const std::string& prompt) {
    std::vector<int> tokens;
    std::vector<std::string> words = split(prompt, ' ');
    for (const auto& word : words) {
        if (!word.empty()) {
            tokens.push_back(tokenize_word(word));
        }
    }
    return tokens;
}

void silu(float* out, const float* in, int size) {
    for (int i = 0; i < size; ++i) {
        out[i] = in[i] * (1.0f / (1.0f + std::exp(-in[i])));
    }
}

InferenceResult run_batched_inference_step(const std::vector<int>& token_ids, std::vector<std::vector<float>>& states, const std::vector<ContinuousRequest*>& batch) {
    int M = batch.size();
    int h_dim = g_config.hidden_dim;
    InferenceResult result;
    result.next_tokens.resize(M, 0);
    result.batch_logits.resize(M * g_config.vocab_size, 0.0f);
    
    std::vector<float> batch_q(M * h_dim, 0.0f);
    std::vector<float> batch_k(M * h_dim, 0.0f);
    std::vector<float> batch_v(M * h_dim, 0.0f);
    std::vector<float> batch_o(M * h_dim, 0.0f);
    std::vector<std::vector<float>> norm_states(M, std::vector<float>(h_dim));
    
    Tensor& embed = g_tensors["embed_tokens"];
    for (int m = 0; m < M; ++m) {
        std::memcpy(states[m].data(), &embed.data_f32[token_ids[m] * h_dim], h_dim * sizeof(float));
    }
    
    for (int l = 0; l < g_config.num_layers; ++l) {
        for (int m = 0; m < M; ++m) {
            rmsnorm_f32(norm_states[m].data(), states[m].data(), g_tensors["layers." + std::to_string(l) + ".attn_norm"].data_f32, h_dim, 1e-5f);
        }
        
        std::vector<float> flat_norm(M * h_dim);
        for (int m = 0; m < M; ++m) {
            std::memcpy(flat_norm.data() + m * h_dim, norm_states[m].data(), h_dim * sizeof(float));
        }
        
        Tensor& Wq = g_tensors["layers." + std::to_string(l) + ".q_proj"];
        Tensor& Wk = g_tensors["layers." + std::to_string(l) + ".k_proj"];
        Tensor& Wv = g_tensors["layers." + std::to_string(l) + ".v_proj"];
        
        gemv_s8_f32_batched(Wq.data_i8, flat_norm.data(), batch_q.data(), Wq.scales, M, h_dim, h_dim);
        gemv_s8_f32_batched(Wk.data_i8, flat_norm.data(), batch_k.data(), Wk.scales, M, h_dim, h_dim);
        gemv_s8_f32_batched(Wv.data_i8, flat_norm.data(), batch_v.data(), Wv.scales, M, h_dim, h_dim);
        
        std::vector<int> request_ids;
        for (int m = 0; m < M; ++m) request_ids.push_back(batch[m]->request_id);
        
        std::vector<std::vector<int>> batch_block_tables;
        std::vector<int> batch_seq_lens;
        if (g_paged_cache) {
            for (int m = 0; m < M; ++m) {
                g_paged_cache->append_tokens(batch[m]->request_id, 
                                             std::vector<float>(batch_k.begin() + m * h_dim, batch_k.begin() + (m + 1) * h_dim),
                                             std::vector<float>(batch_v.begin() + m * h_dim, batch_v.begin() + (m + 1) * h_dim));
            }
            g_paged_cache->get_cache_for_batch(request_ids, batch_block_tables, batch_seq_lens);
        }
        
        // Day 10: Replaced naive nested token/block loop with optimized kernel call
        std::fill(batch_o.begin(), batch_o.end(), 0.0f);
        compute_paged_attention(
            batch_q,
            g_paged_cache->get_k_pool(),
            g_paged_cache->get_v_pool(),
            batch_block_tables,
            batch_seq_lens,
            batch_o,
            h_dim,
            g_paged_cache->get_block_size()
        );
        
        Tensor& Wo = g_tensors["layers." + std::to_string(l) + ".o_proj"];
        std::vector<float> batch_o_proj(M * h_dim, 0.0f);
        gemv_s8_f32_batched(Wo.data_i8, batch_o.data(), batch_o_proj.data(), Wo.scales, M, h_dim, h_dim);
        
        for (int m = 0; m < M; ++m) {
            for (int i = 0; i < h_dim; ++i) {
                states[m][i] += batch_o_proj[m * h_dim + i];
            }
        }
        
        for (int m = 0; m < M; ++m) {
            rmsnorm_f32(norm_states[m].data(), states[m].data(), g_tensors["layers." + std::to_string(l) + ".ffn_norm"].data_f32, h_dim, 1e-5f);
            Tensor& Wgate = g_tensors["layers." + std::to_string(l) + ".gate_proj"];
            Tensor& Wup = g_tensors["layers." + std::to_string(l) + ".up_proj"];
            Tensor& Wdown = g_tensors["layers." + std::to_string(l) + ".down_proj"];
            
            std::vector<float> ffn_gate(h_dim * 2, 0.0f);
            std::vector<float> ffn_up(h_dim * 2, 0.0f);
            std::vector<float> ffn_down(h_dim, 0.0f);
            
            gemv_s8_f32(Wgate.data_i8, norm_states[m].data(), ffn_gate.data(), Wgate.scales, h_dim * 2, h_dim);
            gemv_s8_f32(Wup.data_i8, norm_states[m].data(), ffn_up.data(), Wup.scales, h_dim * 2, h_dim);
            
            std::vector<float> silu_gate(h_dim * 2, 0.0f);
            silu(silu_gate.data(), ffn_gate.data(), h_dim * 2);
            for (int i = 0; i < h_dim * 2; ++i) silu_gate[i] *= ffn_up[i];
            
            gemv_s8_f32(Wdown.data_i8, silu_gate.data(), ffn_down.data(), Wdown.scales, h_dim, h_dim * 2);
            for (int i = 0; i < h_dim; ++i) states[m][i] += ffn_down[i];
        }
    }
    
    std::vector<float> batch_final_norm(M * h_dim);
    for (int m = 0; m < M; ++m) {
        rmsnorm_f32(batch_final_norm.data() + m * h_dim, states[m].data(), g_tensors["norm"].data_f32, h_dim, 1e-5f);
    }
    
    Tensor& lm_head = g_tensors["lm_head"];
    std::vector<float> batch_logits(M * g_config.vocab_size, 0.0f);
    gemv_s8_f32_batched(lm_head.data_i8, batch_final_norm.data(), batch_logits.data(), lm_head.scales, M, g_config.vocab_size, h_dim);
    
    for (int m = 0; m < M; ++m) {
        int best_token = 0;
        float max_logit = batch_logits[m * g_config.vocab_size];
        for (int i = 1; i < g_config.vocab_size; ++i) {
            if (batch_logits[m * g_config.vocab_size + i] > max_logit) {
                max_logit = batch_logits[m * g_config.vocab_size + i];
                best_token = i;
            }
        }
        result.next_tokens[m] = best_token;
    }
    
    result.batch_logits = std::move(batch_logits);
    return result;
}

void scheduler_loop(ContinuousScheduler* scheduler) {
    while (scheduler->is_running()) {
        auto batch = scheduler->step();
        if (batch.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        
        int M = batch.size();
        std::vector<int> current_tokens(M);
        std::vector<std::vector<float>> states(M, std::vector<float>(g_config.hidden_dim, 0.0f));
        std::vector<std::vector<int>> recent_tokens_history(M);

        for (int m = 0; m < M; ++m) {
            auto* req = batch[m];
            
            if (!req->chunked_state.is_prefill_complete) {
                std::vector<int> chunk = req->chunked_state.get_next_chunk();
                if (!chunk.empty()) {
                    current_tokens[m] = chunk.back();
                    recent_tokens_history[m] = chunk;
                }
            } else {
                current_tokens[m] = req->generated_tokens.back();
                recent_tokens_history[m] = req->generated_tokens;
            }
        }
        
        std::vector<DraftResult> drafts(M);
        for (int m = 0; m < M; ++m) {
            drafts[m] = g_speculative_decoder->generate_draft(recent_tokens_history[m]);
            if (drafts[m].draft_length > 0) {
                batch[m]->speculative_total_drafts += drafts[m].draft_length;
            }
        }

        InferenceResult inf_result = run_batched_inference_step(current_tokens, states, batch);

        for (int m = 0; m < M; ++m) {
            auto* req = batch[m];
            int accepted_count = 0;
            int replacement_token = -1;

            if (drafts[m].draft_length > 0) {
                std::vector<float> req_logits(
                    inf_result.batch_logits.begin() + m * g_config.vocab_size,
                    inf_result.batch_logits.begin() + (m + 1) * g_config.vocab_size
                );

                g_speculative_decoder->verify_draft(
                    drafts[m].draft_tokens,
                    req_logits,
                    g_config.vocab_size,
                    accepted_count,
                    replacement_token
                );
                req->speculative_accepted_count += accepted_count;
            }

            std::vector<int> tokens_to_add;
            if (accepted_count > 0) {
                for (int i = 0; i < drafts[m].draft_length; ++i) {
                    tokens_to_add.push_back(drafts[m].draft_tokens[i]);
                }
            } else {
                int next_token = (replacement_token != -1) ? replacement_token : inf_result.next_tokens[m];
                tokens_to_add.push_back(next_token);
            }

            auto handle_completion = [&]() {
                if (req->speculative_total_drafts > 0) {
                    float acceptance_rate = (float)req->speculative_accepted_count / req->speculative_total_drafts * 100.0f;
                    std::cout << "[Speculative] Req " << req->request_id 
                              << " | Accepted: " << req->speculative_accepted_count 
                              << "/" << req->speculative_total_drafts 
                              << " drafts (" << acceptance_rate << "%)\n";
                }

                req->state = RequestState::COMPLETED;
                if (req->stream) {
                    StreamingManager::instance().end_session(req->request_id);
                } else {
                    std::string json_response = "{\"generated_text\":\"" + req->generated_text + "\"}";
                    std::ostringstream response_stream;
                    response_stream << "HTTP/1.1 200 OK\r\n"
                                    << "Content-Type: application/json\r\n"
                                    << "Content-Length: " << json_response.length() << "\r\n"
                                    << "Connection: close\r\n\r\n"
                                    << json_response;
                    std::string response_str = response_stream.str();
                    send(req->client_fd, response_str.c_str(), response_str.length(), 0);
                    closesocket(req->client_fd);
                }
                if (g_paged_cache) g_paged_cache->free_request(req->request_id);
            };

            for (int next_token : tokens_to_add) {
                if (req->state == RequestState::COMPLETED || req->state == RequestState::FAILED) {
                    break;
                }
                
                if (next_token < 0 || next_token >= g_config.vocab_size) {
                    handle_completion();
                    break;
                }
                
                req->generated_tokens.push_back(next_token);
                req->current_step++;
                
                std::string token_str = g_vocab[next_token];
                if (token_str == "</s>") {
                    handle_completion();
                    break;
                }
                
                if (req->stream) {
                    StreamingManager::instance().push_token(req->request_id, token_str);
                } else {
                    req->generated_text += token_str + " ";
                }

                if (req->current_step >= req->max_tokens) {
                    handle_completion();
                    break;
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    std::string config_path = "config.yaml";
    if (argc > 1) config_path = argv[1];
    
    std::cout << "Reading config from " << config_path << "..." << std::endl;
    auto yaml_config = parse_yaml(config_path);
    
    std::string model_path = yaml_config.count("model_path") ? yaml_config["model_path"] : "model/llama4-7b.sme";
    std::string host = yaml_config.count("host") ? yaml_config["host"] : "0.0.0.0";
    int port = yaml_config.count("port") ? std::stoi(yaml_config["port"]) : 8080;
    
    std::cout << "Loading model from: " << model_path << std::endl;
    MappedFile model_mapping;
    extern bool mmap_load(const std::string&, MappedFile&);
    if (!mmap_load(model_path, model_mapping)) {
        std::cerr << "Fatal error: Failed to load model." << std::endl;
        return 1;
    }
    
    if (!load_sme_model(model_mapping)) {
        std::cerr << "Fatal error: Failed to parse model." << std::endl;
        extern void mmap_unload(MappedFile&);
        mmap_unload(model_mapping);
        return 1;
    }
    
    int cache_blocks = 256;
    g_block_size = 16;
    g_paged_cache = new PagedKVCache(cache_blocks, g_block_size, g_config.hidden_dim);
    g_radix_cache = new RadixKVCache();
    
    g_speculative_decoder = new SpeculativeDecoder(3);
    
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
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);
    
    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        std::cerr << "Bind failed on port " << port << std::endl;
        closesocket(server_fd);
        return 1;
    }
    
    if (listen(server_fd, 512) < 0) {
        std::cerr << "Listen failed." << std::endl;
        closesocket(server_fd);
        return 1;
    }
    
    std::cout << "HTTP Server running at http://" << host << ":" << port << "/" << std::endl;
    
    ContinuousScheduler global_scheduler(g_paged_cache, g_speculative_decoder, g_radix_cache, 8, 64);
    std::thread scheduler_thread(scheduler_loop, &global_scheduler);
    
    while (true) {
        socklen_t addrlen = sizeof(address);
        SOCKET client_fd = accept(server_fd, (struct sockaddr*)&address, &addrlen);
        if (client_fd == INVALID_SOCKET) continue;
        
        std::vector<char> buffer(4096, 0);
        int bytes_received = recv(client_fd, buffer.data(), buffer.size() - 1, 0);
        if (bytes_received <= 0) {
            closesocket(client_fd);
            continue;
        }
        
        std::string request(buffer.data());
        std::size_t first_space = request.find(' ');
        std::size_t second_space = request.find(' ', first_space + 1);
        if (first_space == std::string::npos || second_space == std::string::npos) {
            closesocket(client_fd);
            continue;
        }
        
        std::string method = request.substr(0, first_space);
        std::string path = request.substr(first_space + 1, second_space - first_space - 1);
        
        if (method == "POST" && path == "/generate") {
            std::size_t body_pos = request.find("\r\n\r\n");
            std::string body = (body_pos != std::string::npos) ? request.substr(body_pos + 4) : "";
            
            std::string prompt = "local server";
            int max_tokens = 30;
            bool stream = (body.find("\"stream\": true") != std::string::npos);
            
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
            
            std::cout << "Received prompt: \"" << prompt << "\", max_tokens: " << max_tokens << ", stream: " << (stream ? "true" : "false") << std::endl;
            
            int req_id = g_request_counter.fetch_add(1);
            
            std::vector<int> tokenized_prompt = tokenize_prompt(prompt);
            
            if (stream) {
                StreamingManager::instance().start_session(req_id, client_fd, true);
            }
            
            global_scheduler.enqueue_request(req_id, prompt, tokenized_prompt, max_tokens, stream, client_fd);
            
        } else {
            std::string not_found = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            send(client_fd, not_found.c_str(), not_found.length(), 0);
            closesocket(client_fd);
        }
    }
    
    global_scheduler.stop();
    if (scheduler_thread.joinable()) {
        scheduler_thread.join();
    }
    
    closesocket(server_fd);
#ifdef _WIN32
    WSACleanup();
#endif
    extern void mmap_unload(MappedFile&);
    mmap_unload(model_mapping);
    
    delete g_speculative_decoder;
    delete g_radix_cache;
    delete g_paged_cache;
    return 0;
}