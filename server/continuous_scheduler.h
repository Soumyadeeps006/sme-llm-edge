#pragma once
#include <vector>
#include <mutex>
#include <queue>
#include <string>
#include "paged_kv_cache.h"
#include "speculative_decoder.h"
#include "chunked_prefill.h"
#include "radix_kv_cache.h"
#include "moe_router.h"
#include "pipeline_manager.h"
#include "early_exit_tracker.h"
#include "tome_tracker.h"
#include "eviction_tracker.h"
#include "flash_decoding_tracker.h"
#include "mtp_heads.h" // DAY 36: Added MTP Heads include

#ifdef _WIN32
#include <winsock2.h>
typedef int socklen_t;
#else
#include <sys/socket.h>
#define closesocket close
typedef int SOCKET;
#define INVALID_SOCKET -1
#endif

enum class RequestState {
    PENDING,
    RUNNING,
    COMPLETED,
    FAILED
};

struct ContinuousRequest {
    int request_id;
    std::string prompt;
    std::vector<int> generated_tokens;
    std::string generated_text;
    int max_tokens;
    bool stream;
    SOCKET client_fd;
    
    RequestState state = RequestState::PENDING;
    int current_step = 0;
    
    ChunkedPrefillState chunked_state;
    
    int prefill_chunk_size = 512;
    int prefill_tokens_processed = 0;
    bool is_prefill_chunked = false;
    
    int speculative_accepted_count = 0;
    int speculative_total_drafts = 0;
    std::vector<int> pending_draft_tokens;
    
    uint32_t num_q_heads = 32;
    uint32_t num_kv_heads = 8;
    uint32_t qkv_head_ratio = 4;
};

class ContinuousScheduler {
private:
    std::vector<ContinuousRequest*> active_batch;
    std::queue<ContinuousRequest> pending_queue;
    std::mutex scheduler_mutex;
    bool running = true;
    
    PagedKVCache* paged_cache;
    SpeculativeDecoder* speculative_decoder;
    RadixKVCache* radix_cache;
    MoERouter* moe_router;
    PipelineManager* pipeline_mgr_;
    EarlyExitTracker* early_exit_tracker_;
    TomeTracker* tome_tracker_;
    EvictionTracker* eviction_tracker_;
    FlashDecodingTracker* flash_decoding_tracker_;
    MTPHeads* mtp_heads_;                          // DAY 36: MTP heads instance
    int max_batch_size;
    int prefill_chunk_size;
    bool use_sparse_moe;
    bool use_mtp_;                                 // DAY 36: MTP enabled flag

public:
    explicit ContinuousScheduler(PagedKVCache* cache, SpeculativeDecoder* spec_decoder, 
                                 RadixKVCache* r_cache, MoERouter* moe = nullptr,
                                 PipelineManager* pipeline = nullptr,
                                 EarlyExitTracker* early_exit = nullptr,
                                 TomeTracker* tome = nullptr,
                                 EvictionTracker* eviction = nullptr,
                                 FlashDecodingTracker* flash_decoding = nullptr,
                                 MTPHeads* mtp = nullptr,                        // DAY 36
                                 int max_batch = 8, int chunk_size = 64,
                                 bool sparse_moe = false, bool use_mtp = false); // DAY 36
    
    void enqueue_request(int request_id, const std::string& prompt, const std::vector<int>& tokenized_prompt, 
                         int max_tokens, bool stream, SOCKET client_fd);
    
    std::vector<ContinuousRequest*> step();
    
    // DAY 37: Unified decode dispatch logic
    void execute_decode_step(ContinuousRequest* req, const std::vector<float>& hidden_states);
    
    void verify_and_update_drafts(SpeculativeDecoder* decoder, int vocab_size, const std::vector<float>& target_logits);
    
    void execute_moe_ffn(const float* input_hidden_states, float* output_hidden_states, int num_tokens);
    
    void execute_overlap_moe_gqa(const float* input_hidden_states, float* output_hidden_states, 
                                 int num_tokens, const std::vector<const float*>& next_k_ptrs,
                                 const std::vector<const float*>& next_v_ptrs);
                                 
    void execute_overlap_moe_gqa_early_exit(const float* input_hidden_states, float* output_hidden_states, 
                                            int num_tokens, const std::vector<const float*>& next_k_ptrs,
                                            const std::vector<const float*>& next_v_ptrs,
                                            const uint64_t* active_token_mask);
                                            
    void execute_tome_merge(ContinuousRequest* req, const std::vector<float>& hidden_states);
    
    // DAY 36: MTP dispatch hook
    void generate_mtp_drafts(const std::vector<float>& hidden_states);
    
    bool is_sparse_moe_enabled() const { return use_sparse_moe; }
    bool is_mtp_enabled() const { return use_mtp_; }
    
    void mark_completed(int request_id);
    void stop() { std::lock_guard<std::mutex> lock(scheduler_mutex); running = false; }
    bool is_running() const { return running; }
    
    ContinuousRequest* get_request(int request_id);
};