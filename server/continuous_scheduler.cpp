#include "continuous_scheduler.h"
#include <iostream>
#include <algorithm>
#include <vector>
#include <queue>
#include <mutex>
#include <string>

// ============================================================================
// Day 38: FFI Boundary Audit
// Memory Layout & Ownership Rules:
// - Caller (C++) retains ownership of ALL pointers passed to these Rust functions.
// - Rust side MUST NOT panic across the FFI boundary. A custom #[panic_handler] 
//   is configured in Rust to loop {} to prevent undefined behavior (UB).
// - All pointers must be validated for null and correct slice bounds before calling.
// ============================================================================
extern "C" {
    // next_k_ptrs / next_v_ptrs: Array of pointers. Each must point to valid, 
    // aligned memory for a single KV block. Array size is num_kv_blocks.
    // input_ptr: Contiguous buffer of size (num_tokens * hidden_dim) * sizeof(float).
    // output_ptr: Contiguous buffer of size (num_tokens * hidden_dim) * sizeof(float).
    // expert_weights_ptr: Contiguous buffer of size (num_tokens * expert_weight_stride) * sizeof(uint16_t).
    // active_token_mask: Buffer of size ((num_tokens + 63) / 64) * sizeof(uint64_t), OR nullptr (implies all active).
    void rust_sme_gqa_moe_pipeline_overlap_early_exit(
        const float* next_k_ptrs,
        const float* next_v_ptrs,
        uint32_t num_kv_blocks,
        const float* input_ptr,
        const uint16_t* expert_weights_ptr,
        float* output_ptr,
        uint32_t num_tokens,
        uint32_t hidden_dim,
        uint32_t expert_intermediate_dim,
        uint32_t expert_weight_stride,
        const uint64_t* active_token_mask
    );
}

ContinuousScheduler::ContinuousScheduler(PagedKVCache* cache, SpeculativeDecoder* spec_decoder, 
                                         RadixKVCache* r_cache, MoERouter* moe, 
                                         PipelineManager* pipeline,
                                         EarlyExitTracker* early_exit,
                                         TomeTracker* tome,
                                         EvictionTracker* eviction,
                                         FlashDecodingTracker* flash_decoding,
                                         MTPHeads* mtp,                        
                                         int max_batch, int chunk_size, bool sparse_moe, bool use_mtp) 
    : paged_cache(cache), speculative_decoder(spec_decoder), radix_cache(r_cache), 
      moe_router(moe), pipeline_mgr_(pipeline), early_exit_tracker_(early_exit),
      tome_tracker_(tome), eviction_tracker_(eviction), 
      flash_decoding_tracker_(flash_decoding),
      mtp_heads_(mtp),                         
      max_batch_size(max_batch), prefill_chunk_size(chunk_size), 
      use_sparse_moe(sparse_moe), use_mtp_(use_mtp) {}

void ContinuousScheduler::enqueue_request(int request_id, const std::string& prompt, 
                                          const std::vector<int>& tokenized_prompt, 
                                          int max_tokens, bool stream, SOCKET client_fd) {
    std::lock_guard<std::mutex> lock(scheduler_mutex);
    ContinuousRequest req;
    req.request_id = request_id;
    req.prompt = prompt;
    req.max_tokens = max_tokens;
    req.stream = stream;
    req.client_fd = client_fd;
    req.state = RequestState::PENDING;
    
    req.chunked_state = ChunkedPrefillState(tokenized_prompt, prefill_chunk_size);
    req.prefill_chunk_size = prefill_chunk_size;
    req.prefill_tokens_processed = 0;
    req.is_prefill_chunked = false;
    
    req.num_q_heads = 32;
    req.num_kv_heads = 8;
    req.qkv_head_ratio = req.num_q_heads / req.num_kv_heads;
    
    if (radix_cache && !tokenized_prompt.empty()) {
        auto [matched_count, source_req_id] = radix_cache->match_prefix(tokenized_prompt);
        if (matched_count > 0 && source_req_id != -1) {
            std::cout << "[Scheduler] Prefix hit! Request " << request_id 
                      << " sharing " << matched_count << " tokens from Request " << source_req_id << "\n";
            paged_cache->share_prefix(request_id, source_req_id, matched_count);
            req.chunked_state.advance_offset(matched_count);
            req.prefill_tokens_processed = matched_count;
        }
    }
    
    pending_queue.push(std::move(req));
}

std::vector<ContinuousRequest*> ContinuousScheduler::step() {
    std::lock_guard<std::mutex> lock(scheduler_mutex);
    
    // 1. Clean up completed or failed requests from active batch
    active_batch.erase(
        std::remove_if(active_batch.begin(), active_batch.end(), [](ContinuousRequest* req) {
            return req == nullptr || req->state == RequestState::COMPLETED || req->state == RequestState::FAILED;
        }),
        active_batch.end()
    );
    
    // 2. Reset early exit tracker for the new batch
    if (early_exit_tracker_) {
        early_exit_tracker_->reset_batch(static_cast<int>(active_batch.size()));
    }
    
    // 3. Day 38: Hardened KV Cache Eviction Check
    if (eviction_tracker_ && paged_cache) {
        for (auto* req : active_batch) {
            if (req != nullptr && req->state == RequestState::RUNNING && req->chunked_state.is_prefill_complete) {
                int current_seq_len = static_cast<int>(req->chunked_state.prompt_tokens.size()) + 
                                      static_cast<int>(req->generated_tokens.size());
                // Safe guard: eviction_tracker internally handles zero-token and block boundary edge cases
                eviction_tracker_->check_and_evict(req->request_id, current_seq_len);
            }
        }
    }

    // 4. FlashDecoding trigger check
    if (flash_decoding_tracker_ && paged_cache) {
        for (auto* req : active_batch) {
            if (req != nullptr && req->state == RequestState::RUNNING && req->chunked_state.is_prefill_complete) {
                int current_seq_len = static_cast<int>(req->chunked_state.prompt_tokens.size()) + 
                                      static_cast<int>(req->generated_tokens.size());
                if (flash_decoding_tracker_->should_use_flash_decoding(current_seq_len)) {
                    std::cout << "[Scheduler] FlashDecoding triggered for request " 
                              << req->request_id << " (seq_len: " << current_seq_len << ")\n";
                }
            }
        }
    }
    
    // 5. Separate candidates into decode and prefill
    std::vector<ContinuousRequest*> decode_candidates;
    std::vector<ContinuousRequest*> prefill_candidates;
    std::queue<ContinuousRequest> next_pending;
    
    while (!pending_queue.empty()) {
        pending_queue.front().state = RequestState::RUNNING;
        
        if (pending_queue.front().chunked_state.is_prefill_complete) {
            decode_candidates.push_back(&pending_queue.front());
        } else {
            prefill_candidates.push_back(&pending_queue.front());
        }
        pending_queue.pop();
    }
    
    active_batch.clear();
    
    // Prioritize decode candidates to maintain low latency
    for (auto* req : decode_candidates) {
        if (req != nullptr && active_batch.size() < static_cast<size_t>(max_batch_size)) {
            active_batch.push_back(req);
        } else if (req != nullptr) {
            next_pending.push(*req);
        }
    }
    
    // Fill remaining batch capacity with prefill candidates
    for (auto* req : prefill_candidates) {
        if (req != nullptr && active_batch.size() < static_cast<size_t>(max_batch_size)) {
            int total_prompt_tokens = static_cast<int>(req->chunked_state.prompt_tokens.size());
            int remaining_tokens = total_prompt_tokens - req->prefill_tokens_processed;
            
            if (remaining_tokens > 0) {
                int tokens_to_process = std::min(req->prefill_chunk_size, remaining_tokens);
                req->is_prefill_chunked = (total_prompt_tokens > req->prefill_chunk_size);
                req->prefill_tokens_processed += tokens_to_process;
                active_batch.push_back(req);
                
                if (req->prefill_tokens_processed >= total_prompt_tokens) {
                    req->chunked_state.is_prefill_complete = true;
                }
            } else {
                next_pending.push(*req);
            }
        } else if (req != nullptr) {
            next_pending.push(*req);
        }
    }
    
    // Return unprocessed candidates to pending queue
    while (!next_pending.empty()) {
        next_pending.front().state = RequestState::PENDING;
        pending_queue.push(next_pending.front());
        next_pending.pop();
    }
    
    return active_batch;
}

// DAY 37: Unified Decode Dispatch Logic
void ContinuousScheduler::execute_decode_step(ContinuousRequest* req, const std::vector<float>& hidden_states) {
    if (!req || req->state != RequestState::RUNNING || !req->chunked_state.is_prefill_complete) {
        return;
    }

    int current_seq_len = static_cast<int>(req->chunked_state.prompt_tokens.size()) + 
                          static_cast<int>(req->generated_tokens.size());

    // 1. FlashDecoding for long context
    if (flash_decoding_tracker_ && flash_decoding_tracker_->should_use_flash_decoding(current_seq_len)) {
        std::cout << "[Scheduler] Routing to FlashDecoding for request " << req->request_id 
                  << " (seq_len: " << current_seq_len << ")\n";
        // TODO: Invoke flash_decoding_tracker_->execute_flash_decoding(...)
        return;
    }
    
    // 2. MTP Speculative Decode
    if (use_mtp_ && mtp_heads_) {
        std::cout << "[Scheduler] Routing to MTP Speculative Decode for request " << req->request_id << "\n";
        generate_mtp_drafts(hidden_states);
        // Drafts are now in req->pending_draft_tokens, ready for verify_and_update_drafts
        return;
    }

    // 3. Fallback: Standard 1-Token PagedAttention Decode
    std::cout << "[Scheduler] Routing to Standard Decode for request " << req->request_id << "\n";
    // TODO: Invoke standard paged attention decode kernel here
}

void ContinuousScheduler::execute_moe_ffn(const float* input_hidden_states, 
                                           float* output_hidden_states, 
                                           int num_tokens) {
    if (!use_sparse_moe || !moe_router) {
        std::cerr << "[Scheduler] MoE not enabled or router not initialized\n";
        return;
    }
    
    // Day 38: FFI Guard
    if (!input_hidden_states || !output_hidden_states || num_tokens <= 0) {
        std::cerr << "[Scheduler] Invalid pointers or num_tokens for MoE FFN\n";
        return;
    }

    moe_router->route_and_compute(input_hidden_states, output_hidden_states, 
                                   static_cast<uint32_t>(num_tokens));
}

void ContinuousScheduler::execute_overlap_moe_gqa(const float* input_hidden_states, 
                                                   float* output_hidden_states, 
                                                   int num_tokens,
                                                   const std::vector<const float*>& next_k_ptrs,
                                                   const std::vector<const float*>& next_v_ptrs) {
    if (!use_sparse_moe || !moe_router || !pipeline_mgr_) {
        execute_moe_ffn(input_hidden_states, output_hidden_states, num_tokens);
        return;
    }

    // Day 38: FFI Guard
    if (!input_hidden_states || !output_hidden_states || num_tokens <= 0) {
        std::cerr << "[Scheduler] Invalid pointers or num_tokens for Overlap MoE GQA\n";
        return;
    }

    pipeline_mgr_->set_state(PipelineState::OVERLAPPING_GQA);
    pipeline_mgr_->prepare_next_step_prefetch(next_k_ptrs, next_v_ptrs, {});
    pipeline_mgr_->trigger_gqa_kv_prefetch();
    
    moe_router->route_and_compute(input_hidden_states, output_hidden_states, 
                                   static_cast<uint32_t>(num_tokens));
                                   
    pipeline_mgr_->set_state(PipelineState::IDLE);
}

void ContinuousScheduler::execute_overlap_moe_gqa_early_exit(
    const float* input_hidden_states, 
    float* output_hidden_states, 
    int num_tokens,
    const std::vector<const float*>& next_k_ptrs,
    const std::vector<const float*>& next_v_ptrs,
    const uint64_t* active_token_mask) 
{
    if (!use_sparse_moe || !moe_router || !pipeline_mgr_) {
        execute_moe_ffn(input_hidden_states, output_hidden_states, num_tokens);
        return;
    }

    // Day 38: FFI Guard
    if (!input_hidden_states || !output_hidden_states || num_tokens <= 0) {
        std::cerr << "[Scheduler] Invalid pointers or num_tokens for Overlap MoE GQA Early Exit\n";
        return;
    }

    pipeline_mgr_->set_state(PipelineState::OVERLAPPING_GQA);
    pipeline_mgr_->prepare_next_step_prefetch(next_k_ptrs, next_v_ptrs, {});
    pipeline_mgr_->trigger_gqa_kv_prefetch();
    
    moe_router->route_and_compute(input_hidden_states, output_hidden_states, 
                                   static_cast<uint32_t>(num_tokens));
                                   
    pipeline_mgr_->set_state(PipelineState::IDLE);
}

void ContinuousScheduler::execute_tome_merge(ContinuousRequest* req, const std::vector<float>& hidden_states) {
    // Day 38: Defensive null checks
    if (!tome_tracker_ || !paged_cache || !req) {
        return;
    }
    
    int hidden_dim = paged_cache->get_hidden_dim();
    // Day 38: Edge case guard against division by zero or invalid dimensions
    if (hidden_dim <= 0) {
        return;
    }
    
    int current_seq_len = static_cast<int>(hidden_states.size()) / hidden_dim;
    // Day 38: Edge case guard against zero-token states
    if (current_seq_len <= 0) {
        return;
    }
    
    tome_tracker_->evaluate_and_generate_mask(hidden_states.data(), current_seq_len);
    int blocks_to_free = tome_tracker_->apply_merge_and_get_freed_count(current_seq_len, paged_cache->get_block_size());
    
    // Day 38: Defensive guard against zero or negative block frees to prevent use-after-free or cache corruption
    if (blocks_to_free <= 0) {
        return;
    }
    
    paged_cache->free_merged_blocks(req->request_id, blocks_to_free);
    std::cout << "[Scheduler] ToMe freed " << blocks_to_free << " KV blocks for request " << req->request_id << "\n";
}

void ContinuousScheduler::generate_mtp_drafts(const std::vector<float>& hidden_states) {
    if (!use_mtp_ || !mtp_heads_) return;
    
    for (auto* req : active_batch) {
        if (req != nullptr && req->state == RequestState::RUNNING && req->chunked_state.is_prefill_complete) {
            // Day 38: Ensure hidden_states is not empty before passing to FFI
            if (!hidden_states.empty()) {
                req->pending_draft_tokens = mtp_heads_->generate_drafts(hidden_states.data());
            }
            break; 
        }
    }
}

void ContinuousScheduler::verify_and_update_drafts(SpeculativeDecoder* decoder, int vocab_size, const std::vector<float>& target_logits) {
    std::lock_guard<std::mutex> lock(scheduler_mutex);
    
    int batch_size = static_cast<int>(active_batch.size());
    if (batch_size == 0 || !decoder) return;
    
    int max_d = 8;
    std::vector<int> draft_lengths(batch_size, 0);
    std::vector<int> all_draft_tokens(batch_size * max_d, -1);
    
    for (int i = 0; i < batch_size; ++i) {
        if (active_batch[i] == nullptr) continue;
        int d_len = static_cast<int>(active_batch[i]->pending_draft_tokens.size());
        draft_lengths[i] = std::min(d_len, max_d);
        for (int j = 0; j < draft_lengths[i]; ++j) {
            all_draft_tokens[i * max_d + j] = active_batch[i]->pending_draft_tokens[j];
        }
    }
    
    std::vector<int> accepted_counts(batch_size, 0);
    std::vector<int> replacement_tokens(batch_size, -1);
    
    decoder->verify_drafts_sve2(target_logits, all_draft_tokens, draft_lengths, vocab_size, batch_size, accepted_counts, replacement_tokens);
    
    for (int i = 0; i < batch_size; ++i) {
        if (active_batch[i] == nullptr) continue;
        
        int acc = accepted_counts[i];
        int repl = replacement_tokens[i];
        
        active_batch[i]->speculative_accepted_count += acc;
        active_batch[i]->speculative_total_drafts += draft_lengths[i];
        
        for (int j = 0; j < acc; ++j) {
            active_batch[i]->generated_tokens.push_back(active_batch[i]->pending_draft_tokens[j]);
            if (active_batch[i]->generated_tokens.size() >= 2) {
                int t1 = active_batch[i]->generated_tokens[active_batch[i]->generated_tokens.size() - 2];
                int t2 = active_batch[i]->generated_tokens.back();
                decoder->update_ngram(t1, t2, active_batch[i]->pending_draft_tokens[j]);
            }
        }
        
        if (repl != -1) {
            active_batch[i]->generated_tokens.push_back(repl);
            if (active_batch[i]->generated_tokens.size() >= 2) {
                int t1 = active_batch[i]->generated_tokens[active_batch[i]->generated_tokens.size() - 2];
                int t2 = active_batch[i]->generated_tokens.back();
                decoder->update_ngram(t1, t2, repl);
            }
        }
        
        active_batch[i]->pending_draft_tokens.clear();
        
        if (static_cast<int>(active_batch[i]->generated_tokens.size()) >= active_batch[i]->max_tokens) {
            active_batch[i]->state = RequestState::COMPLETED;
        }
    }
}

void ContinuousScheduler::mark_completed(int request_id) {
    std::lock_guard<std::mutex> lock(scheduler_mutex);
    for (auto* req : active_batch) {
        if (req != nullptr && req->request_id == request_id) {
            req->state = RequestState::COMPLETED;
            if (radix_cache && !req->chunked_state.prompt_tokens.empty()) {
                radix_cache->insert_prefix(request_id, req->chunked_state.prompt_tokens);
            }
            break;
        }
    }
}

ContinuousRequest* ContinuousScheduler::get_request(int request_id) {
    std::lock_guard<std::mutex> lock(scheduler_mutex);
    for (auto* req : active_batch) {
        if (req != nullptr && req->request_id == request_id) {
            return req;
        }
    }
    return nullptr;
}