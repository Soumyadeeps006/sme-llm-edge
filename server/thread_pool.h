#pragma once
#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include "core_affinity.h"

namespace edge_ai {

// Forward declare the Rust FFI function
extern "C" {
    void rust_sme_transformer_block_i4_f32(
        const uint8_t* a_packed_ptr, const float* b_ptr, const float* scales,
        const float* norm_weight, float* c_ptr,
        uint32_t m, uint32_t k, uint32_t n,
        float eps, float alpha, float beta
    );
}

class ThreadPool {
public:
    explicit ThreadPool(const std::vector<int>& core_ids) : stop(false) {
        if (core_ids.empty()) {
            throw std::invalid_argument("Core IDs list cannot be empty");
        }
        
        for (size_t i = 0; i < core_ids.size(); ++i) {
            workers.emplace_back([this, i, core_id = core_ids[i]] {
                if (pin_thread_to_core(core_id)) {
                    std::cout << "[OK] Worker " << i << " (Thread " << std::this_thread::get_id() 
                              << ") pinned to core " << core_id << "\n";
                }
                
                for (;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this]{ return this->stop || !this->tasks.empty(); });
                        if (this->stop && this->tasks.empty()) return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type> {
        using return_type = typename std::invoke_result<F, Args...>::type;
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop) throw std::runtime_error("enqueue on stopped ThreadPool");
            tasks.emplace([task](){ (*task)(); });
        }
        condition.notify_one();
        return res;
    }

    void dispatch_transformer_chunk(
        const uint8_t* a_packed, const float* b, const float* scales,
        const float* norm_weight, float* c,
        uint32_t m_start, uint32_t m_end, uint32_t k, uint32_t n,
        float eps, float alpha, float beta, uint32_t k_packed_usize, uint32_t n_usize
    ) {
        uint32_t chunk_m = m_end - m_start;
        if (chunk_m == 0) return;

        const uint8_t* chunk_a = a_packed + (m_start * k_packed_usize);
        float* chunk_c = c + (m_start * n_usize);

        enqueue([chunk_a, b, scales, norm_weight, chunk_c, chunk_m, k, n, eps, alpha, beta] {
            rust_sme_transformer_block_i4_f32(
                chunk_a, b, scales, norm_weight, chunk_c,
                chunk_m, k, n, eps, alpha, beta
            );
        });
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread &worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

} // namespace edge_ai