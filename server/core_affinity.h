#pragma once
#include <thread>
#include <vector>
#include <iostream>

#if defined(_WIN32) || defined(_WIN64)
    #include <windows.h>
    inline bool pin_thread_to_core(int core_id) {
        DWORD_PTR mask = 1ULL << core_id;
        HANDLE hThread = GetCurrentThread();
        DWORD_PTR result = SetThreadAffinityMask(hThread, mask);
        if (result == 0) {
            std::cerr << "[WARN] Failed to pin thread to core " << core_id << std::endl;
            return false;
        }
        return true;
    }
#else
    #include <sched.h>
    inline bool pin_thread_to_core(int core_id) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core_id, &cpuset);
        if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0) {
            std::cerr << "[WARN] Failed to pin thread to core " << core_id << std::endl;
            return false;
        }
        return true;
    }
#endif

namespace edge_ai {

class CoreAffinity {
public:
    // Pin the current thread to a specific logical core ID
    static bool set_affinity(int core_id) {
        return pin_thread_to_core(core_id);
    }

    // Get a list of recommended "Big" core IDs (e.g., upper half of available cores)
    static std::vector<int> get_big_core_ids() {
        unsigned int num_cores = std::thread::hardware_concurrency();
        std::vector<int> big_cores;
        // Heuristic: Assume the second half of cores are "Big" performance cores
        unsigned int start_core = num_cores / 2;
        for (unsigned int i = start_core; i < num_cores; ++i) {
            big_cores.push_back(static_cast<int>(i));
        }
        return big_cores.empty() ? std::vector<int>{0} : big_cores;
    }
};

} // namespace edge_ai

// Global helper for round-robin affinity mapping (as specified in Action Plan)
inline void setup_thread_affinity(int thread_index, int total_cores) {
    int target_core = thread_index % total_cores;
    if (pin_thread_to_core(target_core)) {
        std::cout << "[OK] Thread " << std::this_thread::get_id() << " pinned to core " << target_core << "\n";
    }
}