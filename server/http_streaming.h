#pragma once
#include <string>
#include <string_view>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <unordered_map>
#include <cstdint>

#ifdef _WIN32
#include <winsock2.h>
typedef int SOCKET;
#else
#include <sys/socket.h>
typedef int SOCKET;
#endif

struct StreamSession {
    SOCKET client_fd;
    std::queue<int32_t> token_queue; // Zero-copy: store token IDs, not strings
    const char** vocab;              // Pre-loaded memory-mapped vocabulary array
    std::mutex mtx;
    std::condition_variable cv;
    bool finished = false;
    std::thread io_thread;
};

class StreamingManager {
private:
    std::unordered_map<int, StreamSession*> sessions;
    std::mutex map_mutex;
    const char** global_vocab;

    void io_worker(StreamSession* session);
    StreamingManager() : global_vocab(nullptr) {}

public:
    static StreamingManager& instance() {
        static StreamingManager inst;
        return inst;
    }

    void set_vocab(const char** vocab_ptr) { global_vocab = vocab_ptr; }
    void start_session(int request_id, SOCKET client_fd, bool stream);
    void push_token_id(int request_id, int32_t token_id);
    void end_session(int request_id);
};

// Legacy direct functions (kept for non-streaming fallback)
void send_stream_headers(SOCKET client_fd);
void send_token_chunk(SOCKET client_fd, std::string_view token);
void send_stream_end(SOCKET client_fd);