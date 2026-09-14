#include "http_streaming.h"
#include <iostream>
#include <cstring>
#include <cstdio>

#ifdef _WIN32
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")
#else
#include <unistd.h>
#include <netinet/in.h>
#define closesocket close
#endif

void StreamingManager::io_worker(StreamSession* session) {
    while (true) {
        int32_t token_id = -1;
        bool is_done = false;
        
        {
            std::unique_lock<std::mutex> lock(session->mtx);
            session->cv.wait(lock, [session] { 
                return !session->token_queue.empty() || session->finished; 
            });
            
            if (!session->token_queue.empty()) {
                token_id = session->token_queue.front();
                session->token_queue.pop();
            } else if (session->finished) {
                is_done = true;
            }
        }
        
        if (is_done) {
            const char* end_payload = "data: [DONE]\n\n";
            send(session->client_fd, end_payload, strlen(end_payload), 0);
            break;
        }
        
        if (token_id >= 0) {
            // Zero-copy vocabulary lookup (bounds check for safety)
            const char* token_str = (session->vocab && token_id < 100000) ? session->vocab[token_id] : "<unk>";
            
            // Stack-allocated payload formatting to avoid heap allocation
            char payload[256];
            int len = std::snprintf(payload, sizeof(payload), "data: {\"token\": \"%s\"}\n\n", token_str);
            
            if (len > 0 && len < sizeof(payload)) {
                int sent = 0;
                while (sent < len) {
                    int n = send(session->client_fd, payload + sent, len - sent, 0);
                    if (n <= 0) {
                        goto cleanup; // Connection dropped or error
                    }
                    sent += n;
                }
            }
        }
    }
    
cleanup:
    closesocket(session->client_fd);
    delete session;
}

void StreamingManager::start_session(int request_id, SOCKET client_fd, bool stream) {
    if (!stream) return; 
    
    StreamSession* session = new StreamSession();
    session->client_fd = client_fd;
    session->vocab = global_vocab;
    
    std::lock_guard<std::mutex> lock(map_mutex);
    sessions[request_id] = session;
    
    session->io_thread = std::thread(&StreamingManager::io_worker, this, session);
    session->io_thread.detach();
    
    const char* headers = "HTTP/1.1 200 OK\r\n"
                          "Content-Type: text/event-stream\r\n"
                          "Cache-Control: no-cache\r\n"
                          "Connection: keep-alive\r\n\r\n";
    send(session->client_fd, headers, strlen(headers), 0);
}

void StreamingManager::push_token_id(int request_id, int32_t token_id) {
    std::lock_guard<std::mutex> map_lock(map_mutex);
    auto it = sessions.find(request_id);
    if (it != sessions.end()) {
        StreamSession* session = it->second;
        {
            std::lock_guard<std::mutex> qlock(session->mtx);
            session->token_queue.push(token_id);
        }
        session->cv.notify_one();
    }
}

void StreamingManager::end_session(int request_id) {
    std::lock_guard<std::mutex> map_lock(map_mutex);
    auto it = sessions.find(request_id);
    if (it != sessions.end()) {
        StreamSession* session = it->second;
        {
            std::lock_guard<std::mutex> qlock(session->mtx);
            session->finished = true;
        }
        session->cv.notify_one();
        sessions.erase(it);
    }
}

void send_stream_headers(SOCKET client_fd) {
    const char* headers = "HTTP/1.1 200 OK\r\n"
                          "Content-Type: text/event-stream\r\n"
                          "Cache-Control: no-cache\r\n"
                          "Connection: keep-alive\r\n\r\n";
    send(client_fd, headers, strlen(headers), 0);
}

void send_token_chunk(SOCKET client_fd, std::string_view token) {
    char payload[256];
    int len = std::snprintf(payload, sizeof(payload), "data: {\"token\": \"%.*s\"}\n\n", 
                            static_cast<int>(token.length()), token.data());
    if (len > 0) {
        send(client_fd, payload, len, 0);
    }
}

void send_stream_end(SOCKET client_fd) {
    const char* end_payload = "data: [DONE]\n\n";
    send(client_fd, end_payload, strlen(end_payload), 0);
}