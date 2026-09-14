cmake_minimum_required(VERSION 3.15)
project(EdgeAIServer VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(RUST_LIB_PATH "${CMAKE_SOURCE_DIR}/src/sme_kernel/rust_sme/target/aarch64-unknown-linux-gnu/release/librust_sme.a")
if(NOT EXISTS ${RUST_LIB_PATH})
    message(FATAL_ERROR "Rust static library not found. Run: cd src/sme_kernel/rust_sme && cargo build --release")
endif()

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(edge_server 
    server/main.cpp 
    server/sme_gemm.cpp
    server/model_loader.cpp
    server/http_streaming.cpp
    server/radix_kv_cache.cpp
    server/paged_kv_cache.cpp
    server/continuous_scheduler.cpp
    server/speculative_decoder.cpp
    server/chunked_prefill.cpp
    server/paged_attention.cpp
    server/inference_queue.cpp
    server/sve_packing.cpp
    server/pipeline_manager.cpp
    server/early_exit_tracker.cpp
    server/tome_tracker.cpp             
    server/eviction_tracker.cpp         
    server/flash_decoding_tracker.cpp   
    server/mtp_heads.cpp                
)

target_link_libraries(edge_server PRIVATE 
    ${RUST_LIB_PATH}
    Threads::Threads
    dl
    m
    OpenMP::OpenMP_CXX
)

target_include_directories(edge_server PRIVATE 
    ${CMAKE_SOURCE_DIR}/server
    ${CMAKE_SOURCE_DIR}/src/sme_kernel/rust_sme/src
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(edge_server PRIVATE -O3 -march=native)
endif()

# ==============================================================================
# Day 38: AddressSanitizer (ASan) Build Configuration
# ==============================================================================
if(CMAKE_BUILD_TYPE STREQUAL "ASAN")
    message(STATUS "🛡️  Building with AddressSanitizer (ASan) enabled")
    target_compile_options(edge_server PRIVATE 
        -fsanitize=address 
        -fno-omit-frame-pointer 
        -g 
        -O1
    )
    target_link_options(edge_server PRIVATE 
        -fsanitize=address
    )
endif()