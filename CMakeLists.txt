cmake_minimum_required(VERSION 3.15)
project(EdgeAIProject CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compile flags and options
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    message(STATUS "Configuring for ARM64 with SME and SVE2 support")
    # Using armv8.6-a+sve2+sme for targeting Neoverse V3 or similar cores
    add_compile_options(-march=armv8.6-a+sve2+sme -O3 -Wall -Wextra)
else()
    message(STATUS "Configuring for host architecture (${CMAKE_SYSTEM_PROCESSOR}) with portable fallbacks")
    add_compile_options(-O3 -Wall -Wextra)
endif()

# Rust static library integration
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(RUST_TARGET "aarch64-unknown-linux-gnu")
    set(RUST_LIB_PATH "${CMAKE_CURRENT_BINARY_DIR}/rust_target/${RUST_TARGET}/release/librust_sme.a")
    set(CARGO_ARGS --target ${RUST_TARGET})
else()
    set(RUST_LIB_PATH "${CMAKE_CURRENT_BINARY_DIR}/rust_target/release/librust_sme.a")
    set(CARGO_ARGS "")
endif()

add_custom_command(
    OUTPUT ${RUST_LIB_PATH}
    COMMAND cargo build --release ${CARGO_ARGS} --target-dir ${CMAKE_CURRENT_BINARY_DIR}/rust_target
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/sme_kernel/rust_sme
    COMMENT "Building Rust static library (${RUST_LIB_PATH})"
)
add_custom_target(rust_sme_lib DEPENDS ${RUST_LIB_PATH})

add_library(rust_sme STATIC IMPORTED GLOBAL)
set_target_properties(rust_sme PROPERTIES
    IMPORTED_LOCATION ${RUST_LIB_PATH}
)
add_dependencies(rust_sme rust_sme_lib)

# C++ SME Kernel library
add_library(sme_kernel STATIC src/sme_kernel/sme_gemm.cpp)
target_include_directories(sme_kernel PUBLIC src/sme_kernel)

# Inference Server Executable
add_executable(llm_server server/main.cpp)
target_link_libraries(llm_server PRIVATE sme_kernel rust_sme)
if(NOT WIN32)
    target_link_libraries(llm_server PRIVATE pthread dl)
endif()

# Benchmarks
add_executable(benchmark_sme benchmarks/benchmark_sme.cpp)
target_link_libraries(benchmark_sme PRIVATE sme_kernel rust_sme)
if(NOT WIN32)
    target_link_libraries(llm_server PRIVATE pthread dl)
endif()
