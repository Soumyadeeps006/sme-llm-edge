# SME‑Optimized Local LLM Home Server

This repository implements a low‑power ARM home server that runs a 7‑14 B LLM (e.g., Llama 4) with custom C++/Rust kernels leveraging ARM SME/SVE2 instructions for high‑throughput inference.

## Directory Layout
```
edge_ai_project/
├─ hardware_spec.md          # Board specifications and power budget
├─ flash_image.sh            # Flash Ubuntu 22.04 onto the board
├─ cross_toolchain.md        # Cross‑compilation toolchain setup
├─ docker_cross_env/
│   └─ Dockerfile            # Docker environment for reproducible builds
├─ src/
│   └─ sme_kernel/           # C++/Rust kernel sources
├─ CMakeLists.txt            # Build configuration
├─ benchmarks/
│   └─ benchmark_sme.cpp     # Micro‑benchmark for kernels
├─ convert_model.py          # Model conversion & quantization script
├─ model/README.md           # Model metadata
├─ server/                   # HTTP inference server source
├─ config.yaml               # Runtime configuration
├─ rag/                      # Optional Retrieval‑Augmented Generation pipeline
├─ docs/                     # Documentation
├─ tests/                    # Unit and integration tests
└─ .github/workflows/ci.yml # CI pipeline
```

## Getting Started
1. Review `hardware_spec.md` and procure the board.
2. Flash Ubuntu 22.04 LTS using `flash_image.sh`.
3. Set up the cross‑compilation environment per `cross_toolchain.md`.
4. Build the kernels and server (`mkdir build && cd build && cmake .. && make`).
5. Convert the LLM model with `python3 convert_model.py`.
6. Run the inference server (`./server/llm_server`).

## License
© 2026 Your Name. MIT License.
