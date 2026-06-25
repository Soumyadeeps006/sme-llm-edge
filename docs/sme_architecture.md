# ARM SME (Scalable Matrix Extension) Architecture

This document explains the hardware mechanisms of ARM Scalable Matrix Extension (SME) and how it accelerates large language model (LLM) inference compared to SVE2 (Scalable Vector Extension 2) and traditional Neon vector instructions.

## Overview of SME

SME builds on top of SVE2 to target high-throughput matrix computations. The key architectural features are:

1. **Streaming SVE Mode**: SME introduces a new mode of operation where vector instructions execute with a dynamically determined Scalable Vector Length (SVL), decoupled from the standard CPU core SVE state.
2. **ZA Array Storage**: SME adds a new 2D accumulator register state, called the **ZA storage array**. ZA is a square tile structure (SVL × SVL bytes) that remains active and shared across function calls, preventing register spills during nested loops.
3. **Outer Product Engine**: SME adds hardware outer product instructions (like `MOPA` - Matrix Outer Product and Accumulate) that multiply a vector of inputs and a vector of weights, accumulating the results directly into a tile of the ZA array in a single cycle.

---

## Neon vs. SVE2 vs. SME

| Feature | Neon (ASIMD) | SVE2 | SME |
|---|---|---|---|
| **Register Width** | Fixed (128-bit) | Scalable (128-bit to 2048-bit) | Scalable (ZA Array: SVL × SVL bytes) |
| **Execution Model** | Vector-register | Vector-scalable (predicate-driven) | Matrix-accumulator (Outer product) |
| **Core Instruction** | FMA (Float Multiply-Accumulate) | FMA (Scalable width) | MOPA (Matrix Outer Product Accumulate) |
| **Primary Use Case** | General DSP, multimedia | Vector processing, loops | GEMM, Deep Learning inference |

---

## SME Outer Product Operation

For a matrix multiplication $C = A \times B$, traditional architectures calculate inner products (dot products). In contrast, SME computes using **outer products**:

$$C \leftarrow C + \vec{u} \otimes \vec{v}^T$$

where $\vec{u}$ is a vector of activations (loaded from matrix $A$), $\vec{v}^T$ is a vector of weights (loaded from matrix $B$), and $C$ is a sub-tile of the ZA accumulator array.

### Assembly / Instruction Mapping

In assembly, loading weights and calculating the outer product is represented as:

```assembly
// Load vector of 8-bit inputs into SVE vector z0
ld1b {z0.b}, p0/z, [x0]

// Load vector of 8-bit weights into SVE vector z1
ld1b {z1.b}, p1/z, [x1]

// Accumulate outer product of z0 and z1 into ZA tile 0
svmopa za0.s, p0/m, p1/m, z0.b, z1.b
```

This hardware-level integration allows our LLM server to process matrix weights with minimal power consumption, reaching up to 4x efficiency gains compared to Neon or SVE2 vectors.
