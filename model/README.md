# SME Model File Format (`.sme`)

The `.sme` file format compiles and stores quantized neural network weights, hyperparameters, and vocabularies specifically optimized for low-overhead loading and low-power ARM servers using SME/SVE2 kernels.

## Binary Layout

The model file is serialized in a single contiguous binary file matching the layout below:

| Offset (Bytes) | Size (Bytes) | Type | Field Name | Description |
|---|---|---|---|---|
| 0 | 9 | `char[]` | Magic Header | Contains ASCII characters `"SME_MODEL"` |
| 9 | 4 | `uint32` | Config Size | Length of the model configuration JSON string |
| 13 | `Config Size` | `char[]` | Config JSON | UTF-8 encoded JSON containing `num_layers`, `hidden_dim`, `num_heads`, etc. |
| ... | 4 | `uint32` | Vocab Size | Total number of tokens in the vocabulary |
| ... | * | * | Vocabulary | Array of `Vocab Size` elements, where each element contains:<br>- `uint32` (4 bytes): Token length<br>- `char[]` (Length bytes): Token string |
| ... | 4 | `uint32` | Tensor Count | Total number of weight matrices/tensors stored in this file |
| ... | * | * | Tensors | Array of `Tensor Count` elements. Details below. |

### Tensor Serialization Layout

For each tensor in the `.sme` file, the following structure is written:

1. **Tensor Name**:
   - `uint32` (4 bytes): Name length
   - `char[]` (Length bytes): Name string (e.g., `"layers.0.q_proj"`)
2. **Dimensions**:
   - `uint32` (4 bytes): Dimensions count (e.g. 2 for matrices)
   - `uint32[]` (Count * 4 bytes): Shape array (e.g., `[hidden_dim, hidden_dim]`)
3. **Quantization Flag**:
   - `uint8` (1 byte): Quantization status (`1` for int8 quantized, `0` for float32 raw weights)
4. **Data Size**:
   - `uint32` (4 bytes): Size of raw tensor data in bytes
5. **Scale Factor Table** (Only written if Quantization Flag is `1`):
   - `float32[]` (Dim[0] * 4 bytes): Array of row-wise scales
6. **Tensor Weights**:
   - `int8[]` or `float32[]` (`Data Size` bytes): Continuous weight memory block

---

## Symmetric Quantization

Our quantization scheme compresses weights per-row to symmetric 8-bit precision:

$$\text{scale}_i = \frac{127.0}{\max_{j} |W_{i, j}|}$$

$$W^{quant}_{i, j} = \text{round}(W_{i, j} \times \text{scale}_i)$$

During matrix-vector multiplication $y = W \cdot x$, the rows are dequantized dynamically in streaming memory:

$$y_i = \frac{1}{\text{scale}_i} \sum_{j} W^{quant}_{i, j} \cdot x_j$$

This is mapped directly to our high-throughput ARM SME/SVE2 vector-accumulate kernels.
