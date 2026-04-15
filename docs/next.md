# Roadmap Checklist

Check an item off (`[x]`) when it is fully implemented and tested on all target backends.

---

## Ops

| Op                          | CPU | WASM | WebGPU |
| --------------------------- | :-: | :--: | :----: |
| Add                         | [x] | [x]  |  [x]   |
| Sub                         | [x] | [x]  |  [x]   |
| Mul                         | [x] | [x]  |  [x]   |
| Div                         | [x] | [x]  |  [x]   |
| MatMul (2D)                 | [x] | [x]  |  [x]   |
| MatMul (rank ≥ 3, max 64)   | [ ] | [ ]  |  [ ]   |
| Transpose                   | [x] | [x]  |  [x]   |
| Relu                        | [x] | [x]  |  [x]   |
| Sigmoid                     | [ ] | [ ]  |  [ ]   |
| Tanh                        | [ ] | [ ]  |  [ ]   |
| Softmax                     | [ ] | [ ]  |  [ ]   |
| Reshape                     | [x] | [x]  |  [x]   |
| Slice                       | [x] | [x]  |  [x]   |
| Concat                      | [ ] | [ ]  |  [ ]   |
| Reduce (Sum / Mean)         | [ ] | [ ]  |  [ ]   |
| Exp                         | [ ] | [ ]  |  [ ]   |
| Log                         | [ ] | [ ]  |  [ ]   |

## Autograd

| Feature                            | Done |
| ---------------------------------- | :--: |
| Add backward                       | [x]  |
| Mul backward                       | [x]  |
| MatMul backward                    | [x]  |
| Transpose backward                 | [x]  |
| Relu backward — CPU + WASM         | [x]  |
| Relu backward — WebGPU             | [ ]  |
| Sub backward                       | [ ]  |
| Div backward                       | [ ]  |
| Reduce backward                    | [ ]  |
| Sigmoid / Tanh backward            | [ ]  |
| Softmax backward                   | [ ]  |

## Training

| Feature                      | Done |
| ---------------------------- | :--: |
| Loss: MSE                    | [ ]  |
| Loss: CrossEntropy           | [ ]  |
| Optimizer: SGD               | [ ]  |
| Optimizer: Adam              | [ ]  |
| Training loop in `core`      | [ ]  |

## Dtypes

A cell is checked when the dtype is fully supported on that backend: type system, `allocate()`, and kernels for all applicable ops.

| Dtype     | CPU | WASM | WebGPU | Notes                                            |
| --------- | :-: | :--: | :----: | ------------------------------------------------ |
| `float32` | [x] | [x]  |  [x]   | Primary dtype                                    |
| `float16` | [ ] | [ ]  |  [ ]   | Half-precision; important for WebGPU efficiency  |
| `int32`   | [ ] | [ ]  |  [ ]   | Type system + `allocate()` exist; kernels missing|
| `int64`   | [ ] | [ ]  |  [ ]   | Needed for index ops (Gather, Scatter)           |
| `int8`    | [ ] | [ ]  |  [ ]   | Quantization                                     |
| `bool`    | [ ] | [ ]  |  [ ]   | Type system + `allocate()` exist; no op kernels  |

## Infrastructure

| Feature                                        | Done |
| ---------------------------------------------- | :--: |
| Strided tensor model (strides, offset, views)  | [x]  |
| Broadcasting via stride-0                      | [x]  |
| Cross-backend parity tests                     | [x]  |
| Max rank = 64 (package-wide constraint)        | [ ]  |
| Async engine execution model                   | [ ]  |

## Packages

| Package                                              | Done |
| ---------------------------------------------------- | :--: |
| `packages/onnx` — protobuf parser, op mapping to IR  | [ ]  |
| `packages/devtools` — graph viz, training dashboard  | [ ]  |
