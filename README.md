# webtensor

A tensor library that runs entirely in the browser. Train, visualize, and run inference with WebGPU acceleration as a TypeScript library.

> **Status: active development — building the foundation.**

## Next time when i revisit this project

- [ ] differentiate onnx ops vs our own ops (mostly grad of ops) in registry.ts for onnx interopt
- [ ] benchmark against pytorch
- [ ] update docs to reflect current state of implementation and roadmap

Taking off from the project...

---

## Vision

- **Train models in the browser.** Torch-like authoring API, autograd, and optimizer loop, executing on WebGPU, WASM (Rust), and CPU (TypeScript).
- **Visualize model dynamics in real-time.** Watch weights, activations, and gradients update as training runs.
- **Load and run ONNX models.** Import a `.onnx` file and execute it via the same backend pipeline.
- **Drop into a React app.** All packages are TypeScript-first and browser-native.

---

## Architecture

![Package map](/docs/public/diagrams/package-map.svg)

**Boundary rules:**

| Package     | Can                                                         | Cannot                              |
| ----------- | ----------------------------------------------------------- | ----------------------------------- |
| `core`      | build graphs, eager ops, autograd                           | know backend memory layout          |
| `ir`        | describe computation, shapes, attributes                    | know about devices or gradients     |
| `runtime`   | execute graphs, dispatch to backend, manage tensor lifetime | implement op math                   |
| `backend-*` | run kernels, own memory                                     | define user-facing tensor semantics |

---

## Getting Started

**Prerequisites:** [Bun](https://bun.sh), [`wasm-pack`](https://rustwasm.github.io/wasm-pack/) (for WASM backend build)

```sh
# Install all dependencies (root workspace + docs)
bun run setup

# Build all packages (chains wasm-pack → bun build → tsc for backend-wasm)
bun run build

# Run tests (CPU + WASM + WebGPU parity, 1e-5 tolerance)
bun run test
```

If you're only iterating on the WASM backend, `bun run build:backend-wasm` rebuilds just that package (`wasm-pack` → bundler plugin → type emit).

Tests use [Vitest](https://vitest.dev/). Browser/WebGPU tests run via Playwright (`@vitest/browser`).

---

## What's Implemented

All ops run on CPU, WASM (Rust), and WebGPU with cross-backend parity tests.

| Category         | Ops                                                                                                                |
| ---------------- | ------------------------------------------------------------------------------------------------------------------ |
| Binary           | add, sub, mul, div                                                                                                 |
| Compare          | eq, ne, lt, le, gt, ge, isclose                                                                                    |
| Linalg           | matmul — 1D·1D (dot), 1D·2D, 2D·1D, 2D·2D, N-D batched with broadcast                                              |
| Unary math       | neg, exp, log, sqrt, abs, pow                                                                                      |
| Activations      | relu, sigmoid, tanh, softmax                                                                                       |
| Reductions       | sum, mean, all, any (arbitrary axes, keepdim)                                                                      |
| Movement         | transpose, reshape, view, slice, unsqueeze, squeeze, permute, expand, flatten, concat, pad                         |
| Memory           | contiguous, clone, detach                                                                                          |
| Cast             | cast (float32 ↔ int32 ↔ bool)                                                                                      |

**Autograd:** backward through all differentiable ops above (compares, isclose, cast, and bool-producing ops break the gradient chain by design).

**Training:** `SGD` (with optional momentum), `mseLoss`, all activations in `@webtensor/nn`. End-to-end XOR MLP trains on all three backends. `compile(fn, spec)` + `grad(loss, param)` cache the traced graph for repeat calls.

**Dtype system:** `float32 | int32 | bool` — all three can be allocated and round-tripped. Binary arithmetic supports `float32` and `int32` across all backends; unary / reductions / matmul on `int32` are CPU-only for now.

**Not yet implemented:** `cross_entropy`, `bceLoss`, `logSoftmax`, `Adam`, `gather` / `scatter`, conv ops.

---

## Docs

Full documentation is built with [Fumadocs](https://fumadocs.dev/). Start with the [onboarding series](docs/content/docs/onboarding/) for a guided tour, or jump to the [advanced reference](docs/content/docs/advanced/) for architecture and IR specs.

**Key guides:**

- [Roadmap](docs/content/docs/roadmap.mdx) — implementation roadmap and checklist
- [Architecture](docs/content/docs/advanced/architecture.mdx) — diagrams, design decisions, backend contract
- [IR Reference](docs/content/docs/advanced/ir-reference.mdx) — IR schema: Node/Value/Graph types, shape system, dtype reference
- [Adding an Op](docs/content/docs/advanced/adding-an-op.mdx) — how to add a kernel across all three backends

**Local development:**

```sh
bun run docs:dev          # Start docs site at http://localhost:3000
bun run docs:build        # Build static docs for GitHub Pages
bun run docs:plantuml     # Regenerate SVG diagrams from .puml sources
```

---

## Contributing

See the full [contributor guide](docs/content/docs/onboarding/contributor-guide.mdx) in the docs.

**Quick start for adding a new op:**

1. Define the op in [packages/core/src/ops/](packages/core/src/ops/) under the matching category (elementwise / reduction / linalg / activation / movement / memory) with a backward closure.
2. Implement kernels in CPU, WASM, and WebGPU backends.
3. Register in each backend's kernel registry.
4. Add tests to [tests/ops/](tests/ops/).
5. Follow the step-by-step [adding an op guide](docs/content/docs/advanced/adding-an-op.mdx).
