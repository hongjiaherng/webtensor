# webtensor

A tensor library that runs entirely in the browser. Train, visualize, and run inference with WebGPU acceleration as a TypeScript library.

> **Status: active development — building the foundation.**

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

**Prerequisites:** [Bun](https://bun.sh), `wasm-pack` (for WASM backend build)

```sh
bun install

# Build the WASM backend (required before running tests that use WASMBackend)
cd packages/backend-wasm && wasm-pack build rust --target bundler --out-dir ../pkg && cd ../..

bun run test
```

Tests use [Vitest](https://vitest.dev/). Browser/WebGPU tests run via Playwright (`@vitest/browser`).

---

## What's Implemented

All ops run on CPU, WASM (Rust), and WebGPU with cross-backend parity tests.

| Category         | Ops                                                                           |
| ---------------- | ----------------------------------------------------------------------------- |
| Binary           | add, sub, mul, div                                                            |
| Linalg           | matmul (2D)                                                                   |
| Unary math       | neg, exp, log, sqrt, abs, pow                                                 |
| Activations      | relu, sigmoid, tanh                                                           |
| View (zero-copy) | transpose, reshape, view, slice, unsqueeze, squeeze, permute, expand, flatten |
| Memory           | contiguous, clone, detach                                                     |

**Autograd:** backward through add, sub, mul, div, matmul, transpose, relu, sigmoid, tanh, reshape, contiguous, neg, exp, log, sqrt, abs, pow.

**Dtype system:** `float32 | int32 | bool` — all three can be allocated and round-tripped across all backends. Op kernels currently run on `float32` only.

**Not yet implemented:** softmax, concat, reduce (sum/mean), batched matmul (rank >= 3), loss functions, optimizers, training loop.

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

1. Define the op in [packages/core/src/ops.ts](packages/core/src/ops.ts) with backward closure.
2. Implement kernels in CPU, WASM, and WebGPU backends.
3. Register in each backend's kernel registry.
4. Add tests to [tests/ops/](tests/ops/).
5. Follow the step-by-step [adding an op guide](docs/content/docs/advanced/adding-an-op.mdx).
