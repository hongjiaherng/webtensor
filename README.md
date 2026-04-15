# minitensor

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

![Package map](docs/diagrams/package-map.svg)

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

## Docs

- [docs/next.md](docs/next.md) — implementation roadmap and checklist
- [docs/architecture.md](docs/architecture.md) — diagrams, design decisions, backend contract
- [docs/ir-reference.md](docs/ir-reference.md) — IR schema: Node/Value/Graph types, shape system
- [docs/adding-an-op.md](docs/adding-an-op.md) — how to add a kernel across all three backends
