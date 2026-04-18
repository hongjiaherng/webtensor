# webtensor smoke-test

A standalone Bun + React app that imports the **published** `@webtensor/*`
packages (from npm, not workspace aliases) and runs a small battery of ops on
all three backends. Use this after a release to verify the `dist/` bundles
actually work for a real consumer.

## Install

```bash
bun install
```

## Run

```bash
bun dev
```

Open the dev URL, click **Run tests**, and confirm all three backends pass.

## What it exercises

- Side-effect backend registration (`import '@webtensor/backend-cpu'` etc.).
- The high-level `run(tensor, { device })` helper.
- Basic ops: `add`, `mul`, `matmul`, broadcasting, `sum`, `softmax`, `relu`, `eq`.
- A full training micro-step: `compile(fn, spec)` + `grad(loss, param)` +
  `mseLoss` + `SGD.step(...)` fitting a 2-parameter linear model on all three
  backends.
