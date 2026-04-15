DIAGRAMS_DIR := docs/diagrams

.DEFAULT_GOAL := help

.PHONY: help diagrams wasm test test-watch

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  diagrams    Regenerate SVGs from .puml sources in $(DIAGRAMS_DIR)/"
	@echo "  wasm        Build the WASM backend (requires wasm-pack)"
	@echo "  test        Run full test suite"
	@echo "  test-watch  Run tests in watch mode"

diagrams:
	plantuml -tsvg $(DIAGRAMS_DIR)/*.puml

wasm:
	cd packages/backend-wasm/rust && wasm-pack build --target bundler --out-dir ../pkg

test:
	bun run test

test-watch:
	bun run test:watch
