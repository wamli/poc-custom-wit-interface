# Stateful Actor

## Prerequisites

- `cargo` 1.74
- `wash` 0.25.0
- `wasmtime` 16.0.0 (if running with wasmtime)

## Building

```bash
# set the environment variable 'AI_MODEL', e.g.
export AI_MODEL=/home/finnfalter/git/wamli/mlinference-provider/bindle/models/squeezenetv1-1-7.onnx

wash build
```

