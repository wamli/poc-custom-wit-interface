# Proof of concept (POC)

The __mission__ of this poc is to examine how to define a wasmcloud **actor** and **provider** where both communicate via a custom interface that is based on WIT.

From a business perspective, we assume the actor to contain some state which is supposed to be offloaded to a provider. 

> **NOTE:** wasmcloud actors are intended to be [stateless](https://wasmcloud.com/docs/concepts/actors#stateless-by-design). In order to resolve this paradoxon the constraint is that the kind of state we talk about here is __*immutable*__.

The archetype of this poc is [wasmcloud-ollama](https://github.com/Iceber/wasmcloud-ollama).

## What is wasmCloud
wasmCloud is a sandbox project of CNCF, which is mainly used for distributed deployment and management of Wasm, and breaks the function limitation of Wasm Runtime through the Actor-Provider way, allowing users to provide customized capabilities according to their business.

For more information about wasmCloud, please visit the [repository](https://github.com/wasmCloud/wasmCloud) and the [website](https://wasmcloud.com/).

## What is WIT?

The [Wasm Interface](https://github.com/WebAssembly/component-model/blob/main/design/mvp/WIT.md) Type (WIT) format is an **IDL** to provide tooling for the [WebAssembly Component Model](https://github.com/webassembly/component-model).

## Actor

### Design

* The actor contains *immutable* state which it may export via its [interface](./stateful-actor/wit/deps/wamli/stateful.wit).
* Additionally, there is an [adaptor](./stateful-actor/wit/world.wit) which imports `wasmcloud:bus/host` in order to be compatible to wasmcloud.
* The [implementation](./stateful/src/lib.rs) is a dummy right now. However, it is able to demonstrate the mechanism.

### Open Questions

1. Each actor needs to *claim* the capabilities it has to access in [wasmcloud.toml](./stateful-actor/wasmcloud.toml). Since the intended interface currently only exists in a local .wit file, what to put in here?

## Provider

The provider of [wasmcloud-ollama](https://github.com/Iceber/wasmcloud-ollama) is implemented in to. Even though, it clearly implements the functions `Chat`, `List`, `Show`, it does not seem to use the WIT (nor any other).

### Open Questions

Given the provider to implement shall be implemented in Rust, what are the constraints? Would it be enough to *"just"* implement the WIT, e.g.

```rust
wasmtime::component::bindgen!({
    path: "./wit/world.wit",
    world: "interfaces",
    async: false
});
```

And IF that was the case, how to satisfy `wasmcloud:bus/host`?
