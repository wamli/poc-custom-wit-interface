## Build

```bash
cargo build --release

# using wash par create, e.g.
wash par create --capid wamli:mlinference --vendor wamli --name fakeml --arch x86_64-linux --binary ../target/release/wamli
mv wamli.par build/fakeml.par
```

## Run

```bash
wash up --nats-websocket-port 4001 --allowed-insecure localhost:5000

wash app deploy wadm.yaml

wash app list

wash app delete model-on-demand v0.1.0
```