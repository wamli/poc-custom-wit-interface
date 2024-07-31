# Inference application based on WIT components

## Architecture

![application](docs/images/application.png)

* Circles are __components__
* Rounded rectangles are __providers__

The application has two endpoints (only http is shown in the diagram)
* http
* nats

### Http endpoint

Some example API calls

```bash
curl -X GET localhost:8081/prefetch/wamli-mobilenetv27%3Alatest

curl -T ../data/imagenet/cat.jpg localhost:8081/preprocessing-only/wamli-mobilenetv27%3Alatest

curl -T ../data/imagenet/cat.jpg localhost:8081/wamli-mobilenetv27%3Alatest
```

### NATS endpoint

Experimental for now, just returning an echo. Some example API calls,
requires [NATS-cli](https://github.com/nats-io/natscli) to be installed 

```bash
nats req "wasmcloud.echo" "HELLLLOOOOOO"

# Sending a file
nats req "wasmcloud.echo" "$(cat ../data/imagenet/cat.jpg)"
```

## Deployment

```bash
# build all artifacts first
scripts/build.sh build
# note that you may call the same script with 'clean' to cleanup

# load the artifacts into a local registry
scripts/configure.sh

# start wasmcloud in a first window
wash up --allowed-insecure localhost:5000

# start the application
scripts/start.sh

# do some inference
curl -T ../data/imagenet/cat.jpg localhost:8081/preprocessing-only/wamli-mobilenetv27%3Alatest
```

## Debugging the application

```bash
# Start the nats server separately in a first terminal
nats-server -js -V
# start as follows in order to record the logs
# nats-server -js -V &> nats-logs.txt

# start wasmcloud in a second terminal
# make sure the port matches the one of nats server
wash up --nats-websocket-port 4222 --allowed-insecure localhost:5000
# append '> scripts/wasmcloud-logs.txt' in order to record the logs

# watch nats subsciptions in a third terminal
nats sub '*.*.wrpc.>'

# wath in a fourth terminal
nats sub '_INBOX.*.>'

# deploy from a fifth terminal
scripts/configure.sh && scripts/start.sh
```

## Debugging application and runtime

```bash
# Start the nats server separately in a first terminal
nats-server -js -V
# start as follows in order to record the logs
# nats-server -js -V &> nats-logs.txt

# start wasmcloud host, e.g. from `main` branch, in a second terminal
cargo run --release -- --allowed-insecure localhost:5000 --allow-file-load

# start wadm, e.g. from `main` branch, in a third terminal
cargo run --release --bin wadm

# watch nats subsciptions in a fourth terminal
nats sub '*.*.wrpc.>'

# wath in a fifth terminal
nats sub '_INBOX.*.>'

# deploy from a sixth terminal
scripts/configure.sh && scripts/start.sh
```