#!/bin/bash

BASE_DIR=providers/fakeml

cargo build --release --manifest-path "$BASE_DIR/Cargo.toml"

wash par create --capid wamli:mlinference --vendor wamli --name fakeml --arch x86_64-linux --binary $BASE_DIR/target/release/wamli
mv wamli.par $BASE_DIR/build/fakeml.par

echo -e "Provider build: "
wash inspect $BASE_DIR/build/fakeml.par