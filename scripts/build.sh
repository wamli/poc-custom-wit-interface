#!/bin/bash

BASE_DIR=providers/inference

cargo build --release --manifest-path "$BASE_DIR/Cargo.toml"

wash par create --capid wamli:mlinference --vendor wamli --name inference --arch x86_64-linux --binary $BASE_DIR/target/release/wamli
mv wamli.par $BASE_DIR/build/inference.par

echo -e "Provider build: "
wash inspect $BASE_DIR/build/inference.par