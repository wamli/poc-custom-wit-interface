#!/bin/bash

# Find all processes containing "wasmcloud" in their names
processes=$(ps -aux | grep "wasmcloud" | awk '{print $2}')

# Loop through each process ID and kill it
for pid in $processes; do
   echo -e "killing process $pid .."
   sudo kill $pid
done

echo "All processes containing 'wasmcloud' have been killed."