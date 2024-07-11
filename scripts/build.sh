#!/bin/bash

# Define the command based on the argument
if [ "$1" == "build" ]; then
    command="wash build"
elif [ "$1" == "clean" ]; then
    command="cargo clean"
else
    echo "Invalid argument. Must be 'build' or 'clean'."
    exit 1
fi

# Function to execute the command in subdirectories
execute_command() {
    local dir="../$1"
    for subdir in "$dir"/*; do
        if [ -d "$subdir" ]; then
            echo
            echo "----------------------------------------------"
            echo "Executing '$command' in $subdir"
            echo "----------------------------------------------"
            (cd "$subdir" && $command)
        fi
    done
}

# Execute the command in components and providers directories
execute_command "components"
execute_command "providers"