#!/bin/bash

EXEC_PATH=`dirname "$0"`
EXEC_PATH=`( cd "$EXEC_PATH" && pwd )`
echo This script executes from $EXEC_PATH

# start the host first - but in a different shell
# wash up --nats-websocket-port 4223 --allowed-insecure localhost:5000
# wash up --allowed-insecure localhost:5000

wash app list

wash app deploy $EXEC_PATH/../wadm.yaml

wash app list

# wash app delete model-on-demand v0.1.0