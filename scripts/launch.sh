#!/bin/bash

# start the host first - but in a different shell
# wash up --nats-websocket-port 4001 --allowed-insecure localhost:5000

wash app list

wash app deploy wadm.yaml

wash app list