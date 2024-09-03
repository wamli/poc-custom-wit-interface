#!/bin/bash

nats stream del wadm_commands --force
nats stream del wadm_events --force
nats stream del wadm_event_consumer --force
nats stream del wadm_notify --force
nats stream del wadm_status --force
nats stream del KV_wadm_state --force
nats stream del KV_wadm_manifests --force