#!/bin/bash

## INITIALIZE
export LANG=en_US.UTF-8

EXEC_PATH=`dirname "$0"`
EXEC_PATH=`( cd "$EXEC_PATH" && pwd )`
echo This script executes from $EXEC_PATH

##
#   REMOTE REGISTRY 
##

# REMOTE_REG_SERVER=wasmcloud.azurecr.io
REMOTE_REG_SERVER=ghcr.io/wasmcloud

##
#   CAPABILITY PROVIDERS
##

# HTTPSERVER=httpserver:0.19.1
HTTPSERVER=http-server:0.21.0
REMOTE_HTTPSERVER=$REMOTE_REG_SERVER/$HTTPSERVER
# HTTP_PROVIDER_FILE=$EXEC_PATH/../images/httpserver.par.gz
HTTP_PROVIDER_FILE=$EXEC_PATH/../images/http-server.par.gz

INFERENCE=inference:0.1.0
INFERENCE_PROVIDER_FILE=$EXEC_PATH/../providers/inference/build/inference.par.gz

##
#   COMPONENTS
##

API_ACTOR=api:0.1.0
API_ACTOR_FILE=$EXEC_PATH/../components/api/build/api_s.wasm

IMAGENET_PREPROCESSOR_ACTOR=imagenetpreprocessor:0.1.0
IMAGENET_PREPROCESSOR_ACTOR_FILE=$EXEC_PATH/../components/imagenetpreprocessor/build/imagenetpreprocessor_s.wasm

POSTPROCESSOR_ACTOR=imagenetpostprocessor:0.1.0
POSTPROCESSOR_ACTOR_FILE=$EXEC_PATH/../components/imagenetpostprocessor/build/imagenetpostprocessor_s.wasm

SQUEEZENET_MODEL_ACTOR=squeezenet_model:0.1.0
SQUEEZENET_MODEL_ACTOR_FILE=$EXEC_PATH/../components/model/build/model_s.wasm

##
#   AI MODELS
##
MOBILENETV27_MODEL=wamli-mobilenetv27:latest
MOBILENETV27_MODEL_FILE=mobilenetv27.tar

##
#   LOCAL REGISTRY 
##

HOST_DEVICE_IP=localhost

# oci registry - as used by wash
LOCAL_REG_SERVER=${HOST_DEVICE_IP}:5000

REGISTRY_CONTAINER_NAME="local-docker-registry"

export WASMCLOUD_OCI_ALLOWED_INSECURE=${LOCAL_REG_SERVER}

# echo -e "starting local registry"
# docker run -d -p 5000:5000 --name registry registry:latest

start_local_registry() {
   # Check if the registry container is running
   if ! docker ps | grep -q "$REGISTRY_CONTAINER_NAME"; then
      echo "Local registry is not running. Starting it now..."

      docker run -d -p 5000:5000 --name "$REGISTRY_CONTAINER_NAME" registry:2

      # Check if the registry started successfully
      if [ $? -eq 0 ]; then
         echo -e "\tLocal registry started successfully.\n"
      else
         echo "Failed to start local registry."
         exit 1
      fi
   else
      echo -e "\tLocal registry is already running."
   fi
}

stop_local_registry() {
   echo -e "Ramping down local registry .."
   docker stop $REGISTRY_CONTAINER_NAME
   docker rm -f "$REGISTRY_CONTAINER_NAME"
}

is_image_in_registry() {
   local image_name="$1"
   if curl -sX GET "http://${LOCAL_REG_SERVER}/v2/_catalog" | grep -q "$image_name"; then
      # echo "YES"
      return 0 # exists
   else
      # echo "NO"
      return 1 # does NOT exist
   fi
}

push_artefact() {
   local image_name="$1"
   local local_file="$2"
   local local_registry="$LOCAL_REG_SERVER"
   local remote_registry="$REMOTE_REG_SERVER"

   echo
   echo -e "processing 'push_artefact()' with the following parameters:"
   echo -e "\timage_name: $image_name"
   echo -e "\tlocal_file: $local_file"
   echo -e "\tlocal_registry: $local_registry"
   echo -e "\tremote_registry: $remote_registry"

   # set -x
   while true; do
      # IF image already is in local registry, done
      is_image_in_registry ${image_name}
      if [[ $? -eq 0 ]]; then
        echo -e "${image_name} already IS in local registry\n"
        break
      else
         echo -e "${image_name} is NOT yet in local registry"
      fi
      
      # IF image can be fetched from file, done
      if [[ -f "$local_file" ]]; then
         # The file exists, execute the wash reg push command
         echo -e "${local_file} is available - pushing it to local registry .."
         wash push "$LOCAL_REG_SERVER"/v2/$image_name "$local_file" --insecure
         break
      else
         # The file does not exist, print an error message
         echo "File '$local_file' does not exist."
      fi

      pushd ../images
      echo -e "pulling ${image_name} from remote .."
      wash pull $REMOTE_REG_SERVER/$image_name
      pushd

   done
}

show_images() {
   local local_registry="$1"
   echo -e "\nThe following images are in registry '$1/v2': "
   curl -sX GET "http://${local_registry}/v2/_catalog" | jq
}

##
#   BUSINESS LOGIC
##

wash drain all

stop_local_registry
start_local_registry

push_artefact $HTTPSERVER $HTTP_PROVIDER_FILE
push_artefact $IMAGENET_PREPROCESSOR_ACTOR $IMAGENET_PREPROCESSOR_ACTOR_FILE
push_artefact $POSTPROCESSOR_ACTOR $POSTPROCESSOR_ACTOR_FILE
push_artefact $API_ACTOR $API_ACTOR_FILE
push_artefact $INFERENCE $INFERENCE_PROVIDER_FILE
# push_artefact $SQUEEZENET_MODEL_ACTOR $SQUEEZENET_MODEL_ACTOR_FILE

# Push the reference model to local registry
docker load -i ../images/$MOBILENETV27_MODEL_FILE
docker push $LOCAL_REG_SERVER/$MOBILENETV27_MODEL

# set +x

show_images $LOCAL_REG_SERVER




# echo -e "pulling httpserver from remote"
# wash pull wasmcloud.azurecr.io/httpserver:0.19.1

# echo -e "pulling kvredis from remote"
# wash pull wasmcloud.azurecr.io/kvredis:0.22.0

# echo -e "pushing provider to local registry"
# wash reg push $HTTPSERVER_REF $HTTP_PROVIDER_FILE --insecure

# curl -X GET http://localhost:5000/v2/_catalog







