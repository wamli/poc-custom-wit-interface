#!/bin/bash

##
#   REMOTE REGISTRY 
##

REMOTE_REG_SERVER=wasmcloud.azurecr.io

##
#   CAPABILITY PROVIDERS
##

HTTPSERVER=httpserver:0.19.1
REMOTE_HTTPSERVER=$REMOTE_REG_SERVER/$HTTPSERVER
HTTP_PROVIDER_FILE=images/httpserver.par.gz

FAKE_ML=fakeml:0.1.0
FAKE_ML_PROVIDER_FILE=providers//build/fakeml.par

##
#   ACTORS
##

ECHO_ACTOR=echo:0.3.4
ECHO_ACTOR_FILE=images/echo.wasm

API_ACTOR=api:0.1.0
API_ACTOR_FILE=actors/api/build/api_s.wasm

SQUEEZENET_MODEL_ACTOR=squeezenet_model:0.1.0
SQUEEZENET_MODEL_ACTOR_FILE=../poc-actor-from-ai-model/model/build/model_s.wasm


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

   echo -e "processing 'push_artefact()' with the following parameters:"
   echo -e "\timage_name: $image_name"
   echo -e "\tlocal_file: $local_file"
   echo -e "\tlocal_registry: $local_registry"
   echo -e "\tremote_registry: $remote_registry"

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
         echo "Error: File '$local_file' does not exist."
      fi

      pushd images
      echo -e "pulling ${image_name} from remote .."
      wash pull $REMOTE_REG_SERVER/$image_name
      pushd

   done
}

show_images() {
   local local_registry="$1"
   echo -n "The following images are in $1/v2: "
   # curl -sX GET "http://localhost:5000/v2/_catalog"
   curl -sX GET "http://${local_registry}/v2/_catalog"
}

##
#   BUSINESS LOGIC
##

wash drain all

stop_local_registry
start_local_registry

push_artefact $HTTPSERVER $HTTP_PROVIDER_FILE
# push_artefact $ECHO_ACTOR $ECHO_ACTOR_FILE
push_artefact $API_ACTOR $API_ACTOR_FILE
push_artefact $FAKE_ML $FAKE_ML_PROVIDER_FILE
push_artefact $SQUEEZENET_MODEL_ACTOR $SQUEEZENET_MODEL_ACTOR_FILE

show_images $LOCAL_REG_SERVER




# echo -e "pulling httpserver from remote"
# wash pull wasmcloud.azurecr.io/httpserver:0.19.1

# echo -e "pulling kvredis from remote"
# wash pull wasmcloud.azurecr.io/kvredis:0.22.0

# echo -e "pushing provider to local registry"
# wash reg push $HTTPSERVER_REF $HTTP_PROVIDER_FILE --insecure

# curl -X GET http://localhost:5000/v2/_catalog







