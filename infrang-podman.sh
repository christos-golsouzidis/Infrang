#!/bin/bash
if podman ps | grep "infrang-api" > /dev/null; then
    echo "The container 'infrang-api' is already running."
    exit 1
fi
if ! podman images | grep "infrang-img" > /dev/null; then
    echo "Creating the image 'infrang-img'..."
    podman build -t infrang-img .
fi
echo "Image: infrang-img"
if ! podman volume ls | grep "infrang-data" > /dev/null; then
    echo "Creating the volume 'infrang-data'..."
    podman volume create infrang-data
fi
echo "Volume: infrang-data"
if podman ps -a | grep "infrang-api" > /dev/null; then
    podman start infrang-api
else
    if [ -z "$1" ]; then
        echo "Usage: $0 <mount-path>"
        echo "Error: No mount path provided." >&2
        exit 1
    fi
    podman run -d --name infrang-api \
    -p 7456:7456 \
    --env-file .env \
    -v infrang-data:/data/:Z \
    -v "$1:/base/:Z" \
    infrang-img
fi