#!/bin/bash

xhost +local:docker

docker run --rm -it \
  --privileged \
  --device=/dev/video0 \
  --device=/dev/vchiq \
  --device=/dev/v4l \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /home/pi:/home/pi \
  signature_gui
