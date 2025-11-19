#!/bin/bash

# Allow GUI applications from Docker
xhost + > /dev/null

docker run --rm \
  --privileged \
  --device /dev/video0:/dev/video0 \
  --user $(id -u):$(id -g) \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /home/pi:/home/pi \
  signature-gui
