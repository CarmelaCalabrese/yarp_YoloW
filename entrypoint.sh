#!/bin/bash
# Start YARP server (optional, if needed)
yarp server &

# Wait for YARP server to be ready
sleep 2

# Start the detection node
python /app/yarp_yolo_world.py
