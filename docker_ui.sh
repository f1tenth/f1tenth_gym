xhost local:$USER
docker run --gpus all -it --name=f1tenth_gym_container --rm -p 5557:5557 -p 5558:5558 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix f1tenth_gym
