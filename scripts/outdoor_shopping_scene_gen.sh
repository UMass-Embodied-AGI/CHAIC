port=8888
pkill -f -9 "port $port"

python scenes/outdoor_scene/shopping/scene_gen.py --type train
pkill -f -9 "port $port"

python scenes/outdoor_scene/shopping/scene_gen.py --type test