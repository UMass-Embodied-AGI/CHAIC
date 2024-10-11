port=10001
pkill -f -9 "port $port"

python detection_pipeline/transform_into_coco.py \
--dataset_dir /home/ubuntu/data/tdw_helper_perception_dataset \
--train_image_dir /home/ubuntu/data/captured_imgs_helper/train \
--test_image_dir /home/ubuntu/data/captured_imgs_helper/test \
--name_map_path dataset/name_map.json