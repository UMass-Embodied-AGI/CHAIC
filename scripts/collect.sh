port=10001
pkill -f -9 "port $port"

python detection_pipeline/collect.py \
--save_dir captured_imgs_helper \
--train_data_prefix dataset/train_dataset \
--test_data_prefix dataset/test_dataset \
--tasks outdoor_shopping \
--train_scene 2a \
--test_scene 2a \
--train_layout 0