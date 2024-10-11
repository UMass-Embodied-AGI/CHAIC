import os
import json
import sys
import numpy as np
import argparse

def opponent_in_view_rate(start_frame: int, end_frame: int, logs) -> float:
    total = 0
    in_view = 0
    for i in range(len(logs)):
        if logs[i]['frame'] > start_frame and logs[i]['frame'] < end_frame:
            total += 1
            if logs[i]['oppo_in_view']:
                in_view += 1
    return in_view / total

def video_clip_from_raw_data(folder = 'results/normal_task_follow_helper/vision_1', processed_folder = 'results_action_clip'):
  num_clip = 0
  walking_clip = 0
  # rm 
  import shutil
  shutil.rmtree(processed_folder, ignore_errors=True)
  os.makedirs(processed_folder, exist_ok=True)
  for folder_name in os.listdir(folder):
    if folder_name.isdigit():
        action_log = os.path.join(folder, folder_name, "info", "0.json")
        watch_log = os.path.join(folder, folder_name, "info", "1.json")
        img_folder = os.path.join(folder, folder_name, "Images", "1")
        with open(action_log, 'r') as f:
            action_log = json.load(f)
        with open(watch_log, 'r') as f:
            watch_log = json.load(f)
        keep_walking_from = 0
        for i in range(len(action_log) - 1):
            act = eval(action_log[i]['action'])
            # 0: ongoing, 1: success, 2: fail
            assert action_log[i + 1]['last_action_status'] != 0
            if act['type'] > 2:
                act_len = action_log[i+1]['frame'] - action_log[i]['frame']
                os.makedirs(os.path.join(processed_folder, str(num_clip)), exist_ok=True)
                os.makedirs(os.path.join(processed_folder, str(num_clip), 'Images'), exist_ok=True)
                '''
                    class criteria
                    id = (act['type'] - 3) * 2 + (success / fail)
                    walking: id 7
                '''
                meta_data = {
                    "action": act,
                    "length": act_len,
                    "begin": action_log[i]['frame'],
                    "end": action_log[i+1]['frame'],
                    "origin": img_folder,
                    "success": action_log[i + 1]['last_action_status'] - 1,
                    "class": (act['type'] - 3) * 2 + action_log[i + 1]['last_action_status'] - 1,
                }
                if opponent_in_view_rate(action_log[i]['frame'], action_log[i+1]['frame'], watch_log) < 0.2:
                    print("Opponent not in view")
                    continue
                with open(os.path.join(processed_folder, str(num_clip), 'info.json'), 'w') as f:
                    json.dump(meta_data, f)
                for j in range(action_log[i]['frame'], action_log[i+1]['frame']):
                    img_path = os.path.join(img_folder, '{:04d}'.format(j) + ".png")
                    # copy
                    import shutil
                    shutil.copy(img_path, os.path.join(processed_folder, str(num_clip), 'Images', '{:04d}'.format(j - action_log[i]['frame']) + ".png"))
                    # ffmpeg -framerate 30 -i demo_$1/images_a/img_%04d.jpg -c:v libx264 -r 30 -pix_fmt yuv420p $1.mp4
                os.system(f"ffmpeg -framerate 30 -i {os.path.join(processed_folder, str(num_clip), 'Images', '%04d.png')} -c:v libx264 -r 30 -pix_fmt yuv420p {os.path.join(processed_folder, str(num_clip), 'video.mp4')}")
                num_clip += 1
                keep_walking_from = action_log[i + 1]['frame']
            else:
                #0: move forward
                #1: turn left
                #2: turn right
                if walking_clip > num_clip * 0.4:
                    continue
                threshold = np.random.randint(50, 100)
                if action_log[i + 1]['frame'] - keep_walking_from > threshold:
                    act_len = threshold
                    os.makedirs(os.path.join(processed_folder, str(num_clip)), exist_ok=True)
                    os.makedirs(os.path.join(processed_folder, str(num_clip), 'Images'), exist_ok=True)
                    meta_data = {
                        "action": act,
                        "length": threshold,
                        "begin": action_log[i + 1]['frame'] - threshold,
                        "end": action_log[i + 1]['frame'],
                        "origin": img_folder,
                        "success": 1,
                        "class": 7,
                    }
                    if opponent_in_view_rate(action_log[i + 1]['frame'] - threshold, action_log[i + 1]['frame'], watch_log) < 0.2:
                        print("Opponent not in view")
                        continue
                    with open(os.path.join(processed_folder, str(num_clip), 'info.json'), 'w') as f:
                        json.dump(meta_data, f)
                    for j in range(action_log[i + 1]['frame'] - threshold, action_log[i + 1]['frame']):
                        img_path = os.path.join(img_folder, '{:04d}'.format(j) + ".png")
                        # copy
                        import shutil
                        shutil.copy(img_path, os.path.join(processed_folder, str(num_clip), 'Images', '{:04d}'.format(j - (action_log[i + 1]['frame'] - threshold)) + ".png"))
                        # ffmpeg -framerate 30 -i demo_$1/images_a/img_%04d.jpg -c:v libx264 -r 30 -pix_fmt yuv420p $1.mp4
                    os.system(f"ffmpeg -framerate 30 -i {os.path.join(processed_folder, str(num_clip), 'Images', '%04d.png')} -c:v libx264 -r 30 -pix_fmt yuv420p {os.path.join(processed_folder, str(num_clip), 'video.mp4')}")
                    num_clip += 1
                    walking_clip += 1
                    keep_walking_from = action_log[i + 1]['frame']

def train_dataset_from_video_clip(processed_folder = 'results_action_clip', dataset = 'action_dataset', train_ratio = 0.8):
    os.makedirs(dataset, exist_ok=True)
    train_dataset = os.path.join(dataset, "train")
    os.makedirs(train_dataset, exist_ok=True)
    val_dataset = os.path.join(dataset, "val")
    os.makedirs(val_dataset, exist_ok=True)
    os.makedirs(os.path.join(train_dataset, "video"), exist_ok=True)
    os.makedirs(os.path.join(val_dataset, "video"), exist_ok=True)
    all_data_path = os.listdir(processed_folder)
    all_data_path = [os.path.join(processed_folder, i) for i in all_data_path]
    np.random.shuffle(all_data_path)
    train_data_path = all_data_path[:int(len(all_data_path) * train_ratio)]
    train_data_path.sort()
    val_data_path = all_data_path[int(len(all_data_path) * train_ratio):]
    val_data_path.sort()
    train_name_and_class = []
    val_name_and_class = []
    for i in train_data_path:
        meta_data_path = os.path.join(i, 'info.json')
        meta_data = json.load(open(meta_data_path, 'r'))
        video_class = meta_data['class']
        video_path = os.path.join(i, 'video.mp4')
        move_to_video_path = os.path.join(train_dataset, "video", str(i).split('/')[-1] + ".mp4")
        # cp video_path move_to_video_path
        import shutil
        shutil.copy(video_path, move_to_video_path)
        train_name_and_class.append((move_to_video_path, video_class))
    train_list_file = os.path.join(train_dataset, "train_list.txt")
    with open(train_list_file, 'w') as f:
        for i in train_name_and_class:
            f.write(i[0] + " " + str(i[1]) + "\n")
    for i in val_data_path:
        meta_data_path = os.path.join(i, 'info.json')
        meta_data = json.load(open(meta_data_path, 'r'))
        video_class = meta_data['class']
        video_path = os.path.join(i, 'video.mp4')
        move_to_video_path = os.path.join(val_dataset, "video", str(i).split('/')[-1] + ".mp4")
        # cp video_path move_to_video_path
        import shutil
        shutil.copy(video_path, move_to_video_path)
        val_name_and_class.append((move_to_video_path, video_class))
    val_list_file = os.path.join(val_dataset, "val_list.txt")
    with open(val_list_file, 'w') as f:
        for i in val_name_and_class:
            f.write(i[0] + " " + str(i[1]) + "\n")
    total_train_number = len(train_name_and_class)
    total_val_number = len(val_name_and_class)
    print("Total train number: ", total_train_number)
    print("Total val number: ", total_val_number)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_folder", type=str, default='results/normal_task_follow_helper/train_data_collection_revised')
    parser.add_argument("--processed_folder", type=str, default='results_action_clip')
    parser.add_argument("--destination_dataset", type=str, default='results_action_dataset')
    parser.add_argument("--train_ratio", type=float, default=0.8)
    args = parser.parse_args()
    video_clip_from_raw_data(folder = args.raw_data_folder, processed_folder = args.processed_folder)
    train_dataset_from_video_clip(processed_folder=args.processed_folder, dataset=args.destination_dataset, train_ratio=args.train_ratio)