import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data_list', nargs= '+', type=str, required=True)
parser.add_argument('--output_dir', type=str, default = 'behavior_data_all')

args = parser.parse_args()
data_list = args.data_list
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'train', 'video'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'val', 'video'), exist_ok=True)

destination_train = os.path.join(output_dir, 'train', 'video')
destination_val = os.path.join(output_dir, 'val', 'video')
total_train_count = 0
total_val_count = 0
total_train_list = []
total_val_list = []

for data in data_list:
    # find data root
    data = data.strip('/')
    data_root = '/'.join(data.split("/")[: -1])
    train_data_list = os.path.join(data, 'train/train_list.txt')
    with open(train_data_list, 'r') as f:
        train_data = f.readlines()

    for i in range(len(train_data)):
        source = train_data[i].strip().split()
        destination = os.path.join(destination_train, str(total_train_count) + '.mp4')
        os.system('cp ' + os.path.join(data_root, source[0]) + ' ' + destination)
        total_train_list.append(os.path.join(destination_train, str(total_train_count) + '.mp4') + ' ' + source[1])
        total_train_count += 1
    
    val_data_list = os.path.join(data, 'val/val_list.txt')
    with open(val_data_list, 'r') as f:
        val_data = f.readlines()
    for i in range(len(val_data)):
        source = val_data[i].strip().split()
        destination = os.path.join(destination_val, str(total_val_count) + '.mp4')
        os.system('cp ' + os.path.join(data_root, source[0]) + ' ' + destination)
        total_val_list.append(os.path.join(destination_val, str(total_val_count) + '.mp4') + ' ' + source[1])
        total_val_count += 1

with open(os.path.join(args.output_dir, 'train/train_list.txt'), 'w') as f:
    for item in total_train_list:
        f.write("%s\n" % item)

with open(os.path.join(args.output_dir, 'val/val_list.txt'), 'w') as f:
    for item in total_val_list:
        f.write("%s\n" % item)