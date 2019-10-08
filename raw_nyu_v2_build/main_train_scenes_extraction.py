import argparse
import subprocess
from utils import *


parser = argparse.ArgumentParser(description='raw_nyu_v2_build')
parser.add_argument('--source_dir', type=str, default='/home/hkzhang/Documents/sdb_a/raw_data/raw_nyu_v2_600k')
parser.add_argument('--save_dir', type=str, default='/home/hkzhang/Documents/sdb_a/raw_data')
parser.add_argument('--train_info_dir', type=str, default='./json/train_scenes.json')
parser.add_argument('--dataset', type=str, default='raw_nyu_v2_120k')
parser.add_argument('--fps_inv', type=int, default=5)
args = parser.parse_args()

train_scene_list = json_loader(args.train_info_dir)
train_scene_array = [scene_info['scene_name'] for scene_info in train_scene_list]
train_source_dir = args.source_dir + '/train'
source_train_list = os.listdir(train_source_dir)

save_dir = os.path.join(args.save_dir, args.dataset)
make_if_not_exits(save_dir)

target_save_dir = save_dir + '/train/'
make_if_not_exits(target_save_dir)
for scene in source_train_list:
    if scene in train_scene_array:
        source_scene_dir = os.path.join(train_source_dir, scene)
        target_scene_dir = target_save_dir + scene
        command = 'cp -r {} {}'.format(source_scene_dir, target_scene_dir)
        subprocess.call(command, shell=True)
        print('done: {}'.format(command))

# for scene in train_scene_array:
#     if scene not in source_train_list:
#         print(scene)

