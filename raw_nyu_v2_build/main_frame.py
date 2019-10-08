import argparse
import subprocess
from utils import *
from glob import glob
import numpy as np


parser = argparse.ArgumentParser(description='raw_nyu_v2_build')
parser.add_argument('--source_dir', type=str, default='/home/hkzhang/Documents/sdb_a/raw_data/raw_nyu_v2_300k')
parser.add_argument('--save_dir', type=str, default='/home/hkzhang/Documents/sdb_a/raw_data')
parser.add_argument('--train_info_dir', type=str, default='./json/train_scenes.json')
parser.add_argument('--test_info_dir', type=str, default='./json/test_frames.json')
parser.add_argument('--dataset', type=str, default='raw_nyu_v2_50k')
parser.add_argument('--fps_inv', type=int, default=5)
args = parser.parse_args()

train_scene_list = json_loader(args.train_info_dir)
train_scene_array = [scene_info['scene_name'] for scene_info in train_scene_list]
train_source_dir = args.source_dir + '/train'
source_train_list = os.listdir(train_source_dir)
test_frame_list = json_loader(args.test_info_dir)

save_dir = os.path.join(args.save_dir, args.dataset)
make_if_not_exits(save_dir)

train_target_save_dir = save_dir + '/train/'
make_if_not_exits(train_target_save_dir)

# for scene in source_train_list:
#     if scene in train_scene_array:
#         source_scene_dir = os.path.join(train_source_dir, scene)
#         rgb_target_scene_dir = train_target_save_dir + scene + '/rgb/'
#         depth_target_scene_dir = train_target_save_dir + scene + '/depth/'
#
#         make_if_not_exits(rgb_target_scene_dir)
#         make_if_not_exits(depth_target_scene_dir)
#
#         rgb_frame_list = glob(source_scene_dir + '/rgb/rgb_*.jpg')
#         rgb_frame_list.sort()
#         depth_frame_list = glob(source_scene_dir + '/depth/depth_*png')
#         depth_frame_list.sort()
#
#         frame_len = len(depth_frame_list)
#         indices = list(range(0, frame_len, args.fps_inv))
#         for id, index in enumerate(indices):
#             rgb_dir = rgb_frame_list[index]
#             depth_dir = depth_frame_list[index]
#             rgb_id = rgb_dir.split('/')[-1].split('.')[0][-5:]
#             depth_id = depth_dir.split('/')[-1].split('.')[0][-5:]
#
#             assert rgb_id == depth_id
#             depth_save_command = 'cp {} {}'.format(depth_dir, depth_target_scene_dir + 'depth_' + '%05d' % id + '.png')
#             rgb_save_command = 'cp {} {}'.format(rgb_dir, rgb_target_scene_dir + 'rgb_' + '%05d' % id + '.jpg')
#             subprocess.call(depth_save_command, shell=True)
#             subprocess.call(rgb_save_command, shell=True)


test_target_save_dir = save_dir + '/test/'
make_if_not_exits(test_target_save_dir)
for scene_info in test_frame_list:
    scene_id = scene_info['id']
    scene_name = scene_info['scene_name']
    frame_name = scene_info['frame_name']
    scene_dir = os.path.join(args.source_dir, 'test', scene_name)
    depth_im_dir = glob(scene_dir + '/depth/depth_*{}.png'.format(frame_name))[0]
    depth_im = np.array(Image.open(depth_im_dir))

    depth_id = depth_im_dir.split('/')[-1].split('@')[0].split('_')[1]
    rgb_im_dir = glob(scene_dir + '/rgb/rgb_{}@*.jpg'.format(depth_id))[0]

    depth_frame_save_dir = test_target_save_dir + 'depth_' + '%03d' % scene_id + '.png'
    rgb_frame_save_dir = test_target_save_dir + 'rgb_' + '%03d' % scene_id + '.jpg'
    depth_save_command = 'cp {} {}'.format(depth_im_dir, depth_frame_save_dir)
    rgb_save_command = 'cp {} {}'.format(rgb_im_dir, rgb_frame_save_dir)

    subprocess.call(depth_save_command, shell=True)
    subprocess.call(rgb_save_command, shell=True)



