import argparse
import subprocess
from utils import *
from glob import glob
import numpy as np


parser = argparse.ArgumentParser(description='raw_nyu_v2_build')
parser.add_argument('--source_dir', type=str, default='/home/hkzhang/Documents/sdb_a/raw_data/raw_nyu_v2_300k')
parser.add_argument('--save_dir', type=str, default='/home/hkzhang/Documents/sdb_a/raw_data/nyu_v2_r')
parser.add_argument('--train_info_dir', type=str, default='./json/train_scenes.json')
parser.add_argument('--test_info_dir', type=str, default='./json/test_frames.json')
parser.add_argument('--dataset', type=str, default='raw_nyu_v2_250k')
parser.add_argument('--test_loc', type=str, default='end')
parser.add_argument('--fps', type=int, default=30)
parser.add_argument('--fl', type=int, default=180)
args = parser.parse_args()

test_frame_list = json_loader(args.test_info_dir)
save_dir = os.path.join(args.save_dir, args.dataset)

test_target_save_dir = save_dir + '/test_fps{}_fl{}_{}/'.format(args.fps, args.fl, args.test_loc)

for scene_info in test_frame_list:
    scene_id = scene_info['id']
    scene_name = scene_info['scene_name']
    frame_name = scene_info['frame_name']
    scene_dir = os.path.join(args.source_dir, 'test', scene_name)
    depth_im_dir = glob(scene_dir + '/depth/depth_*{}.png'.format(frame_name))[0]
    depth_id = depth_im_dir.split('/')[-1].split('@')[0].split('_')[1]
    depth_frame_num = len(glob(scene_dir + '/depth/depth_*.png'))

    indices, index_in_test_clips = get_indices(depth_frame_num, int(depth_id), args.fl, interval = 30 // args.fps, test_loc=args.test_loc)

    if max(indices)<depth_frame_num:
        depth_scene_save_dir = test_target_save_dir + '%04d' % scene_id + '/depth/'
        rgb_scene_save_dir = test_target_save_dir + '%04d' % scene_id + '/rgb/'
        make_if_not_exits(depth_scene_save_dir)
        make_if_not_exits(rgb_scene_save_dir)
        for save_index, index in enumerate(indices):
            depth_im_dir = glob(scene_dir + '/depth/depth_' + '%05d' % index + '@*.png')[0]
            rgb_im_dir = glob(scene_dir + '/rgb/rgb_' + '%05d' % index + '@*.jpg')[0]

            depth_im_save_dir = depth_scene_save_dir + 'depth_' + '%05d' % save_index + '.png'
            rgb_im_save_dir = rgb_scene_save_dir + 'rgb_' + '%05d' % save_index + '.jpg'

            depth_save_command = 'cp {} {}'.format(depth_im_dir, depth_im_save_dir)
            rgb_save_command = 'cp {} {}'.format(rgb_im_dir, rgb_im_save_dir)

            subprocess.call(depth_save_command, shell=True)
            subprocess.call(rgb_save_command, shell=True)

        with open(depth_scene_save_dir + 'frame_index.txt', 'w') as f:
            f.write(str(index_in_test_clips))

