import os
import json
import argparse
import numpy as np
from glob import glob


parser = argparse.ArgumentParser(description='raw_nyu_v2')
parser.add_argument('--dataset', type=str, default='raw_nyu_v2_250k')
parser.add_argument('--test_loc', type=str, default='end')
parser.add_argument('--fps', type=int, default=30)
parser.add_argument('--fl', type=int, default=20)
parser.add_argument('--overlap', type=int, default=0)
parser.add_argument('--list_save_dir', type=str, default='./data_list')
parser.add_argument('--source_dir', type=str, default='/home/hkzhang/Documents/sdb_a/raw_data/nyu_v2_r/')
args = parser.parse_args()

args.jpg_png_save_dir = args.source_dir + args.dataset


def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def video_split(frame_len, frame_train, interval, overlap):
    sample_interval = frame_train - overlap
    indices = []
    for start in range(interval):
        index_list = list(range(start, frame_len - frame_train * interval + 1, sample_interval))
        [indices.append(list(range(num, num+frame_train*interval, interval))) for num in index_list]
        indices.append(list(range(frame_len - frame_train * interval - start, frame_len - start, interval)))

    return indices


def create_dict(dataset, list_save_dir, jpg_png_save_dir):
    train_dir = os.path.join(jpg_png_save_dir, 'train')
    test_dir = os.path.join(jpg_png_save_dir, 'test_fps{}_fl{}_{}'.format(args.fps, args.fl,args.test_loc))
    interval = 30 // args.fps

    train_dict=[]
    subset_list = os.listdir(train_dir)
    for subset in subset_list:
        subset_source_dir = os.path.join(train_dir, subset)
        rgb_list = glob(subset_source_dir + '/rgb/rgb_*.jpg')
        depth_list = glob(subset_source_dir + '/depth/depth_*.png')
        rgb_list.sort()
        depth_list.sort()

        rgb_list_new = ['/'.join(rgb_info.split('/')[-5:]) for rgb_info in rgb_list]
        depth_list_new = ['/'.join(depth_info.split('/')[-5:]) for depth_info in depth_list]

        indices = video_split(len(depth_list), args.fl, interval, args.overlap)

        for index in indices:
            rgb_index = []
            depth_index = []
            [rgb_index.append(rgb_list_new[id]) for id in index]
            [depth_index.append(depth_list_new[id]) for id in index]

            train_info = {
                'rgb_index': rgb_index,
                'depth_index': depth_index,
                'scene_name': subset,
            }
            train_dict.append(train_info)

    test_dict = []
    subset_list = os.listdir(test_dir)
    for subset in subset_list:
        subset_source_dir = os.path.join(test_dir, subset)
        rgb_list = glob(subset_source_dir + '/rgb/rgb_*.jpg')
        depth_list = glob(subset_source_dir + '/depth/depth_*.png')
        rgb_list.sort()
        depth_list.sort()

        rgb_list_new = ['/'.join(rgb_info.split('/')[-5:]) for rgb_info in rgb_list]
        depth_list_new = ['/'.join(depth_info.split('/')[-5:]) for depth_info in depth_list]

        test_index = int(open(subset_source_dir + '/depth/frame_index.txt').read())

        test_info = {
            'rgb_index': rgb_list_new,
            'depth_index': depth_list_new,
            'scene_name': subset,
            'test_index': test_index
        }
        test_dict.append(test_info)

    list_save_dir = os.path.join(list_save_dir, dataset)
    make_if_not_exist(list_save_dir)
    train_info_save = list_save_dir + '/{}_fps{}_fl{}_op{}_{}_train.json'.format(dataset, args.fps, args.fl, args.overlap, args.test_loc)
    test_info_save = list_save_dir + '/{}_fps{}_fl{}_op{}_{}_test.json'.format(dataset, args.fps, args.fl, args.overlap, args.test_loc)

    with open(train_info_save, 'w') as dst_file:
        json.dump(train_dict, dst_file)
    with open(test_info_save, 'w') as dst_file:
        json.dump(test_dict, dst_file)

if __name__ == '__main__':

    create_dict(args.dataset, args.list_save_dir, args.jpg_png_save_dir)