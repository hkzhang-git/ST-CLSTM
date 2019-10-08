import os
import json
import argparse
from glob import glob


parser = argparse.ArgumentParser(description='raw_nyu_v2')
parser.add_argument('--dataset',type=str, default='raw_nyu_v2_250k')
parser.add_argument('--list_save_dir', type=str, default='./data_list')
parser.add_argument('--source_dir', type=str, default='/home/hkzhang/Documents/sdb_a/raw_data/nyu_v2_r/')
args = parser.parse_args()

args.jpg_png_save_dir = args.source_dir + args.dataset


def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_dict(dataset, list_save_dir, jpg_png_save_dir):
    train_dir = os.path.join(jpg_png_save_dir, 'train')
    test_dir = os.path.join(jpg_png_save_dir, 'test')

    train_dict=[]
    subset_list = os.listdir(train_dir)
    for subset in subset_list:
        subset_source_dir = os.path.join(train_dir, subset)
        rgb_list = glob(subset_source_dir + '/rgb/rgb_*.jpg')
        depth_list = glob(subset_source_dir + '/depth/depth_*.png')
        rgb_list.sort()
        depth_list.sort()
        for i, (rgb_dir, depth_dir) in enumerate(zip(rgb_list, depth_list)):
            rgb_id = rgb_dir.split('/')[-1].split('.')[0][-5:]
            depth_id = depth_dir.split('/')[-1].split('.')[0][-5:]
            assert rgb_id == depth_id

            im_id = 'train' + '_%03d' % i
            train_info = {
                'data_path': '/'.join(rgb_dir.split('/')[-5:]),
                'gt_path': '/'.join(depth_dir.split('/')[-5:]),
                'im_id':im_id
            }
            train_dict.append(train_info)

    test_dict = []
    rgb_list = glob(test_dir + '/rgb_*.jpg')
    depth_list = glob(test_dir + '/depth_*.png')
    rgb_list.sort()
    depth_list.sort()
    for i, (rgb_dir, depth_dir) in enumerate(zip(rgb_list, depth_list)):
        rgb_id = rgb_dir.split('/')[-1].split('.')[0][-3:]
        depth_id = depth_dir.split('/')[-1].split('.')[0][-3:]
        assert rgb_id == depth_id

        im_id = 'test' + '_%03d' % i
        test_info = {
            'data_path': '/'.join(rgb_dir.split('/')[-3:]),
            'gt_path': '/'.join(depth_dir.split('/')[-3:]),
            'im_id': im_id
        }
        test_dict.append(test_info)

    list_save_dir = os.path.join(list_save_dir, dataset)
    make_if_not_exist(list_save_dir)
    train_info_save = list_save_dir + '/{}_train.json'.format(dataset)
    test_info_save = list_save_dir + '/{}_test.json'.format(dataset)

    with open(train_info_save, 'w') as dst_file:
        json.dump(train_dict, dst_file)
    with open(test_info_save, 'w') as dst_file:
        json.dump(test_dict, dst_file)

if __name__ == '__main__':

    create_dict(args.dataset, args.list_save_dir, args.jpg_png_save_dir)