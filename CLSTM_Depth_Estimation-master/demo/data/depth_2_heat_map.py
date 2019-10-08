import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from glob import glob


def tensor_2_img(target):
    cmap = plt.get_cmap('jet')
    target_rgba = cmap(target/110)
    target_rgb = np.delete(target_rgba, 3, 2)
    # im = Image.fromarray(np.array(target_rgb, np.uint8))
    return target_rgb

if __name__ == '__main__':
    depth_dir = './depth/'
    save_dir = './heat_depth/'
    depth_list = glob(depth_dir + 'depth_*.png')
    depth_list.sort()
    for id, item in enumerate(depth_list):
        depth_im = cv2.imread(item)[:,:,0]
        heat_map = tensor_2_img(depth_im)
        matplotlib.image.imsave(save_dir + '{}.jpg'.format(id), heat_map)