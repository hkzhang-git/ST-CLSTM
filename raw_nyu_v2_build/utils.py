import os
import json
import math
import numpy as np
from PIL import Image


def make_if_not_exits(path):
    if not os.path.exists(path):
        os.makedirs(path)


def json_loader(path):
    with open(path, 'r') as json_file:
        return json.load(json_file)


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def get_indices(len, test_index, fl, interval, test_loc):
    if test_loc == 'end':
        index_start = test_index - (fl-1)*interval
        index_end = test_index+interval
    elif test_loc == 'mid':
        index_start = test_index - (math.ceil(fl/2) ) * interval
        index_end = test_index + (math.ceil(fl/2)-1) * interval

    if index_start < 0:
        move_r = math.ceil((0 - index_start) / interval) * interval
        index_start += move_r
        index_end += move_r
    # elif (index_end - interval) > (len - 1):
    #     move_l = math.ceil((((index_end - interval) - (len-1)) / interval)) * interval
    #     index_start -= move_l
    #     index_end -= move_l

    elif (index_end - interval) > (len - 1):
        move_l = math.ceil((((index_end - interval) - (len-1)) / interval)) * interval
        index_start -= move_l
        index_end -= move_l

    indices = np.array(range(index_start, index_end, interval))
    index_in_clips = np.where(indices==test_index)[0][0]

    return indices, index_in_clips