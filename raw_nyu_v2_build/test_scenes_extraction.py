from utils import *


test_scene_list = json_loader('./json/test_frames.json')
test_scene_array = [scene_info['scene_name'] for scene_info in test_scene_list]

source_test_list = os.listdir('/home/hkzhang/Documents/sdb_a/raw_data/raw_nyu_v2_600k/test')
for scene in test_scene_array:
    if scene not in source_test_list:
        print(scene)