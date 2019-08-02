# ST-CLSTM
Code of paper: Exploiting temporal consistency for real-time video depth estimation (ICCV 2019)

Some video results can be found at https://youtu.be/B705k8nunLU
The code and pretrained models will be released soon.

Requirements:
Pytorch>=0.4.0
Python=3.6
matlab (for converting raw data, dump and pgm files to jpg and png files)

Data preprocess:

Data preprocess consists of three steps, including 
1) In raw data, the RGB and corresponding depth data are saved as .dump and .pgm file, respectively. The first step is 
converting raw RGB and Depth data into .jpg and .png files. In this step, synchronization, alignment and padding 
operations are needed. The matlab code "NYU_v2_raw_2_img" is provided to handle the tasks in this step.

cd NYU_v2_raw_2_img
matlab main_x_x.m 

2) Splitting training samples and extract test samples. The python code "raw_nyu_v2_build" is used to finish this task.

cd raw_nyu_v2_build
python main_clips.py --test_loc 'end' --fl 5

3) Creating data_list for dataloader:

cd CLSTM_Depth_Estimation_v1/data/
python create_list_nyu_v2_3D.py

Training:
cd CLSTM_Depth_Estimation_v1
python main_raw_CLSTM.py


Evaluation:
cd CLSTM_Depth_Estimation_v1/prediction/
python prediction_CLSTM_main.py

Demo:
cd CLSTM_Depth_Estimation_v1/
python demo_main.py

Datasets:
NYU-V2 (https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) and Kitti were used.

Example data folder structure:

data_root
|- raw_nyu_v2_250k
|  |- train
|  |  |- basement_0001a
|  |  |  |- rgb
|  |  |  |  |- rgb_00000.jpg
|  |  |  |  |_ ...
|  |  |  |- depth
|  |  |  |  |- depth_00000.png
|  |  |  |  |_ ...
|  |  |- basement_0001b
|  |  |  |- rgb
|  |  |  |  |- rgb_00000.jpg
|  |  |  |  |_ ...
|  |  |  |- depth
|  |  |  |  |- depth_00000.png
|  |  |  |  |_ ...
|  |  |_ ...
|  |- test_fps_30_fl5_end
|  |  |- 0000
|  |  |  |- rgb
|  |  |  |  |- rgb_00000.jpg
|  |  |  |  |- rgb_00001.jpg
|  |  |  |  |- ...
|  |  |  |  |- rgb_00004.jpg
|  |  |  |- depth
|  |  |  |  |- depth_00000.png
|  |  |  |  |- depth_00001.png
|  |  |  |  |- ...
|  |  |  |  |- depth_00004.png
|  |  |- 0001
|  |  |  |- rgb
|  |  |  |  |- rgb_00000.jpg
|  |  |  |  |- rgb_00001.jpg
|  |  |  |  |- ...
|  |  |  |  |- rgb_00004.jpg
|  |  |  |- depth
|  |  |  |  |- depth_00000.png
|  |  |  |  |- depth_00001.png
|  |  |  |  |- ...
|  |  |  |  |- depth_00004.png
|  |  |- ...
|  |  |- 0653
|  |  |  |- rgb
|  |  |  |  |- rgb_00000.jpg
|  |  |  |  |- rgb_00001.jpg
|  |  |  |  |- ...
|  |  |  |  |- rgb_00004.jpg
|  |  |  |- depth
|  |  |  |  |- depth_00000.png
|  |  |  |  |- depth_00001.png
|  |  |  |  |- ...
|  |  |  |  |- depth_00004.png
|  |- test_fps_30_fl3_end
|  |  |- 0000
|  |  |  |- rgb
|  |  |  |  |- rgb_00000.jpg
|  |  |  |  |- rgb_00001.jpg
|  |  |  |  |- rgb_00002.jpg
|  |  |  |- depth
|  |  |  |  |- depth_00000.png
|  |  |  |  |- depth_00001.png
|  |  |  |  |- depth_00002.png
|  |  |- 0001
|  |  |  |- rgb
|  |  |  |  |- rgb_00000.jpg
|  |  |  |  |- rgb_00001.jpg
|  |  |  |  |- rgb_00002.jpg
|  |  |  |- depth
|  |  |  |  |- depth_00000.png
|  |  |  |  |- depth_00001.png
|  |  |  |  |- depth_00002.png
|  |  |- ...






