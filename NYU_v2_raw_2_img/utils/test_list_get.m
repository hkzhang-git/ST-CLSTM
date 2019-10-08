% load 'nyu_depth_v2_labeled.mat'
%load 'splits.mat'

test_frame_list = {};
list_len = size(testNdxs, 1)

for i = 1: list_len
    id = testNdxs(i);
    frame_info = strsplit(rawDepthFilenames{id}, '/');
    test_frame_list{i} = frame_info{1};    
end

save test_frame_list

