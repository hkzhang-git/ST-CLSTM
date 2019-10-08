function rawdata_2_video(start_id, end_id, fps)
% preprocess the raw video to video samples for depth estimation
load 'test_frame_list';
raw_data_dir = '/home/h0/hzhang/NYUD_v2_part';
video_sample_save_dir = '/home/h0/hzhang/Preprocessed_NYUD_v2/train';
test_video_sample_save_dir = '/home/h0/hzhang/Preprocessed_NYUD_v2/test';
[subset_list, subset_count] = file_list_get(raw_data_dir);
disp(char("preprocess " + num2str(start_id) + " to " + num2str(end_id) + " subsets, there are " + num2str(subset_count-2) + " subsets"))

for subset = subset_list(start_id:end_id)
    subset = subset{1};
    if subset ~= "." && subset ~= ".."
        subset_dir = char(raw_data_dir + "/" + subset);
        [sample_list, sample_count] = file_list_get(subset_dir);      
        
        for sample = sample_list
            sample = sample{1};            
            if sample ~= "." && sample ~= ".."                
                flag = test_sample_assert(test_frame_list, sample);
                if flag == 1
                    sample_dir = char(raw_data_dir + "/" + subset + "/" + sample);
                    sample_save_dir = char(test_video_sample_save_dir + "/" + sample);
                    rgb_save_dir = char(sample_save_dir + "/rgb");
                    depth_save_dir = char(sample_save_dir + "/depth");           
                    if ~exist(rgb_save_dir)
                        mkdir(rgb_save_dir)
                    end
                    if ~exist(depth_save_dir)
                        mkdir(depth_save_dir)
                    end
                    
                    framefile = get_synched_frames(sample_dir);
                    frame_id = 0;
                    for ii = 1: numel(framefile)
                        try
                            depth_filename = framefile(ii).rawDepthFilename;
                            rgb_filename = framefile(ii).rawRgbFilename;
                        catch
                            disp("fail to get rawDepthFilename or rawRgbFilename!");
                            continue;
                        end
                        rgb_file = char(sample_dir + "/" + rgb_filename);
                        depth_file = char(sample_dir + "/" + depth_filename);
                        
                        rgb_file_info = strsplit(rgb_file, '/');
                        rgb_file_info = rgb_file_info(end);
                        rgb_file_info = char(rgb_file_info{1});
                        rgb_name = rgb_file_info(1:end-4);
                        
                        depth_file_info = strsplit(depth_file, '/');
                        depth_file_info = depth_file_info(end);
                        depth_file_info = char(depth_file_info{1});
                        depth_name = depth_file_info(1:end-4);                      
                        try
                            if isempty(strfind(depth_file, 'pgm')) || isempty(strfind(rgb_file, 'ppm'))
                                continue;
                            end
                        catch 
                            disp("determin depth and rgb format fail!");
                            continue;
                        end 
                        
                        try
                            rgb = imread(rgb_file);
                            depth = imread(depth_file);
                            depth = swapbytes(depth);
                        catch 
                            disp("read fail!")
                            continue;
                        end
                        
                        [depth_aligned, rgb_undistort] = project_depth_map(depth, rgb);
                        depth_dinoised = fill_depth_colorization(rgb_undistort, depth_aligned, 0.9);
                        depth_dinoised_out = uint16(round(depth_dinoised * 6000.));                              
                        imwrite(rgb_undistort, char(rgb_save_dir + "/" + "rgb_" + num2str(frame_id, '%05d') + "@" + rgb_name + ".jpg" ));
                        imwrite(depth_dinoised_out, char(depth_save_dir + "/" + "depth_" + num2str(frame_id, '%05d') + "@" + depth_name + ".png" )); 
                        frame_id = frame_id + 1;
                    end
                else
                    sample_dir = char(raw_data_dir + "/" + subset + "/" + sample);
                    sample_save_dir = char(video_sample_save_dir + "/" + sample);
                    rgb_save_dir = char(sample_save_dir + "/rgb");
                    depth_save_dir = char(sample_save_dir + "/depth");           
                    if ~exist(rgb_save_dir)
                        mkdir(rgb_save_dir)
                    end
                    if ~exist(depth_save_dir)
                        mkdir(depth_save_dir)
                    end
                    framefile = get_synched_frames(sample_dir);
                    frame_id = 0;
                    for ii = 1: fps : numel(framefile)
                        try
                            depth_filename = framefile(ii).rawDepthFilename;
                            rgb_filename = framefile(ii).rawRgbFilename;
                        catch
                            disp("fail to get rawDepthFilename or rawRgbFilename!");
                            continue;
                        end
                        rgb_file = char(sample_dir + "/" + rgb_filename);
                        depth_file = char(sample_dir + "/" + depth_filename);
                        try
                            if isempty(strfind(depth_file, 'pgm')) || isempty(strfind(rgb_file, 'ppm'))
                                continue;
                            end
                        catch 
                            disp("determin depth and rgb format fail!");
                            continue;
                        end 
                        
                        try
                            rgb = imread(rgb_file);
                            depth = imread(depth_file);
                            depth = swapbytes(depth);
                        catch 
                            disp("read fail!")
                            continue;
                        end                       
                        
                        [depth_aligned, rgb_undistort] = project_depth_map(depth, rgb);
                        depth_dinoised = fill_depth_colorization(rgb_undistort, depth_aligned, 0.9);
                        depth_dinoised_out = uint16(round(depth_dinoised * 6000.));                
                        imwrite(rgb_undistort, char(rgb_save_dir + "/" + "rgb_" + num2str(frame_id, '%05d') + ".jpg" ));
                        imwrite(depth_dinoised_out, char(depth_save_dir + "/" + "depth_" + num2str(frame_id, '%05d') + ".png" )); 
                        frame_id = frame_id + 1;
                    end

                end
                disp(char("sample: " + sample + " is done"))
            end                                            
        end            
        disp(char("subset: " + subset + " is done"))
    end      
end

end
    




