%Para @in_dir: the root dir of dataset, e.g.'./data/nyu_depth_v2'
%Para @data_name: the sub-dataset, e.g.'bedrooms_part1'
%Para @num_sub_dir: the ordinal number of dir under sub-dataset, e.g. 1
%Para @unzip_folder: unzip the folder or not, 0-not unzip, 1-unzip

function transfer_depth_new_2(in_dir, data_name, num_sub_dir, unzip_folder)
    %%Read test data
    test_data_dir = char(in_dir) + "/" + "test_annotation.json";
    test_data = jsondecode(fileread(test_data_dir));
    test_rgbs = [];
    test_depths = [];
    for i = 1 : 1 : 654
        test_rgb = test_data(i).raw_rgb_filenames;
        test_depth = test_data(i).raw_depth_filenames;
        test_rgbs = [test_rgbs; convertCharsToStrings(test_rgb)];
        test_depths = [test_depths; convertCharsToStrings(test_depth)];
    end
        
    data_dir = char(in_dir) + "/" + char(data_name);
    disp(data_dir);
    if ~exist(data_dir)
            mkdir(data_dir);
    end
    out_dir = data_dir + "_aligned_filter";
    if ~exist(out_dir)
            mkdir(out_dir);
    end
    unzip_folder = str2num(unzip_folder);
    if unzip_folder
        disp("Unzip the dataset!");
        unzip(data_dir + ".zip", data_dir);
    end

    dir1 = dir(data_dir);
    scene_folder_names = {dir1.name}; %sub-folders
    in_subfolders_dirs = [];
    out_subfolders_dirs = [];
    scene_names = [];
    for i = scene_folder_names
        if i ~= "." && i ~= ".."
            str = out_dir + "/" + char(i);
            scene_names = [scene_names, i];
            out_subfolders_dirs = [out_subfolders_dirs, str];
            str2 = data_dir + "/" + char(i);
            in_subfolders_dirs = [in_subfolders_dirs, str2];
        end
    end    
    disp(scene_names)
    disp(in_subfolders_dirs)

    num_sub_dir = str2num(num_sub_dir); %start 
    for i = num_sub_dir:1:size(in_subfolders_dirs, 2)
        in_scene_dir = in_subfolders_dirs(1, i);
        out_scene_dir = out_subfolders_dirs(1, i);
        if ~exist(out_scene_dir)
            mkdir(out_scene_dir)
        end
        framefile = get_synched_frames(in_scene_dir);
        rawdata_train_annotation_dir1 = char(out_scene_dir) + "/" + char(scene_names(1,i)) + "_annotation.mat";
        rawdata_train_annotation_dir2 = char(in_dir) + "/" + "annotations/" + char(scene_names(1,i)) + "_annotation.mat";
        if ~exist(char(in_dir) + "/" + "annotations")
            mkdir(char(in_dir) + "/" + "annotations")
        end
        
        anno_depths_name = {};
        anno_rgbs_name = {};
        %%Get paired depth and rgb data, get data every 20fps 
        for ii = 1 : 20 : numel(framefile)
            try
                depth_filename = framefile(ii).rawDepthFilename;
                rgb_filename = framefile(ii).rawRgbFilename;
            catch
                disp("fail to get rawDepthFilename or rawRgbFilename!");
                continue;
            end
            index_find_depth = find(test_depths == convertCharsToStrings(scene_names{i} + "/" + depth_filename));
            index_find_rgb = find(test_rgbs == convertCharsToStrings(scene_names{i} + "/" + rgb_filename));
            if ~isempty(index_find_depth) || ~isempty(index_find_rgb)
                continue;
            end
            depth_file = char(in_scene_dir + "/" + depth_filename);
            rgb_file = char(in_scene_dir + "/" + rgb_filename);
            
            try
                if isempty(strfind(depth_filename, 'pgm')) || isempty(strfind(rgb_filename, 'ppm'))
                    continue;
                end
            catch 
                disp("determin depth and rgb format fail!");
                continue;
            end
            
            try
                depth = imread(depth_file);
                depth = swapbytes(depth);
                rgb = imread(rgb_file);
            catch
		        disp("read fail!");
                continue;
            end
            anno_depths_name{end + 1} = [data_name, '/', scene_names{i}, '/',depth_filename];
            anno_rgbs_name{end + 1} = [data_name, '/', scene_names{i}, '/', rgb_filename];
            [depth_aligned, rgb_undistort] = project_depth_map(depth, rgb);
            depth_dinoised = fill_depth_colorization(rgb_undistort, depth_aligned, 0.9);
            depth_dinoised_out = uint16(round(depth_dinoised * 1000.));

            imwrite(depth_dinoised_out, char(out_scene_dir + "/" + depth_filename));
            imwrite(rgb_undistort, char(out_scene_dir + "/" + rgb_filename));
        end
        rawdata_train_anno = {};
        rawdata_train_anno{end + 1} = numel(anno_depths_name);
        rawdata_train_anno{end + 1} = anno_depths_name;
        rawdata_train_anno{end + 1} = anno_rgbs_name;
        save(rawdata_train_annotation_dir1, 'rawdata_train_anno');
        save(rawdata_train_annotation_dir2, 'rawdata_train_anno');
	    disp("finish a subfolder!");
    end

    if exist(data_dir)
        try
            rmdir(data_dir);
	        disp("remove the unzip folder!");
        catch
            disp("rm dir failed!");
        end
    end
    disp("finished!");

end





