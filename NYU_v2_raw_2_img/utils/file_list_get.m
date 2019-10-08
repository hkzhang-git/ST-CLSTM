function [file_list, file_count]=file_list_get(files_dir)
list=dir(files_dir);
file_list = {list.name};
file_count = size(list, 1);
end