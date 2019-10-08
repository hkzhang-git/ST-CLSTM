function [flag] = test_sample_assert(test_list,sample_name)
flag=0;
for test_sample = test_list
    if strcmp(test_sample{1}, sample_name)
        flag=1;
        break;
    end
end
end

