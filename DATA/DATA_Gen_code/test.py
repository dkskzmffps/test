import torch
import scipy.io
import os
import numpy as np
import JH_utile
# import JH_utile_backup as JH_utile
import platform

system_check = platform.system()
if system_check == 'Windows':
    path_separator = '\\'
elif system_check == 'Linux':
    path_separator = '/'
else:
    JH_utile.Error_system_check


project_path        = os.getcwd()
project_path_split  = project_path.split(path_separator)

separate_criteria = 'sub' # sub or all
data_loc        = -2
data_length     = 384
class_number    = 2


for for_i in range(len(project_path_split) + data_loc):
    if for_i == 0:
        common_path = project_path_split[0] + path_separator
    else:
        common_path = common_path + project_path_split[for_i] + path_separator



save_common_path = 'DATA_Classification' + path_separator

file_name = 'data.mat' # data.mat, data_rest_modify.mat

if separate_criteria == 'all':
    save_name = save_common_path + 'data_length' + '_' + str(data_length) + '_' +\
                'class_'+ str(class_number) + '_mix_all.pt'
elif separate_criteria == 'sub':
    save_name = save_common_path + 'data_length' + '_' + str(data_length) + '_' +\
                'class_'+ str(class_number) + '_sub.pt'
else:
    raise JH_utile.Error_separate_criteria

data_path = common_path + save_name

data_ori        = torch.load(data_path)


print('loadded data')

exit()
