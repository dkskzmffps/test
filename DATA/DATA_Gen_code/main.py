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
data_path       = 'DATA' + path_separator + 'Data_Gen_matlab' + path_separator + 're_DATA' + path_separator
target_class    = ['Holding', 'Rest']
# target_class    = ['Holding', 'Hypercapnia', 'Paced10', 'Paced20', 'Paced30', 'Rest']
# target_class    = ['Holding', 'Hypercapnia', 'Paced10', 'Paced30', 'Rest']
# target_class    = ['Holding', 'Paced10', 'Paced30', 'Rest']

separate_criteria = 'sub' # sub or all
data_loc        = -2
# data_channel    = 24    # All->7ch, Hbo2->2ch, Hhb->2ch, Hbo2HhbThb->6ch
data_length     = 384   # about 3.2 seconds
data_margin     = 20    # Margins to be left at both ends of the data
                        # (most of the data values at the beginning and end of the data are recorded strangely)
data_train_ratio= 0.8

for for_i in range(len(project_path_split) + data_loc):
    if for_i == 0:
        common_path = project_path_split[0] + path_separator
    else:
        common_path = common_path + project_path_split[for_i] + path_separator

data_path = common_path + data_path

save_common_path = 'DATA_Classification' + path_separator

# file_name = 'data.mat' # data.mat, data_rest_modify.mat
file_name = 'data_rest_modify.mat'

if file_name == 'data.mat':
    last_name_str = '.pt'
elif file_name == 'data_rest_modify.mat':
    last_name_str = '_rest_modify.pt'
else:
    JH_utile.Error_dataset_check

if separate_criteria == 'all':
    save_name = save_common_path + 'data_length' + '_' + str(data_length) + '_' +\
                'class_'+ str(len(target_class)) + '_mix_all' + last_name_str
elif separate_criteria == 'sub':
    save_name = save_common_path + 'data_length' + '_' + str(data_length) + '_' +\
                'class_'+ str(len(target_class)) + '_sub' + last_name_str
else:
    raise JH_utile.Error_separate_criteria

data_ori        = scipy.io.loadmat(os.path.join(data_path, file_name))
merged_data_dict= JH_utile.merging_ori_data(data_ori, data_length, data_margin,2,target_class)
selected_data   = JH_utile.select_data(merged_data_dict)


if separate_criteria == 'all':
    data_set        = JH_utile.gen_dataset(selected_data, data_train_ratio)
elif separate_criteria == 'sub':
    data_set        = JH_utile.gen_dataset_sub(selected_data, data_train_ratio)
else:
    raise JH_utile.Error_separate_criteria

print('done data_set')
torch.save(data_set, common_path + save_name)
print('done file_save')
exit()
