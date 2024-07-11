import torch
import numpy as np
import os
import math
import random
import copy
import matplotlib.pyplot as plt

# -*- coding: utf-8 -*-
# from __future__ import print_function

class Error_Channel(Exception):
    def __init__(self):
        super().__init__('Channel should be 24 or 52')

class Error_separate_criteria(Exception):
    def __init__(self):
        super().__init__('separate_criteria should be sub or all')

class Error_system_check(Exception):
    def __init__(self):
        super().__init__('Please check the system for path separator')

class Error_dataset_check(Exception):
    def __init__(self):
        super().__init__('Please check the dataset name')



def linear_interpolation_1D(input, output_size):
    input_size          = input.shape
    # scale_ratio         = (output_size-1) / (input_size[0]-1)
    scale_ratio         = output_size / input_size[0]
    scale_ratio_inverse = 1/scale_ratio        # for projection

    output = np.empty(output_size)
    # output[0] = input[0]
    # output[output_size-1] = input[input_size[0]-1]

    # temp =range(1, output_size-1)

    for for_i in range(output_size):
        pp  = for_i * scale_ratio_inverse
        if pp > input_size[0] - 1:
            pp = input_size[0] - 1
        pps = int(np.floor(pp))   # project_point_s
        ppe = int(np.ceil(pp))
        ppsy = input[pps]
        ppey = input[ppe]
        # temp = ppsy + (pp-pps) * (ppey-ppsy)/(ppe-pps)

        # temp = ppsy + (pp-pps)*(ppey-ppsy)  # ppe - pps = 1


        # temp = ppsy * ((ppe-pp)/(ppe-pps)) + ppey((pp-pps)/(ppe-pps))
        # coef_s = 1 - (pp-pps)/1 # ppe - pps = 1
        # coef_e = 1 - (ppe-pp)/1 # ppe - pps = 1
        output[for_i] = ppsy + (pp-pps)*(ppey-ppsy)  # ppe - pps = 1
        # output[for_i] = ppsy * (ppe - pp) + ppey(pp - pps) # ppe - pps = 1
    return output



def merging_ori_data(data_ori, data_length, data_margin, sampling_interval, target_class_list):
    data_ori_list = list(data_ori.keys())

    data_dict   = { 'Class_info'        : target_class_list,
                    'Act_info'          : ['All', 'Hbo2', 'Hbo2Hhb', 'Hbo2HhbThb', 'Hhb'],
                    'Wave_length_info'  : ['all'],
                    }
    wave_length_all = data_dict['Wave_length_info'][0]

    for for_i in data_dict['Act_info']:
        for for_j in data_dict['Wave_length_info']:
            data_dict['_'.join([for_i, for_j])] = []
        data_dict['_'.join([for_i, 'Sub_ID_info'])] = []
        # data_dict['_'.join([for_i, 'Sub_label_info'])] = []
        data_dict['_'.join([for_i, 'Sub_ID_list'])] = []
        data_dict['_'.join([for_i, 'label'])] = []


    for for_i in data_ori_list[3:len(data_ori_list)]:
        target_split_name = for_i.split('_')

        target_data = copy.deepcopy(data_ori[for_i])

        target_data = target_data.transpose(1,0)    # [Length, Channel] --> [Channel, Length]
        target_data = target_data[:, data_margin:-data_margin]

        # data_length check: if the target data length is shorter than data_length_min, it is ignored
        target_data_length = target_data.shape[1]
        target_class       = target_split_name[3]
        if target_class not in data_dict['Class_info']:
            continue
        if target_data_length < data_length:
            continue
        else:
            if target_split_name[1] not in data_dict['_'.join([target_split_name[0], 'Sub_ID_info'])]:
                data_dict['_'.join([target_split_name[0], 'Sub_ID_info'])].append(target_split_name[1])


            num_sampling = math.ceil((target_data_length - data_length)/sampling_interval)

            for for_j in range(num_sampling):
                temp_ind_start  = for_j * sampling_interval
                temp_ind_end    = temp_ind_start + data_length
                temp_data = target_data[:, temp_ind_start:temp_ind_end]
                data_dict['_'.join([target_split_name[0], wave_length_all])].append(np.array(temp_data))
                data_dict['_'.join([target_split_name[0], 'Sub_ID_list'])].append(target_split_name[1])
                target_label = data_dict['Class_info'].index(target_split_name[3])
                data_dict['_'.join([target_split_name[0], 'label'])].append(target_label)

    return data_dict

def select_data(data_input):
    data_dict = {'Wave_length_info': data_input['Wave_length_info'],
                 'min_samples_each_class': [],
                 'selected_ind_all': [],
                 # 'Act_info': data_input['Act_info'],
                 # 'Class_info': data_input['Class_info'],
                 # 'Sub_ID_info': data_input['_'.join([data_input['Act_info'][0], 'Sub_ID_info'])],
                 # 'Sub_ID_list': data_input['_'.join([data_input['Act_info'][0], 'Sub_ID_list'])],
                 }

    data_key = list(data_input.keys())
    selected_data = {'data_aug_info_shape_note': ['class', 'channel']}
    selected_data['data_aug_info_note_each_class'] = ['max', 'min', 'depth_max', 'depth_min']
    for for_i in data_key:
        if for_i == 'Class_info' or for_i == 'Act_info' or for_i == 'Wave_length_info':
            selected_data[for_i] = data_input[for_i]
        elif 'Sub_ID_info' in for_i:
            selected_data[for_i] = data_input[for_i]
        else:
            selected_data[for_i] =[]

    for for_i in selected_data['Act_info']:
        selected_data['_'.join([for_i, 'data_aug_info'])] = []
        for for_j in range(len(selected_data['Class_info'])):
            selected_data['_'.join([for_i, 'class', str(for_j), 'data_aug_info'])] = []

    Sub_ID_info = data_input['_'.join([data_input['Act_info'][0], 'Sub_ID_info'])]
    Sub_ID_list = data_input['_'.join([data_input['Act_info'][0], 'Sub_ID_list'])]
    label_list  = data_input['_'.join([data_input['Act_info'][0], 'label'])]
    num_data    = len(Sub_ID_list)
    num_class   = len(selected_data['Class_info'])

    for for_i in range(num_class):
        data_dict['min_samples_each_class'].append(num_data)

    for for_i in range(num_class):
        for for_j in Sub_ID_info:
            temp_min_count = 0
            data_dict['_'.join([str(for_i), for_j])] = []
            for for_k in range(num_data):
                if (label_list[for_k] == for_i) and (Sub_ID_list[for_k] == for_j):
                    temp_min_count += 1
                    data_dict['_'.join([str(for_i), for_j])].append(for_k)

            if temp_min_count < data_dict['min_samples_each_class'][for_i]:
                data_dict['min_samples_each_class'][for_i] = temp_min_count

    data_dict['min_sample'] = min(data_dict['min_samples_each_class'])


    for for_i in range(num_class):
        data_dict['_'.join(['selected', 'ind', str(for_i)])] = []
        for for_j in Sub_ID_info:
            target_index = data_dict['_'.join([str(for_i), for_j])]
            target_selected_index = random.sample(target_index, data_dict['min_sample'])

            data_dict['_'.join(['selected', 'ind', str(for_i)])].extend(target_selected_index)
            data_dict['selected_ind_all'].extend(target_selected_index)

    for for_i in data_dict['selected_ind_all']:
        for for_j in selected_data['Act_info']:
            # temp_data       = data_input['_'.join([for_j, 'all'])][for_i]
            # temp_data_size  = temp_data.shape
            # temp_data_ch    = temp_data_size[0]
            selected_data['_'.join([for_j, 'label'])].append(data_input['_'.join([for_j, 'label'])][for_i])
            selected_data['_'.join([for_j, 'Sub_ID_list'])].append(data_input['_'.join([for_j, 'Sub_ID_list'])][for_i])
            for for_k in selected_data['Wave_length_info']:
                selected_data['_'.join([for_j, 'all'])].append(data_input['_'.join([for_j, 'all'])][for_i])


    for for_i in selected_data['Act_info']:
        for for_j in range(len(selected_data['_'.join([for_i, 'all'])])):
            temp_data = selected_data['_'.join([for_i, 'all'])][for_j]
            temp_label = selected_data['_'.join([for_i, 'label'])][for_j]

            temp_data_shape = temp_data.shape
            temp_num_data_ch = temp_data_shape[0]


            if len(selected_data['_'.join([for_i, 'class', str(temp_label), 'data_aug_info'])]) != temp_num_data_ch:
                for for_k in range(temp_num_data_ch):
                    selected_data['_'.join([for_i, 'class', str(temp_label), 'data_aug_info'])].append([])

            for for_k in range(temp_num_data_ch):
                temp_data_ch = temp_data[for_k, :]
                temp_data_max = np.max(temp_data_ch)
                temp_data_min = np.min(temp_data_ch)
                temp_data_depth = temp_data_max - temp_data_min

                if not selected_data['_'.join([for_i, 'class', str(temp_label), 'data_aug_info'])][for_k]:
                    selected_data['_'.join([for_i, 'class', str(temp_label), 'data_aug_info'])][for_k] = \
                        [temp_data_max, temp_data_min, temp_data_depth,temp_data_depth]
                else:
                    if temp_data_max > selected_data['_'.join([for_i, 'class', str(temp_label), 'data_aug_info'])][for_k][0]:
                        selected_data['_'.join([for_i, 'class', str(temp_label), 'data_aug_info'])][for_k][0] = temp_data_max
                    if temp_data_min < selected_data['_'.join([for_i, 'class', str(temp_label), 'data_aug_info'])][for_k][1]:
                        selected_data['_'.join([for_i, 'class', str(temp_label), 'data_aug_info'])][for_k][1] = temp_data_min
                    if temp_data_depth > selected_data['_'.join([for_i, 'class', str(temp_label), 'data_aug_info'])][for_k][2]:
                        selected_data['_'.join([for_i, 'class', str(temp_label), 'data_aug_info'])][for_k][2] = temp_data_depth
                    if temp_data_depth < selected_data['_'.join([for_i, 'class', str(temp_label), 'data_aug_info'])][for_k][3]:
                        selected_data['_'.join([for_i, 'class', str(temp_label), 'data_aug_info'])][for_k][3] = temp_data_depth

    for for_i in selected_data['Act_info']:
        for for_j in range(num_class):
            selected_data['_'.join([for_i, 'data_aug_info'])].append(selected_data['_'.join([for_i, 'class', str(for_j), 'data_aug_info'])])

    return selected_data

def gen_dataset(data_input, data_train_ratio):
    data_test_ratio = 1 - data_train_ratio
    data_ori_list = list(data_input.keys())
    Act_info = data_input['Act_info']
    Wave_length_info = data_input['Wave_length_info']

    data_info = {}
    data_dict = {}
    data_dict['Class_info']                 = data_input['Class_info']
    data_dict['data_aug_info_shape_note']        = data_input['data_aug_info_shape_note']
    data_dict['data_aug_info_note_each_class']   = data_input['data_aug_info_note_each_class']
    num_class = len(data_input['Class_info'])

    for for_i in Act_info:
        data_dict['_'.join([for_i, 'data_aug_info'])] = \
            data_input['_'.join([for_i, 'data_aug_info'])]
        for for_j in range(len(data_dict['Class_info'])):
            data_dict['_'.join([for_i, 'class', str(for_j), 'data_aug_info'])] = \
                data_input['_'.join([for_i, 'class', str(for_j), 'data_aug_info'])]


    for for_i in Act_info:
        for for_j in Wave_length_info:
            data_dict['_'.join([for_i, for_j, 'train'])]    = []
            data_dict['_'.join([for_i, for_j, 'test'])]     = []
        data_dict['_'.join([for_i, 'train', 'label'])]      = []
        data_dict['_'.join([for_i, 'test', 'label'])]       = []
        data_dict['train_number_of_each_class']        = []
        data_dict['test_number_of_each_class']         = []


    # select sub for train and test
    all_Sub_ID_info = data_input['_'.join([Act_info[0], 'Sub_ID_info'])]
    # target_sub_label_info   = data_input['_'.join([Act_info[0], 'Sub_label_info'])]
    num_all_sub  = len(all_Sub_ID_info)
    num_data            = len(data_input['_'.join([Act_info[0], 'all'])])
    min_samples         = int(num_data / num_all_sub / num_class)

    target_num_test         = math.floor(min_samples * data_test_ratio)
    target_data_ind         = list(range(int(min_samples)))
    target_test_ind_list    = random.sample(target_data_ind, target_num_test)
    # target_test_ind_list    = []
    # target_train_ind        = target_data_ind
    target_train_ind_list   = target_data_ind


    data_dict['train_number_of_each_class'] = int((min_samples-target_num_test) *  num_all_sub)
    data_dict['test_number_of_each_class'] = target_num_test * num_all_sub

    for for_i in target_test_ind_list:
        target_train_ind_list.remove(for_i)

    for for_i in range(num_class):
        for for_j in all_Sub_ID_info:
            data_info['_'.join([str(for_i), for_j])] = []

    for for_i in range(num_data):
        temp_class = data_input['_'.join([Act_info[0], 'label'])][for_i]
        temp_sub_ID = data_input['_'.join([Act_info[0], 'Sub_ID_list'])][for_i]
        data_info['_'.join([str(temp_class), temp_sub_ID])].append(for_i)

    for for_i in range(min_samples):
        if for_i in target_train_ind_list:
            temp_task_target = 'train'
        else:
            temp_task_target = 'test'

        for for_j in list(data_info.keys()):
            temp_ind = data_info[for_j][for_i]

            for for_k in Act_info:
                for for_l in Wave_length_info:
                    temp_data = data_input['_'.join([for_k, for_l])][temp_ind]
                    temp_label = data_input['_'.join([for_k, 'label'])][temp_ind]
                    data_dict['_'.join([for_k, for_l, temp_task_target])].append(temp_data)
                    data_dict['_'.join([for_k, temp_task_target, 'label'])].append(temp_label)

    # convert list to numpy
    print('Convert list to numpy')
    for for_i in Act_info:
        for for_j in Wave_length_info:
            data_dict['_'.join([for_i, for_j, 'test'])] = np.array(data_dict['_'.join([for_i, for_j, 'test'])])
            data_dict['_'.join([for_i, for_j, 'train'])] = np.array(data_dict['_'.join([for_i, for_j, 'train'])])


    # convert numpy to tensor
    print('Convert numpy to tensor')
    for for_i in Act_info:
        for for_j in Wave_length_info:
            data_dict['_'.join([for_i, for_j, 'test'])] = \
                torch.from_numpy(data_dict['_'.join([for_i, for_j, 'test'])]).float()
            data_dict['_'.join([for_i, for_j, 'train'])] = \
                torch.from_numpy(data_dict['_'.join([for_i, for_j, 'train'])]).float()
    print('Complete converting list to numpy')



    return data_dict


def gen_dataset_sub(data_input, data_train_ratio):
    data_test_ratio = 1 - data_train_ratio
    data_ori_list = list(data_input.keys())
    Act_info = data_input['Act_info']
    Wave_length_info = data_input['Wave_length_info']

    data_dict = {}
    data_dict['Class_info']                 = data_input['Class_info']
    data_dict['data_aug_info_shape_note']        = data_input['data_aug_info_shape_note']
    data_dict['data_aug_info_note_each_class']   = data_input['data_aug_info_note_each_class']

    for for_i in Act_info:
        data_dict['_'.join([for_i, 'data_aug_info'])] = \
            data_input['_'.join([for_i, 'data_aug_info'])]
        for for_j in range(len(data_dict['Class_info'])):
            data_dict['_'.join([for_i, 'class', str(for_j), 'data_aug_info'])] = \
                data_input['_'.join([for_i, 'class', str(for_j), 'data_aug_info'])]


    for for_i in Act_info:
        for for_j in Wave_length_info:
            data_dict['_'.join([for_i, for_j, 'train'])]    = []
            data_dict['_'.join([for_i, for_j, 'test'])]     = []
        data_dict['_'.join([for_i, 'train', 'label'])]      = []
        data_dict['_'.join([for_i, 'test', 'label'])]       = []
        data_dict['_'.join([for_i, 'train_number'])]        = []
        data_dict['_'.join([for_i, 'test_number'])]         = []
        for for_k in range(len(data_input['Class_info'])):
            data_dict['_'.join([for_i, 'train_number'])].append(0)
            data_dict['_'.join([for_i, 'test_number'])].append(0)



    # select sub for train and test
    target_all_sub_info     = data_input['_'.join([Act_info[0], 'Sub_ID_info'])]
    # target_sub_label_info   = data_input['_'.join([Act_info[0], 'Sub_label_info'])]
    target_num_all_sub      = len(target_all_sub_info)



    target_num_sub_test = math.floor(target_num_all_sub * data_test_ratio)
    target_sub_ind = list(range(target_num_all_sub))
    target_sub_test_ind = random.sample(target_sub_ind, target_num_sub_test)
    target_sub_test_list = []
    target_sub_train_ind = target_sub_ind
    target_sub_train_list = []
    for for_i in target_sub_test_ind:
        target_sub_test_list.append(target_all_sub_info[for_i])
        target_sub_train_ind.remove(for_i)

    for for_i in target_sub_train_ind:
        target_sub_train_list.append(target_all_sub_info[for_i])

    data_dict['_'.join(['Sub_ID_info_train'])] = target_sub_train_list
    data_dict['_'.join(['Sub_ID_info_test'])] = target_sub_test_list

    for for_i in Act_info:
        target_sub_list = data_input['_'.join([for_i, 'Sub_ID_list'])]
        target_data_ind = list(range(len(target_sub_list))) # len(target_sub_list) present the number of data

        target_test_ind = []
        target_train_ind = []

        for for_j in target_data_ind:
            if target_sub_list[for_j] in data_dict['_'.join(['Sub_ID_info_test'])]:
                target_test_ind.append(for_j)

            elif target_sub_list[for_j] in data_dict['_'.join(['Sub_ID_info_train'])]:
                target_train_ind.append(for_j)
            else:
                continue

        # for for_j in target_test_ind:
        for count, for_j in enumerate(target_test_ind):
            print('| [test] Act: ' + for_i + '\t current index: %d / %d' %(count, len(target_test_ind)))

            for for_l in Wave_length_info:
                temp_test_data = data_input['_'.join([for_i, for_l])][for_j]
                data_dict['_'.join([for_i, for_l, 'test'])].append(temp_test_data)

            temp_test_label = data_input['_'.join([for_i, 'label'])][for_j]
            data_dict['_'.join([for_i,'test', 'label'])].append(temp_test_label)
            data_dict['_'.join([for_i, 'test_number'])][temp_test_label] = data_dict['_'.join([for_i, 'test_number'])][temp_test_label] + 1

        for count, for_j in enumerate(target_train_ind):
        # for for_j in target_train_ind:
            print('| [train] Act: ' + for_i + '\t current index: %d / %d' % (count, len(target_train_ind)))

            for for_l in Wave_length_info:
                temp_train_data = data_input['_'.join([for_i, for_l])][for_j]
                data_dict['_'.join([for_i, for_l, 'train'])].append(temp_train_data)

            temp_train_label = data_input['_'.join([for_i, 'label'])][for_j]
            data_dict['_'.join([for_i, 'train', 'label'])].append(temp_train_label)
            data_dict['_'.join([for_i, 'train_number'])][temp_train_label] = data_dict['_'.join([for_i, 'train_number'])][temp_train_label] + 1

        # convert list to numpy
        for for_j in Wave_length_info:
            print('Convert list to numpy')
            data_dict['_'.join([for_i, for_j, 'test'])] = np.array(data_dict['_'.join([for_i, for_j, 'test'])])
            data_dict['_'.join([for_i, for_j, 'train'])] = np.array(data_dict['_'.join([for_i, for_j, 'train'])])

        # convert numpy to tensor
        for for_j in Wave_length_info:
            print('Convert numpy to tensor')
            data_dict['_'.join([for_i, for_j, 'test'])] = \
                torch.from_numpy(data_dict['_'.join([for_i, for_j, 'test'])]).float()
            data_dict['_'.join([for_i, for_j, 'train'])] = \
                torch.from_numpy(data_dict['_'.join([for_i, for_j, 'train'])]).float()
        print('Complete converting list to numpy')
    return data_dict






