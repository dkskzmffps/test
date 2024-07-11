import torch
import numpy as np
import os
import math
import random
import copy
import matplotlib.pyplot as plt

# -*- coding: utf-8 -*-
# from __future__ import print_function

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

    selected_data = {'data_aug_note': ['max', 'min', 'depth']}
    for for_i in data_key:
        if for_i == 'Class_info' or for_i == 'Act_info' or for_i == 'Wave_length_info':
            selected_data[for_i] = data_input[for_i]
        elif 'Sub_ID_info' in for_i:
            selected_data[for_i] = data_input[for_i]
        else:
            selected_data[for_i] =[]

    for for_i in selected_data['Act_info']:
        for for_j in range(len(selected_data['Class_info'])):
            selected_data['_'.join([for_i, 'class', str(for_j), 'data_aug'])] = []

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


            if len(selected_data['_'.join([for_i, 'class', str(temp_label), 'data_aug'])]) != temp_num_data_ch:
                for for_k in range(temp_num_data_ch):
                    selected_data['_'.join([for_i, 'class', str(temp_label), 'data_aug'])].append([])

            for for_k in range(temp_num_data_ch):
                temp_data_ch = temp_data[for_k, :]
                temp_data_max = np.max(temp_data_ch)
                temp_data_min = np.min(temp_data_ch)
                temp_data_depth = temp_data_max - temp_data_min

                if not selected_data['_'.join([for_i, 'class', str(temp_label), 'data_aug'])][for_k]:
                    selected_data['_'.join([for_i, 'class', str(temp_label), 'data_aug'])][for_k] = \
                        [temp_data_max, temp_data_min, temp_data_depth]
                else:
                    if temp_data_max > selected_data['_'.join([for_i, 'class', str(temp_label), 'data_aug'])][for_k][0]:
                        selected_data['_'.join([for_i, 'class', str(temp_label), 'data_aug'])][for_k][0] = temp_data_max
                    if temp_data_min < selected_data['_'.join([for_i, 'class', str(temp_label), 'data_aug'])][for_k][1]:
                        selected_data['_'.join([for_i, 'class', str(temp_label), 'data_aug'])][for_k][1] = temp_data_min
                    if temp_data_depth > selected_data['_'.join([for_i, 'class', str(temp_label), 'data_aug'])][for_k][2]:
                        selected_data['_'.join([for_i, 'class', str(temp_label), 'data_aug'])][for_k][2] = temp_data_depth

    return selected_data

def gen_dataset_sub(data_input, data_train_ratio):
    data_test_ratio = 1 - data_train_ratio
    data_ori_list = list(data_input.keys())
    Act_info = data_input['Act_info']
    Wave_length_info = data_input['Wave_length_info']

    data_dict = {}
    data_dict['Class_info'] = data_input['Class_info']

    for for_i in Act_info:
        for for_j in range(len(data_dict['Class_info'])):
            data_dict['_'.join([for_i, 'class', str(for_j), 'data_aug'])] = \
                data_input['_'.join([for_i, 'class', str(for_j), 'data_aug'])]


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




class Error_Channel(Exception):
    def __init__(self):
        super().__init__('Channel should be 24 or 52')

class Error_separate_criteria(Exception):
    def __init__(self):
        super().__init__('separate_criteria should be sub or all')

class Error_system_check(Exception):
    def __init__(self):
        super().__init__('Please check the system for path separator')



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



def merging_ori_data(data_ori, data_length, data_margin, sampling_interval):
    data_ori_list = list(data_ori.keys())
    # data_length_min = (data_margin * 2) + data_length

    data_dict   = { 'Class_info'        : ['TD', 'ASD'],  # label 0: TD, 1: ASD
                    # 'Sub_ID_info'       : [],
                    'Act_info'          : ['Watch', 'Do', 'Together'],
                    'Wave_length_info'  : ['702', '828', 'all'],
                    # 'Sub_ID_list'       : [],
                    # 'Class_label'       : [],
                    }
    wave_length1 = data_dict['Wave_length_info'][0]
    wave_length2 = data_dict['Wave_length_info'][1]
    wave_length_all = data_dict['Wave_length_info'][2]


    for for_i in data_dict['Act_info']:
        for for_j in data_dict['Wave_length_info']:
            data_dict['_'.join([for_i, for_j])] = []
        data_dict['_'.join([for_i, 'Sub_ID_info'])] = []
        data_dict['_'.join([for_i, 'Sub_label_info'])] = []
        data_dict['_'.join([for_i, 'Sub_ID_list'])] = []
        data_dict['_'.join([for_i, 'label'])] = []


    for for_i in data_ori_list[3:len(data_ori_list)]:
        target_split_name = for_i.split('_')


        # if target_split_name[1] not in data_dict['Sub_ID_info']:
        #     data_dict['Sub_ID_info'].append(target_split_name[1])
        # if target_split_name[1] not in data_dict['_'.join([target_split_name[0], 'Sub_ID_info'])]:
        #     data_dict['_'.join([target_split_name[0], 'Sub_ID_info'])].append(target_split_name[1])
        # data_dict['_'.join([target_split_name[0], 'Sub_ID_list'])].append(target_split_name[1])
        # target_label = data_dict['Class_info'].index(target_split_name[3])
        # data_dict['_'.join([target_split_name[0], 'label'])].append(target_label)


        # data_dict['Sub_ID_list'].append(target_split_name[1])
        # target_label = data_dict['Class_info'].index(target_split_name[3])
        # data_dict['Class_label'].append(target_label)

        target_data = copy.deepcopy(data_ori[for_i])

        target_data = target_data.transpose(1,0)    # [Length, Channel] --> [Channel, Length]
        target_data = target_data[:, data_margin:-data_margin]
        NIRs_channel= int(target_data.shape[0]/2);
        # target_sampling_data = []
        # target_sampling_data_wave1 = []
        # target_sampling_data_wave2 = []


        # data_length check: if the target data length is shorter than data_length_min, it is ignored
        target_data_length = target_data.shape[1]
        if target_data_length < data_length:
            continue
        else:
            if target_split_name[1] not in data_dict['_'.join([target_split_name[0], 'Sub_ID_info'])]:
                sub_label = data_dict['Class_info'].index(target_split_name[3])
                data_dict['_'.join([target_split_name[0], 'Sub_ID_info'])].append(target_split_name[1])
                data_dict['_'.join([target_split_name[0], 'Sub_label_info'])].append(sub_label)

            num_sampling = math.ceil((target_data_length - data_length)/sampling_interval)

            for for_j in range(num_sampling):
                temp_ind_start  = for_j * sampling_interval
                temp_ind_end    = temp_ind_start + data_length
                temp_data = target_data[:, temp_ind_start:temp_ind_end]

                # target_sampling_data.append(target_data[:, temp_ind_start:temp_ind_end])


                temp_data_wave1 = []
                temp_data_wave2 = []
                # odd: : wavelength 702, even: wavelength 828
                for for_k in range(NIRs_channel):
                    target_grab_wave1 = target_data[for_k * 2,       temp_ind_start:temp_ind_end]
                    target_grab_wave2 = target_data[(for_k * 2) + 1, temp_ind_start:temp_ind_end]
                    temp_data_wave1.append(target_grab_wave1)
                    temp_data_wave2.append(target_grab_wave2)
                # target_sampling_data_wave1.append(np.array(temp_data_wave1))
                # target_sampling_data_wave2.append(np.array(temp_data_wave2))
                data_dict['_'.join([target_split_name[0], wave_length1])].append(np.array(temp_data_wave1))
                data_dict['_'.join([target_split_name[0], wave_length2])].append(np.array(temp_data_wave2))
                data_dict['_'.join([target_split_name[0], wave_length_all])].append(np.array(temp_data))
                data_dict['_'.join([target_split_name[0], 'Sub_ID_list'])].append(target_split_name[1])
                target_label = data_dict['Class_info'].index(target_split_name[3])
                data_dict['_'.join([target_split_name[0], 'label'])].append(target_label)



        #     target_data = target_data[:, data_margin:data_margin+data_length]
        #     if target_split_name[1] not in data_dict['_'.join([target_split_name[0], 'Sub_ID_info'])]:
        #         data_dict['_'.join([target_split_name[0], 'Sub_ID_info'])].append(target_split_name[1])
        #         sub_label = data_dict['Class_info'].index(target_split_name[3])
        #         data_dict['_'.join([target_split_name[0], 'Sub_label_info'])].append(sub_label)
        #     data_dict['_'.join([target_split_name[0], 'Sub_ID_list'])].append(target_split_name[1])
        #     target_label = data_dict['Class_info'].index(target_split_name[3])
        #     data_dict['_'.join([target_split_name[0], 'label'])].append(target_label)
        #
        #
        # target_data_wave1 = []
        # target_data_wave2 = []
        # for for_j in range(int(target_data.shape[0]/2)):
        #     target_grab_wave1 = target_data[for_j*2     , :]
        #     target_grab_wave2 = target_data[(for_j*2)+1 , :]
        #
        #     target_data_wave1.append(target_grab_wave1)
        #     target_data_wave2.append(target_grab_wave2)
        #
        # data_dict['_'.join([target_split_name[0], wave_length1])].append(np.array(target_data_wave1))
        # data_dict['_'.join([target_split_name[0], wave_length2])].append(np.array(target_data_wave2))
        # data_dict['_'.join([target_split_name[0], wave_length_all])].append(np.array(target_data))

    return data_dict


def gen_dataset(data_input, data_train_ratio):

    data_test_ratio = 1 - data_train_ratio
    data_ori_list   = list(data_input.keys())
    Act_info        = data_input['Act_info']
    Wave_length_info= data_input['Wave_length_info']

    data_dict       = {}
    for for_i in Act_info:
        for for_j in Wave_length_info:
            data_dict['_'.join([for_i, for_j, 'train'])]            = []
            # data_dict['_'.join([for_i, for_j, 'train', 'label'])]   = []
            data_dict['_'.join([for_i, for_j, 'test'])]             = []
            # data_dict['_'.join([for_i, for_j, 'test', 'label'])]    = []
        data_dict['_'.join([for_i, 'train', 'label'])] = []
        data_dict['_'.join([for_i, 'test', 'label'])] = []


    for for_i in Act_info:
        target_dataset          = data_input['_'.join([for_i, 'label'])]
        target_num_data         = len(target_dataset)
        target_num_data_test    = math.floor(target_num_data * data_test_ratio)
        target_num_data_train   = target_num_data - target_num_data_test

        target_data_ind         = list(range(target_num_data))
        target_test_ind         = random.sample(target_data_ind, target_num_data_test)

        for for_j in target_test_ind:
            target_data_ind.remove(for_j)

            for for_k in Wave_length_info:
                temp_test_data = data_input['_'.join([for_i, for_k])][for_j]
                data_dict['_'.join([for_i, for_k, 'test'])].append(temp_test_data)

            temp_test_label = data_input['_'.join([for_i, 'label'])][for_j]
            data_dict['_'.join([for_i, 'test', 'label'])].append(temp_test_label)


        for for_j in target_data_ind:
            for for_k in Wave_length_info:
                temp_train_data = data_input['_'.join([for_i, for_k])][for_j]
                data_dict['_'.join([for_i, for_k, 'train'])].append(temp_train_data)

            temp_train_label = data_input['_'.join([for_i, 'label'])][for_j]
            data_dict['_'.join([for_i, 'train', 'label'])].append(temp_train_label)

        for for_j in Wave_length_info:
            data_dict['_'.join([for_i, for_j, 'test'])] = np.array(data_dict['_'.join([for_i, for_j, 'test'])])
            data_dict['_'.join([for_i, for_j, 'train'])] = np.array(data_dict['_'.join([for_i, for_j, 'train'])])

        for for_j in Wave_length_info:
            data_dict['_'.join([for_i, for_j, 'test'])] = \
                torch.from_numpy(data_dict['_'.join([for_i, for_j, 'test'])]).float()
            data_dict['_'.join([for_i, for_j, 'train'])] = \
                torch.from_numpy(data_dict['_'.join([for_i, for_j, 'train'])]).float()

    return data_dict


def gen_dataset_sub(data_input, data_train_ratio):
    data_test_ratio = 1 - data_train_ratio
    data_ori_list = list(data_input.keys())
    Act_info = data_input['Act_info']
    Wave_length_info = data_input['Wave_length_info']

    data_dict = {}
    for for_i in Act_info:
        for for_j in Wave_length_info:
            data_dict['_'.join([for_i, for_j, 'train'])] = []
            # data_dict['_'.join([for_i, for_j, 'train', 'label'])]   = []
            data_dict['_'.join([for_i, for_j, 'test'])] = []
            # data_dict['_'.join([for_i, for_j, 'test', 'label'])]    = []
        data_dict['_'.join([for_i, 'train', 'label'])] = []
        data_dict['_'.join([for_i, 'test', 'label'])] = []

    # select sub for train and test



    for for_i in Act_info:
        target_all_sub_info = data_input['_'.join([for_i, 'Sub_ID_info'])]
        target_sub_label_info = data_input['_'.join([for_i, 'Sub_label_info'])]
        target_num_all_sub = len(target_all_sub_info)
        # target_num_sub_test = math.floor(target_num_sub * data_test_ratio)
        # target_num_sub_train = target_num_sub - target_num_sub_test

        data_num_sample = {}
        data_num_sample['label_info'] = []

        for for_j in range(target_num_all_sub):
            if target_sub_label_info[for_j] not in data_num_sample['label_info']:
                data_num_sample['label_info'].append(target_sub_label_info[for_j])

        target_num_label = len(data_num_sample['label_info'])

        for for_j in range(target_num_label):
            data_num_sample['_'.join(['label', str(for_j)])] = []
            data_num_sample['_'.join(['num', 'label', str(for_j)])] = []
            data_num_sample['_'.join(['Sub_ID_info', 'label', str(for_j)])] = []
            # data_num_sample['_'.join(['num', 'label', str(for_j), 'test'])] = []

        for for_j in range(target_num_all_sub):
            data_num_sample['_'.join(['label', str(target_sub_label_info[for_j])])].append(for_j)
            data_num_sample['_'.join(['Sub_ID_info', 'label', str(target_sub_label_info[for_j])])].append(target_all_sub_info[for_j])

        for for_j in range(target_num_label):
            temp_num_label = len(data_num_sample['_'.join(['label', str(for_j)])])
            data_num_sample['_'.join(['num', 'label', str(for_j)])] = temp_num_label
            # data_num_sample['_'.join(['num', 'label', str(for_j), 'test'])] = math.floor(target_num)

            target_sub_info = data_num_sample['_'.join(['Sub_ID_info', 'label', str(for_j)])]

            target_num_sub_test = math.floor(temp_num_label * data_test_ratio)
            target_sub_ind = list(range(temp_num_label))
            target_sub_test_ind = random.sample(target_sub_ind, target_num_sub_test)
            target_sub_test_list = []

            target_sub_train_ind = target_sub_ind
            target_sub_train_list = []


            for for_k in target_sub_test_ind:
                target_sub_test_list.append(target_sub_info[for_k])
                target_sub_train_ind.remove(for_k)

            for for_k in target_sub_train_ind:
                target_sub_train_list.append(target_sub_info[for_k])

            target_sub_list = data_input['_'.join([for_i, 'Sub_ID_list'])]
            target_num_data = len(target_sub_list)
            target_data_ind = list(range(target_num_data))

            target_test_ind = []
            target_train_ind = []

            for for_k in target_data_ind:
                if target_sub_list[for_k] in target_sub_test_list:
                    target_test_ind.append(for_k)

                elif target_sub_list[for_k] in target_sub_train_list:
                    target_train_ind.append(for_k)
                else:
                    # teno=1
                    continue

            for for_k in target_test_ind:
                for for_l in Wave_length_info:
                    temp_test_data = data_input['_'.join([for_i, for_l])][for_k]
                    data_dict['_'.join([for_i, for_l, 'test'])].append(temp_test_data)

                temp_test_label = data_input['_'.join([for_i, 'label'])][for_k]
                data_dict['_'.join([for_i,'test', 'label'])].append(temp_test_label)


            for for_k in target_train_ind:
                for for_l in Wave_length_info:
                    temp_train_data = data_input['_'.join([for_i, for_l])][for_k]
                    data_dict['_'.join([for_i, for_l, 'train'])].append(temp_train_data)

                temp_train_label = data_input['_'.join([for_i, 'label'])][for_k]
                data_dict['_'.join([for_i, 'train', 'label'])].append(temp_train_label)


        for for_j in Wave_length_info:
            data_dict['_'.join([for_i, for_j, 'test'])] = np.array(data_dict['_'.join([for_i, for_j, 'test'])])
            data_dict['_'.join([for_i, for_j, 'train'])] = np.array(data_dict['_'.join([for_i, for_j, 'train'])])

        for for_j in Wave_length_info:
            data_dict['_'.join([for_i, for_j, 'test'])] = \
                torch.from_numpy(data_dict['_'.join([for_i, for_j, 'test'])]).float()
            data_dict['_'.join([for_i, for_j, 'train'])] = \
                torch.from_numpy(data_dict['_'.join([for_i, for_j, 'train'])]).float()

    return data_dict


