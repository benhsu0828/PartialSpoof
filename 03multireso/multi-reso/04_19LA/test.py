#!/usr/bin/env python

# Copyright 2021 National Institute of Informatics (author: Xin Wang, wangxin@nii.ac.jp)
# Copyright 2023 National Institute of Informatics (author: Lin Zhang, zhanglin@nii.ac.jp)
# Licensed under the BSD 3-Clause License.

"""
model.py

Self defined model definition.
Usage:

"""
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import numpy as np

import torch
import torch.nn as torch_nn
import torchaudio
import torch.nn.functional as torch_nn_func

import sandbox.block_nn as nii_nn
import sandbox.util_frontend as nii_front_end
import core_scripts.other_tools.debug as nii_debug
import core_scripts.data_io.seq_info as nii_seq_tk
import core_modules.p2sgrad as nii_p2sgrad

from multi_scale.post import MaxPool1dLin_gmlp_scales
#s3prl
import s3prl.hub as hub
device = 'cuda'

##############
## util
##############
PS_PATH="/home/ben/PartialSpoof" #path to PartialSpoof

Scale_num=7   
SSL_shift=1   ##since SSL use 20ms as frame shift, we start from 1. and 0 is for 10 ms...

Base_step=0.01 #in sec
Frame_shifts= np.array([pow(2, i) for i in np.arange(Scale_num)])[SSL_shift:] 
Frame_shifts_list= [pow(2, i) for i in np.arange(Scale_num)][SSL_shift:] 

LABEL_SCALE = 1
Multi_scales=Frame_shifts * Base_step #[0.01, 0.02, 0.04, 0.08, 0.16]

ASVSPOOF_PROTOCAL=PS_PATH+'/project-NN-Pytorch-scripts.202102/project/02-asvspoof/DATA/asvspoof2019_LA/protocol.txt' #protocal of asvspoof2019


hidd_dims ={'wav2vec':512, 'wav2vec2':768, 'hubert':768, 'wav2vec2_xlsr':1024, 'wavlm_base_plus':768, 'wav2vec2_local':1024}
ssl_model='wav2vec2_local'
ssl_ckpt=PS_PATH+'/modules/ssl_pretrain/w2v_large_lv_fsh_swbd_cv.pt'


def protocol_parse(protocol_filepath):
    """ Parse protocol of ASVspoof2019 and get bonafide/spoof for each trial
    
    input:
    -----
      protocol_filepath: string, path to the protocol file
        for convenience, I put train/dev/eval trials into a single protocol file
    
    output:
    -------
      data_buffer: dic, data_bufer[filename] -> 1 (bonafide), 0 (spoof)
    """ 
    data_buffer = {}
    temp_buffer = np.loadtxt(protocol_filepath, dtype='str')
    for row in temp_buffer:
        if row[-1] == 'bonafide':
            data_buffer[row[1]] = 1
            # print("file name: {}, label: {}".format(row[1], row[-1]))
        else:
            data_buffer[row[1]] = 0
            # print("file name: {}, label: {}".format(row[1], row[-1]))
    return data_buffer

if __name__ == '__main__':
    # Parse protocol file
    data_buffer = protocol_parse(ASVSPOOF_PROTOCAL)
    print('Protocol file parsed, total {} trials.'.format(len(data_buffer)))
    print(data_buffer['LA_D_1008730'])
    # print(data_buffer['LA_D_1047731'])
    # Print the data buffer
    # for key, value in data_buffer.items():
    #     print(f'{key}: {value}')  # Example output: filename: 1 (bonafide) or 0 (spoof)