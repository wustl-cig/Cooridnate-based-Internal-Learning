# This script is for training CoIL
# Yu Sun, WUSTL, 2021

import os
# here indicating the GPU you want to use. if you don't have GPU, just leave it.
gpu_ind = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ind  # 0,1,2,3

import scipy.io as spio
import numpy as np
import h5py

from NeuralNetwork.models.MLP import MLP
from NeuralNetwork import Provider

###### Functions ######

def load_data(data):
    with h5py.File(os.path.abspath(data), 'r', swmr=True) as data:
        train_provider = Provider.StrictEpochProvider(
            np.float32(data['tri_inputs']), np.float32(data['tri_truths']), is_shuffle=True)
        valid_provider = Provider.StrictEpochProvider(
            np.float32(data['val_inputs']), np.float32(data['val_truths']), is_shuffle=False)

        print('\n\n')
        print('Number of Training Example: {} \nSpatial Size: {}'.format(
            np.float32(data['tri_inputs']).shape[0], np.float32(data['tri_inputs']).shape[1:]))
        print()
        print('################')
        print()
        print('Number of Validation Example: {} \nSpatial Size: {}'.format(
            np.float32(data['val_inputs']).shape[0], np.float32(data['val_inputs']).shape[1:]))
        print('\n\n')
    return train_provider, valid_provider


data_kargs = {
    'ic': 2,
    'oc': 1
}


net_kargs = {
    'skip_layers': range(2,16,2),
    'encoder_layer_num': 16,
    'decoder_layer_num': 1,
    'feature_num': 256,
    'ffm': 'linear',
    'L': 10
}

num_proj = 90 # specify the number of under-sampled measurement projecitons

if __name__ == "__main__":

    ##########   Main Loop   ########
    
    for noiseLevel in [30,40,50]:  # specify the input noise level
        
        # loop over testing images
        for img_idx in range(8):

            ####################################################
            ####                DATA LOADING                 ###
            ####################################################

            #-- Training Data --#
            data_root = 'data'
            ori_name = f'PBCT_{img_idx}_{num_proj}_{noiseLevel}_30'
            train_provider, valid_provider = load_data(os.path.join(data_root, ori_name))

            ####################################################
            ####               Neural Network                ###
            ####################################################

            net = MLP(data_kargs, net_kargs) # 1e-5

            ####################################################
            ####                 TRAINING                    ###
            ####################################################

            # epochs
            epochs = 3000

            def exp_decay(Ns, Ne, epochs):
                lamda = - (1/epochs) * np.log(Ne/Ns)
                return Ns*np.exp(-lamda*np.ones([epochs]))
                
            # learning rate
            if noiseLevel >= 50:
                start = 2e-4
                end = 1e-5
                lr = exp_decay(start, end, epochs) # for > 50 dB
            elif noiseLevel >= 40 and noiseLevel < 50:
                start = 1e-4
                end = 1e-5
                lr = exp_decay(start, end, epochs) # for 40~50 dB
            else:
                start = 1e-6
                end = 1e-7
                lr = exp_decay(start, end, epochs) # for < 40 dB


            # output paths for results
            output_path = f'proj{num_proj}/{ori_name}/models'
            prediction_path = f'proj{num_proj}/{ori_name}/validation'

            train_kargs = {
                    'batch_size': 1000,
                    'valid_size': 'full',
                    'epochs': epochs,
                    'learning_rate': lr,
                    'is_restore': False,
                    'prediction_path': prediction_path,
                    'save_epoch': 1000
                }

            print('\n\n################')
            print('##  Training  ##')
            print('################\n\n')

            net.train(output_path, train_provider, valid_provider, **train_kargs)
            del net, train_provider, valid_provider
