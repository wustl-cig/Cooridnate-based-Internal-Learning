# This script is for generating training examples
# Yu Sun, WUSTL, 2019

from os import listdir
from os.path import isfile, join

import scipy.io as spio
import imageio
import numpy as np
import os
import h5py
import odl

np.random.seed(512)

# add noise according to the input SNR
def addawgn(z, input_snr):
    if input_snr is None:
        return z, None
    shape = z.shape
    z = z.flatten('F')
    noiseNorm = np.linalg.norm(z.flatten('F')) * 10 ** (-input_snr / 20)
    xBool = np.isreal(z)
    real = True
    for e in np.nditer(xBool):
        if e == False:
            real = False
    if (real == True):
        noise = np.random.randn(z.size)
    else:
        noise = np.random.randn(z.size) + 1j * np.random.randn(z.size)

    noise = noise / np.linalg.norm(noise.flatten('F')) * noiseNorm
    y = z + noise
    return y.reshape(shape,order='F'), noise.reshape(shape,order='F')


if __name__ == "__main__":

    ####################
    ####    Setup   ####
    ####################

    # Specify the image path
    image_path = 'CT_images_preprocessed.mat'

    # Specify the number of projections (total meas = num_proj * num_dete)
    # num_proj = 90

    # Specify the number of detectors
    num_dete = 512

    # Sepcify the amount of noise (dB)
    # input_snr = 70

    # Sepcify the validation
    num_val_proj = 30

    ######################
    ####   Generate   ####
    ######################

    for num_proj in [60,90,120]:
        for input_snr in [30,40,50]:
            for img_idx in range(8):
                # Sepcify the name of the trianing dataset
                dataset_name = 'PBCT_{}_{}_{}_{}'.format(img_idx, num_proj, input_snr, num_val_proj)

                # start generating...
                print(f'Start Generating the Training & Validation Dataset for Image {img_idx} . . .')
                # load image #90
                img = spio.loadmat(image_path)['img_cropped'][:,:,img_idx]
                print('The image shape is: ', img.shape)
                
                # generate projections
                reco_space = odl.uniform_discr(min_pt=[-1,-1], max_pt=[1,1], shape=img.shape, dtype='float32')
                # Angles: uniformly spaced, n = 360, min = 0, max = pi + fan angle, or being specified.
                theta = np.linspace(-0.5*np.pi, 0.5*np.pi, num_proj, endpoint=False)
                grid = odl.RectGrid(theta)
                angles = odl.uniform_partition_fromgrid(grid)
                # Detector: uniformly sampled, n = 512, min = -40, max = 40
                detector_partition = odl.uniform_partition(-1, 1, num_dete)
                # Geometry with large fan angle
                geometry = odl.tomo.Parallel2dGeometry(angles, detector_partition)
                # Ray transform (= forward projection). We use the ASTRA CUDA backend.
                ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')
                # projections
                projs = np.array(ray_trafo(img))
                # add noise
                noisy, _ = addawgn(projs, input_snr)
                
                x_pairs = []
                y_values = []
                for i in range(num_proj):
                    for j in range(num_dete):
                        x_pairs.append([theta[i]/np.pi, (j+1)/num_dete])  # normalize to [0,1]
                        y_values.append(noisy[i,j])

                train_inputs = np.array(x_pairs)
                train_truths = np.array(y_values)

                # generate validation
                # Angles: uniformly spaced, n = 360, min = 0, max = pi + fan angle, or being specified.
                valid_theta = np.sort(np.random.choice(
                    np.linspace(-0.5*np.pi, 0.5*np.pi, 1800), num_val_proj, replace=False))
                grid = odl.RectGrid(np.sort(valid_theta))
                valid_angles = odl.uniform_partition_fromgrid(grid)
                # Geometry with large fan angle
                geometry = odl.tomo.Parallel2dGeometry(valid_angles, detector_partition)
                # Ray transform (= forward projection). We use the ASTRA CUDA backend.
                ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')
                projs = np.array(ray_trafo(img))

                # generate x and y for validation
                x_pairs = []
                y_values = []
                for i in range(num_val_proj):
                    for j in range(num_dete):
                        x_pairs.append([valid_theta[i]/np.pi, (j+1)/num_dete])
                        y_values.append(projs[i,j])

                valid_inputs = np.array(x_pairs)
                valid_truths = np.array(y_values)
                
                # print out and save
                print('Saving the dataset . . .')
                with h5py.File(dataset_name, 'w') as hf:
                    hf.create_dataset("tri_inputs",  data=train_inputs) # training coordinates 
                    hf.create_dataset("tri_truths",  data=train_truths) # training amplitudes
                    hf.create_dataset("val_inputs",  data=valid_inputs) # testing cooridnates
                    hf.create_dataset("val_truths",  data=valid_truths) # testing coordinates

                print('. . . Finished')
                print('Training Data: #{}, [1,{}] [1,{}]'.format(train_inputs.shape[0], train_inputs.shape[1], train_truths.shape[0]))
                print('Validation Data: #{}, [1,{}] [1,{}]'.format(valid_inputs.shape[0], valid_inputs.shape[1], valid_truths.shape[0]))
