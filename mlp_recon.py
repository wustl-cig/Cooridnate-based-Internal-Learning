# This script is for FBP reconstruction
# Yu Sun, WUSTL, 2021

import os
# here indicating the GPU you want to use. if you don't have GPU, just leave it.
gpu_ind = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ind  # 0,1,2,3

from NeuralNetwork.models.MLP import MLP

import odl
import scipy.io as spio
import numpy as np

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

# snr
evaluateSnr = lambda xtrue, x: 20*np.log10(np.linalg.norm(xtrue.flatten('F'))/np.linalg.norm(xtrue.flatten('F')-x.flatten('F')))

# args for data
data_kargs = {
    'ic': 2,
    'oc': 1
}

net_kargs = {
    'skip_layers': range(2, 16, 2),
    'encoder_layer_num': 16,
    'decoder_layer_num': 1,
    'feature_num': 256,
    'ffm': 'linear',
    'L': 10
}

# args for reconstruction
num_proj = 90
num_proj_pred = 720
num_dete = 512

if __name__ == "__main__":

    # load MLP model
    for idx in range(8):
        img = spio.loadmat('data/CT_images_preprocessed')['img_cropped'][:,:,idx]
        
        for noiseLevel in [50]: # specify the input noise level
            
            for num_proj in [60]: # specify the number of input measurement projections
            
                net = MLP(data_kargs, net_kargs)
                model = f'PBCT_{idx}_{num_proj}_{noiseLevel}_30'
                model_path = f'proj{num_proj}/{model}/models/final/model'
                for num_proj_pred in [num_proj,360]:
                    # generate projections
                    reco_space = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1,1], shape=img.shape, dtype='float32')
                    # Angles: uniformly spaced, n = 360, min = 0, max = pi + fan angle, or being specified.
                    theta = np.linspace(-0.5*np.pi, 0.5*np.pi, num_proj_pred, endpoint=False)
                    grid = odl.RectGrid(theta)
                    angles = odl.uniform_partition_fromgrid(grid)
                    # Detector: uniformly sampled, n = 512, min = -40, max = 40
                    detector_partition = odl.uniform_partition(-1, 1, num_dete)
                    # Geometry with large fan angle
                    geometry = odl.tomo.Parallel2dGeometry(angles, detector_partition)
                    # Ray transform (= forward projection). We use the ASTRA CUDA backend.
                    ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')
                    # projections
                    gt_proj = np.array(ray_trafo(img))
                    gt_proj, _ = addawgn(gt_proj, noiseLevel)

                    x_pairs = []
                    for i in range(num_proj_pred):
                        for j in range(num_dete):
                            x_pairs.append([theta[i]/(np.pi), (j+1)/num_dete])  # normalize to [0,1]
                    x_pairs = np.array(x_pairs)

                    # pass MLP
                    measurements = net.predict(model_path, x_pairs)

                    # reorder results
                    k = 0
                    sinogram = np.zeros([num_proj_pred, num_dete])
                    for i in range(num_proj_pred):
                        for j in range(num_dete):
                            sinogram[i,j] = measurements[k]
                            k = k+1

                    ray_trafo_fbp = odl.tomo.fbp_op(ray_trafo)
                    recon = np.array(ray_trafo_fbp(sinogram))
                    gt_recon = np.array(ray_trafo_fbp(gt_proj))

                    # make directory
                    abs_path = os.path.abspath(f'recon_results/{model}')
                    if not os.path.exists(abs_path):
                        os.makedirs(os.path.abspath(f'recon_results/{model}'))

                    # save mat
                    spio.savemat(os.path.join(abs_path,f'sino_gt_{num_proj_pred}.mat'),{f'sino_gt_{num_proj_pred}': gt_proj})
                    spio.savemat(os.path.join(abs_path,f'recon_gt_{num_proj_pred}.mat'),{f'recon_gt_{num_proj_pred}': gt_recon})
                    spio.savemat(os.path.join(abs_path,f'sino_{num_proj_pred}.mat'),{f'sino_{num_proj_pred}': sinogram})
                    spio.savemat(os.path.join(abs_path,f'recon_{num_proj_pred}.mat'), {f'recon_{num_proj_pred}': recon})
