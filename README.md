# [CoIL: Coordinate-Based Internal Learning for Tomographic Imaging](https://ieeexplore.ieee.org/document/9606601)

We propose Coordinate-based Internal Learning (CoIL) as a new deep-learning (DL) methodology for continuous representation of measurements. Unlike traditional DL methods that learn a mapping from the measurements to the desired image, CoIL trains a multilayer perceptron (MLP) to encode the complete measurement field by mapping the coordinates of the measurements to their responses. CoIL is a self-supervised method that requires no training examples besides the measurements of the test object itself. Once the MLP is trained, CoIL generates new measurements that can be used within most image reconstruction methods. We validate CoIL on sparse-view computed tomography using several widely-used reconstruction methods, including purely model-based methods and those based on DL. Our results demonstrate the ability of CoIL to consistently improve the performance of all the considered methods by providing high-fidelity measurement fields.

**[Video](https://www.youtube.com/watch?v=7LXagKec31U)** | **[arXiv](https://arxiv.org/abs/2102.05181)** | **[IEEE-TCI](https://ieeexplore.ieee.org/document/9606601)**

## How to run the code

### Prerequisites
```
python 3.6  
tensorflow 1.14
scipy 1.2.1
numpy v1.17
matplotlib v3.3.4
```
It is better to use Conda for the installation of all dependecies.

### Data Generation
Go to `/data`, and run
```
generatePBCTData.py
```
to generate the parallel beam CT data for training the MLP. To try with different settings, please open the script and follow the instruction inside.

Next, run the following script in the main folder
```
$ python train_mlp.py
```
to train the MLPs for all the testing images. Note that you need to re-generate the corresponding data if you want to adjust noise level, number of projections, etc. Each MLP takes roughly 10-30 minutes to train on a single Nvidia GTX 1080Ti GPU.

After the training is done, you can run
```
$ python mlp_recon.py
```
to obtain the final image reconstruction via filtered backprojeciton (FBP). You can also feed the generated CoIL measurement field into your customized reconstruction algorithms such as model-based and CNN-based algorithms.

## Citation
Y. Sun, J. Liu, M. Xie, B. Wohlberg, and U. S. Kamilov, “CoIL: Coordinate-based Internal Learning for Tomographic Imaging,” IEEE Trans. Comput. Imag., vol. 7, pp. 1400-1412, November 2021.
```
@ARTICLE{9606601,
  author={Sun, Yu and Liu, Jiaming and Xie, Mingyang and Wohlberg, Brendt and Kamilov, Ulugbek S.},
  journal={IEEE Transactions on Computational Imaging}, 
  title={CoIL: Coordinate-Based Internal Learning for Tomographic Imaging}, 
  year={2021},
  volume={7},
  number={},
  pages={1400-1412},
  doi={10.1109/TCI.2021.3125564}}
```
