Contact: Maheen Rashid (mhnrashid@ucdavis.edu)
##Getting Started

Download the code from GitHub:
```bash
git clone https://github.com/menoRashid/animal_human_kp
cd animal_human_kp
```
Install Torch. Instructions are [here](http://torch.ch/docs/getting-started.html)

Install Torch requirements:
* [torchx](https://github.com/nicholas-leonard/torchx)
```bash
luarocks install torchx
```
* [npy4th](https://github.com/htwaijry/npy4th) (You may need to checkout commit from 5-10-16)
```bash
git clone https://github.com/htwaijry/npy4th.git
cd npy4th
luarocks make
```

Install Python requirements if needed:
* [numpy](http://www.numpy.org/)
* [scipy](https://www.scipy.org/install.html)
* [matplotlib](http://matplotlib.org/users/installing.html)
* [PIL](http://www.pythonware.com/products/pil/)

Install the Spatial Tranformer module provided:
```bash
cd stnbhwd-master
luarocks make
```
It is a modification of the code from [Spatial Transformer Network (Jaderberg et al.)](https://github.com/qassemoquab/stnbhwd) and includes a Thin Plate Spline grid generator layer.

##Dataset
Download the [Horse Dataset](https://www.dropbox.com/s/9t770jhcjqo3mmg/release_data.zip) (580 MB)

Run the following commands
```bash
cd data
unzip <path to data zip file>
```

##Models
To download all the pretrained and untrained models go [here](https://www.dropbox.com/s/4r2ghpwoncq656t/release_models.zip) (8.6 GB)

Run the following commands
```bash
cd models
unzip <path to models zip file>
```
Otherwise add the individual models to *models/*
* [Full model for horses with tps warping](https://www.dropbox.com/s/3tc6qxqv9qw3suq/horse_full_model.dat)(4.4 GB)
* [Full model for horses with affine warping](https://www.dropbox.com/s/crg4ss6bbn0pt0n/horse_full_model_affine.dat)(4.4 GB)
* [TPS Warping model for horses](https://www.dropbox.com/s/penbxkrpu9on0js/horse_tps_model.dat)(738 MB)
* [Affine Warping model for horses](https://www.dropbox.com/s/1c6ujbh49ye8j61/horse_affine_model.dat)(764 MB)
* [Keypoint network trained on human faces](https://www.dropbox.com/s/u3l2qrmke66t62n/human_face_model.dat)(3.4 GB)
* [Untrained TPS Warping model](https://www.dropbox.com/s/19twyz9865zrz2a/tps_localization_net_untrained.dat)(111 MB)
* [Untrained Affine Warping model](https://www.dropbox.com/s/byxphqqmzrkstpa/affine_localization_net_untrained.dat)(121 MB)

##Testing
To test pretrained model run the following commands
```bash
cd torch
th test.th -out_dir_images <path to results directory>
python ../python/visualize_results.py --test_dir <path to results directory> 
```
after replacing <path to results directory> with the path to the folder where you would like the output images to be saved.

A webpage with the results would be in the results directory.
```bash
<path to results directory>/results.html
```

##Training
The file for training the full model is 
```
torch/train_full_model.th
```
For details on training run 
```bash
cd torch
th train_full_model.th -help
```
To train the model with affine warping uncomment lines 373-375. Currently, all parameters are the parameters used in the paper.

The file for training the warping network is 
```
torch/train_warping_net.th
```
For details on training run
```bash
cd torch
th train_warping_net.th -help
```
To train the model with affine warping uncomment lines 313-314.
