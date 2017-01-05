Contact: Maheen Rashid (mhnrashid@ucdavis.edu)
##Getting Started

*Download the code from GitHub:
```bash
git clone https://github.com/menoRashid/animal_human_kp
cd animal_human_kp
```
*Install Torch. Instructions are [here](LINK TO TORCH)
*Install Torch requirements.
```bash
sh install_torch_requirements.sh
```
Install the Spatial Tranformer module provided:
```bash
cd stnbhwd-master
CODE FOR INSTALLING
```
It is a modification of the code from [Spatial Transformer Network (Jaderberg et al.)](https://github.com/qassemoquab/stnbhwd) and includes a Thin Plate Spline grid generator layer.

##Dataset
*Download the [Horse Dataset](DROPBOXLINK) (SIZE)
*Run the following commands
```bash
cd data
CODE FOR UNZIPPING
```bash

##Models
*Download the pretrained and untrained models [here](DROPBOXLINK) (SIZE)
*Run the following commands
```bash
cd models
CODE FOR UNZIPPING
```

##Testing
*To test pretrained model run the following commands
```bash
cd torch
th test.th out_dir_images <path to results directory>
python ../python visualize_results.py --test_dir <path to results directory> 
```
after replacing <path to results directory> with the path to the folder where you would like the output images to be saved.

A webpage with the results would be in the results directory.
```bash
<path to results directory>/results.html
```

##Training
*The file for training the full model is 
```
	torch/train_full_model.th
```
For details on training run 
```bash
cd torch
th train_full_model.th -help
```
To train the model with affine warping uncomment lines 373-375. Currently, all parameters are the parameters used in the paper.

*The file for training the warping network is 
```
	torch/train_warping_net.th
```
For details on training run
```bash
	cd torch
	th train_warping_net.th -help
```
To train the model with affine warping uncomment lines 313-314.
