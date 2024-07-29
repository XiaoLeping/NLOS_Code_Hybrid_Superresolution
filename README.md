# NLOS_Code_Hybrid_Superresolution

This repository contains reconstruction code for the paper "Fast Non-line-of-sight Imaging with Hybrid Super-resolution Network over 18 m" by Leping Xiao, Jianyu Wang, Yi Wang, Ziyu Zhan, Zuoqiang Shi, Lingyun Qiu, and Xing Fu.

## Folder Structure

### 1. `dataset_18m`
This folder contains the original signals (32x32x512) of the flat objects measured over 18 m.
- `1.mat`: Original signals of Letter N.
- `2.mat`: Original signals of Letter Z.
- `3.mat`: Original signals of the rectangular composite object.
- `4.mat`: Original signals of Letter L.
- `5.mat`: Original signals of Letter Y.

### 2. `models`
This folder contains the source code for:
- Signal super-resolution network (SSN)
- Phasor field method
- Image super-resolution network (ISN)

### 3. `weight`
This folder stores the pre-trained weights utilized for both the SSN and ISN.
- `net1_18m.pkl`: Pre-trained weight of SSN used in the long-distance scenario.
- `net2_18m.pkl`: Pre-trained weight of ISN used in the long-distance scenario.

## Demo Code
We provide a demo code for the reconstruction of the flat objects over 18 m. The parameters and the tool functions used for testing are provided in:
- `parameters.py`
- `utils.py`

By running `demo.py`, reconstruction results are generated in the following folders:
- `image-Experiment18m`: Contains a comparison of images using 'SSN+phasor' and 'SSN+phasor+ISN'.
- `OUTPUTS_Experiment18m`: Contains the recovered high-resolution signals of SSN and the reconstructed images in `.mat` format.
