# Setting up the Factory AI environment for Virtual Outlier Synthesis (VOS) Project

Load conda module if not already done:
```bash
module load anaconda/4.9.3
```
Load cuda module if not already done:
```bash
module load cuda/11.6
```

Create conda environment for the project with python 3.8
```bash
conda create -n p38_vos_env python=3.8
```

Activate the environment
```bash
conda activate p38_vos_env
```

Install pytorch with cuda, tqdm, mlflow, cython
```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c conda-forge tqdm=4.65.0
pip install mlflow=2.2.2
pip install cython
```

Clone the detectron2 repository to a different folder (For simplicity in the same parent folder as the VOS repository).
The original repo can be cloned here:
```bash
git clone -c http.sslVerify=false https://github.com/facebookresearch/detectron2.git
```

In order to ensure compatibility, a snapshot fork of this repo has been created on 11/05/2023 at this location:
```bash
git clone -c http.sslVerify=false https://robots.intra.cea.fr/TrustworthyDeepLearning/detectron2.git
```

Once cloned execute:
```bash
cd detectron2
pip install -e .
pip install opencv-python==4.7.0.*
```

Make sure the BDD dataset has been copied to a directory you know its location. For this you can use scp 
or vscode drag & drop.

Go back to the VOS folder.

scp -r .\bdd100k\ dmontoya@132.167.191.34:/home/users/dmontoya/data_repos/datasets