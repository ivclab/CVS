# Continual Learning for Visual Search with Backward Consistent Feature Embedding

PyTorch implementation for CVS (CVPR 2022).

Timmy S. T. Wan, Jun-Cheng Chen, Tzer-Yi Wu, Chu-Song Chen

[[Paper](https://arxiv.org/abs/2205.13384)]

## Abstract

In visual search, the gallery set could be incrementally growing and added to the database in practice. However, existing methods rely on the model trained on the entire dataset, ignoring the continual updating of the model. Besides, as the model updates, the new model must re-extract features for the entire gallery set to maintain compatible feature space, imposing a high computational cost for a large gallery set. To address the issues of long-term visual search, we introduce a continual learning (CL) approach that can handle the incrementally growing gallery set with backward embedding consistency. We enforce the losses of inter-session data coherence, neighbor-session model coherence, and intra-session discrimination to conduct a continual learner. In addition to the disjoint setup, our CL solution also tackles the situation of increasingly adding new classes for the blurry boundary without assuming all categories known in the beginning and during model update. To our knowledge, this is the first CL method both tackling the issue of backward-consistent feature embedding and allowing novel classes to occur in the new sessions. Extensive experiments on various benchmarks show the efficacy of our approach under a wide range of setups.

## Install

~~~~
conda create -n cvs python=3.6.13 -y
conda activate cvs
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.2 -c pytorch -y
conda install numpy=1.19.2 -y
conda install python-dateutil -y
conda install -c conda-forge pretrainedmodels=0.7.4 -y
conda install pandas=1.0.4 -y
conda install -c conda-forge scikit-learn=0.23.0 -y
pip install randaugment==1.0.2
pip install easydict==1.9
conda install -c pytorch faiss-gpu=1.7.0 -y
pip install matplotlib
~~~~

## Dataset

Please save datasets in `dataset` folder. Each dataset format is the same as [Rainbow's](https://github.com/clovaai/rainbow-memory#datasets).

1. Unzip the [CIFAR100](https://github.com/hwany-j/cifar100_png) and rename the extracted folder to `cifar100`
2. Unzip the [Tiny ImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip) and run `dataset/build_tinyimagenet.py` to convert the format. Then, put train-val-test files([here](https://drive.google.com/drive/folders/1CpbWbynZXZMOxV6gcedeJYKPp8aLn1xS?usp=sharing)) into `collections`.
3. Unzip the [Product-10K dataset (train.zip)](https://onedrive.live.com/?cid=1bfdba15301520ef&id=1BFDBA15301520EF%211598&authkey=!ABwlxkUe6Gyxh4s) and put train-val-test files([here](https://drive.google.com/drive/folders/1CpbWbynZXZMOxV6gcedeJYKPp8aLn1xS?usp=sharing)) into `collections`. Then, run `dataset/build_productm.py` to generate the Proudct-M.
4. For Stanford Dog and iNat-M, please download the files ([Stanford Dog](http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar) / [iNatualist 2017](https://github.com/visipedia/inat_comp/tree/master/2017)). You can generate the same setting as the paper mentioned with slightly modification to `dataset/build_productm.py`.


## Train

#### Lower Bound: Finetune the model directly.

~~~~
# For cifar100
bash scripts/cifar.sh -s blurry10 # change it depends on the experimental setup.
# E.g. bash scripts/cifar.sh -s general10
# E.g. bash scripts/cifar.sh -s disjoint

# For Tiny Imagenet
bash scripts/tinyimagenet.sh -s blurry30 # Options: [blurry30, general30, disjoint]
# E.g. bash tinyimagenet.sh -s general30
~~~~

The result of the first session will be in `exp/[DATASET]_[SETUP]/result.json`

For the later session j, the result will be in `exp/[DATASET]_[SETUP]_[METHOD]$(j-1)/result.json`

#### Upper Bound: Perform joint training while allowing re-extraction.

~~~~
bash scripts/cifar.sh -s blurry10 -m jt # For cifar100 under blurry setup

bash scripts/tinyimagenet.sh -s blurry30 -m jt # For Tiny ImageNet under blurry setup
~~~~

#### CVS

~~~~
bash scripts/cifar.sh -s blurry10 -m cvs # For cifar100 under blurry setup
bash scripts/cifar.sh -s general10 -m cvs # For cifar100 under general setup
bash scripts/cifar.sh -s disjoint -m cvs # For cifar100 under disjoint setup

bash scripts/tinyimagenet.sh -s blurry30 -m cvs # For Tiny ImageNet under blurry setup
bash scripts/tinyimagenet.sh -s general30 -m cvs # For Tiny ImageNet under general setup
bash scripts/tinyimagenet.sh -s disjoint -m cvs # For Tiny ImageNet under disjoint setup
~~~~

#### [MMD](https://arxiv.org/abs/2010.08020)

~~~~
bash scripts/cifar.sh -s blurry10 -m mmd # For cifar100
~~~~

#### [BCT](https://arxiv.org/abs/2003.11942)

~~~~
bash scripts/cifar.sh -s blurry10 -m bct # For cifar100
~~~~

#### [LWF](https://arxiv.org/abs/1606.09282)

~~~~
bash scripts/cifar.sh -s blurry10 -m lwf # For cifar100
~~~~


## Test

Download the checkpoint folder [here](https://drive.google.com/drive/folders/1wroCN-cxKSQ2zei7IX0RgC8y-REZ2PT2?usp=sharing) and put it in the same path as `test.py`. Then run `bash test_CVS.sh`.
