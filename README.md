# Continual Learning for Visual Search with Backward Consistent Feature Embedding

PyTorch implementation for CVS (CVPR 2022).

Timmy S. T. Wan, Jun-Cheng Chen, Tzer-Yi Wu, Chu-Song Chen

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
~~~~

## Train

~~~~
Coming Soon...
~~~~

## Test

~~~~
Coming Soon...
~~~~
