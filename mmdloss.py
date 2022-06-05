import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision import datasets, transforms
import numpy as np

"""
MMD Objective using Gaussian Kernel.
The repo is from https://github.com/Saswatm123/MMD-VAE/blob/master/MMD_VAE.ipynb
"""
def gaussian_kernel(a, b):
    dim1_1, dim1_2 = a.shape[0], b.shape[0]
    depth = a.shape[1]#depth = LATENT_SIZE
    a = a.view(dim1_1, 1, depth)
    b = b.view(1, dim1_2, depth)
    a_core = a.expand(dim1_1, dim1_2, depth)
    b_core = b.expand(dim1_1, dim1_2, depth)
    numerator = (a_core - b_core).pow(2).mean(2)/depth
    return torch.exp(-numerator)

def MMD(a, b):
    return gaussian_kernel(a, a).mean() + gaussian_kernel(b, b).mean() - 2*gaussian_kernel(a, b).mean()


class Reshape(nn.Module):
    def __init__(self, *target_shape):
        super().__init__()
        self.target_shape = target_shape
    def forward(self, x):
        return x.view(*self.target_shape)

class MMD_VAE(nn.Module):
    def __init__(self,LATENT_SIZE):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 5, kernel_size = 5, padding = 2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels = 5, out_channels = 5, kernel_size = 5),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels = 5, out_channels = 5, kernel_size = 5),
            nn.LeakyReLU(),
            Reshape([-1,5*20*20]),
            nn.Linear(in_features = 5*20*20, out_features = 5*12),
            nn.LeakyReLU(),
            nn.Linear(in_features = 5*12, out_features = LATENT_SIZE)
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features = LATENT_SIZE, out_features = 5*12),
            nn.ReLU(),
            nn.Linear(in_features = 5*12, out_features = 24*24),
            nn.ReLU(),
            Reshape([-1,1,24,24]),
            nn.ConvTranspose2d(in_channels = 1, out_channels = 5, kernel_size = 3),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 5, out_channels = 10, kernel_size = 5),
            nn.ReLU(),
            nn.Conv2d(in_channels = 10, out_channels = 1, kernel_size = 3),
            nn.Sigmoid()
        )
        
    def forward(self, X):
        if self.training:
            latent = self.encoder(X)
            return self.decoder(latent), latent
        else:
            return self.decoder( self.encoder(X) )


if __name__ == "__main__":
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SEED, BATCH_SIZE, LATENT_SIZE = 123, 128, 4
    torch.manual_seed(SEED)
    # DATA I/O
    train = datasets.MNIST('mnist',train=True,transform=transforms.ToTensor(),download=True)
    train = train.data.float().to(DEVICE)/256# Converting from integer to float
    train_loader = DataLoader(dataset = train,batch_size = BATCH_SIZE,shuffle = True)
    # MODEL
    net = MMD_VAE(LATENT_SIZE).to(DEVICE)
    optimizer = optim.Adam(net.parameters(),lr=1e-4)
    net.train()
    for batchnum, X in enumerate(train_loader):
        optimizer.zero_grad()
        X = X.reshape(-1, 1, 28, 28)
        print("X.shape: ",X.shape)#torch.Size([96, 1, 28, 28])
        _, mu = net(X)
        print("mu.shape: ",mu.shape)#torch.Size([96, 4])
        mmd = MMD(torch.randn(96,LATENT_SIZE,requires_grad=False).to(DEVICE), mu)
        mmd.backward()
        optimizer.step()
        print("mmd loss: ",mmd.item())





