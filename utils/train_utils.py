import numpy as np
import torch, os, pdb
import torch.nn.functional as F
import matplotlib.pyplot as plt
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
#import torch_optimizer, pdb
from easydict import EasyDict as edict
from torch import optim
from models import cifar, imagenet, resnet

def select_optimizer(opt_name, lr, model, sched_name="cos"):
    # Set optimizer
    if opt_name == "adam":
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    elif opt_name == "radam":
        pass#opt = torch_optimizer.RAdam(model.parameters(), lr=lr, weight_decay=0.00001)
    elif opt_name == "sgd":
        opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4)
    else:
        raise NotImplementedError("Please select the opt_name [adam, sgd]")
    # Set scheduler
    if sched_name == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=1, T_mult=2, eta_min=lr * 0.01)
    elif sched_name == "anneal":
        scheduler = optim.lr_scheduler.ExponentialLR(opt, 1 / 1.1, last_epoch=-1)
    elif sched_name == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[30, 60, 80, 90], gamma=0.1)
    elif sched_name == "none":
        scheduler = None
    else:
        raise NotImplementedError("Please select the sched_name [cos, anneal, multistep]")
    return opt, scheduler


def select_model(model_name, dataset, num_classes=None, feature_size=128, nsloss=False,pretrain=False):
    opt = edict(
        {
            "depth": 18,
            "num_classes": num_classes,
            "in_channels": 3,
            "bn": True,
            "normtype": "BatchNorm",
            "activetype": "ReLU",
            "pooltype": "MaxPool2d",
            "preact": False,
            "affine_bn": True,
            "bn_eps": 1e-6,
            "compression": 0.5,
            "feature_size": feature_size,
            "nsloss":nsloss,
        }
    )

    ###############
    if pretrain:
        #print("!!!!! [Load ImageNet-pretrained model] !!!!!")
        if model_name == "resnet18":
            model = resnet.PretrainedFrozenResNet18(opt)
        elif model_name == "resnet34":
            model = resnet.PretrainedFrozenResNet34(opt)
        elif model_name == "resnet50":
            model = resnet.PretrainedFrozenResNet50(opt)
        else:
            raise NotImplementedError("Not pretrained implemenation allowed")
        return model
    ###############

    if "cifar" in dataset:
        model_class = getattr(cifar, "ResNet")
    elif "imagenet" in dataset:
        model_class = getattr(imagenet, "ResNet")
    else:
        raise NotImplementedError("Please select the appropriate datasets (cifar100, imagenet)")
    if model_name == "resnet18":
        opt["depth"] = 18
    elif model_name == "resnet32":
        opt["depth"] = 32
    elif model_name == "resnet34":
        opt["depth"] = 34
    else:
        raise NotImplementedError("Please choose the model name in [resnet18, resnet32, resnet34]")
    model = model_class(opt)
    return model



###################################################################################################

def savefig(history, save_dir, title):
    plt.plot(history,'-o')
    plt.title("best %s:%.4f"%(title,max(history)))
    plt.xlabel('epoch')
    plt.ylabel('recall_1')
    plt.savefig(os.path.join(save_dir,"%s.png"%(title)))
    plt.clf()

def savemodel(net, epoch, c_recall_1, save_dir):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    print('Saving..')
    state = {'net': net.state_dict(),'acc': c_recall_1,'epoch': epoch,}
    torch.save(state, '%s/ckpt.pth'%(save_dir))

def load_center(load_dir, const_weight=True):
    ### load previous extracted gallery features
    prev_feats = np.load('%s/gallery_features.npy'%(load_dir))
    prev_feats_cos = F.normalize(torch.from_numpy(prev_feats),p=2,dim=1)
    ### load previous extracted labels of gallery features
    prev_labels = np.load('%s/gallery_labels.npy'%(load_dir))
    prev_labels = torch.from_numpy(prev_labels.reshape(-1))
    ### obtain mean vector as exemplars
    cat_dict = {}
    for feat, cat in zip(prev_feats_cos,prev_labels):
        cat = int(cat)
        if cat not in cat_dict:
            cat_dict[cat] = []
        cat_dict[cat].append(feat)
    mean_feats, mean_labels = [], []
    for cat in sorted(cat_dict.keys()):
        mean_feats.append(torch.mean(torch.stack(cat_dict[cat]),axis=0))
        mean_labels.append(torch.LongTensor([cat]))
    #######
    if const_weight:
        idx_weights = 1
    else:
        idx_weights = torch.Tensor([len(cat_dict[i]) for i in range(max(prev_labels)+1)]) / len(prev_labels)# * 100.0
        idx_weights = (idx_weights - min(idx_weights)) / (max(idx_weights) - min(idx_weights))#min-max normalization
    #######
    return torch.FloatTensor(torch.stack((mean_feats))), torch.stack((mean_labels)).flatten(0), idx_weights

def icarl(net,dataset,memory_size,device="cuda"):
    gallery = deepcopy(dataset)#prevent from modification
    def addNewExemplars(net,loader,device,exem_num):
        # extract embeddings per class
        net.eval()
        embedding_set, image_set = [], []
        with torch.no_grad():
            for idx, out in enumerate(loader):
                images, class_labels, name = out
                images = images.to(device)
                embeddings = net(images,True)
                embedding_set.append(embeddings)
                image_set.extend(name)
            embedding_set = torch.cat(embedding_set)
            embedding_set = F.normalize(embedding_set.detach(),p=2,dim=1).cpu().numpy()
            embedding_mean = np.mean(embedding_set, axis=0)
        embedding_set = np.array(embedding_set)
        now_embedding_sum = np.zeros_like(embedding_mean)
        # collect exemplars per class
        images = []
        for i in range(exem_num):
            x = embedding_mean - (now_embedding_sum + embedding_set) / (i + 1)
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)
            now_embedding_sum += embedding_set[index]
            images.append(image_set[index])
            embedding_set = np.delete(embedding_set, index, axis=0)
            image_set = np.delete(image_set, index, axis=0)
            if len(embedding_set) == 0: break
        return images
    # Collect exemplars
    cat_dict = {}
    for dat, tar in zip(gallery.data,gallery.targets):
        tar = int(tar)
        if tar not in cat_dict:
            cat_dict[tar] = []
        cat_dict[tar].append(dat)
    num_class = len(cat_dict.keys())
    mem_per_cls = memory_size // num_class
    exemplar_set = {}
    for class_id in cat_dict:
        gallery.data = cat_dict[class_id]
        gallery.targets = [class_id] * len(gallery.data)
        loader = DataLoader(gallery, batch_size=100, shuffle=False, num_workers=0)
        images = addNewExemplars(net,loader,device,mem_per_cls)
        exemplar_set[class_id] = images
    return exemplar_set

def weighted_mse_loss(pred, target, weight):
    return ( weight * (pred - target).pow(2).sum(1)).mean()
    