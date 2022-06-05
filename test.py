import torch, pdb, argparse
from utils.train_utils import select_model, select_optimizer, icarl, load_center, savefig, savemodel
from loader import jd, dog, inature, cifar, tinyimagenet
from torch.utils.data import Dataset, DataLoader
from evaluate import eval
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernels',default=8,type=int,help='Number of workers for pytorch dataloader.')
    parser.add_argument('--jt',action='store_true',help="Enable joint training")
    parser.add_argument('--reindex',action='store_true',help='Enable feature reindexing')

    parser.add_argument('--dataset',default='tinyimagenet',type=str,choices=['cifar100', 'dog','tinyimagenet','inat','product'],help='Dataset')
    parser.add_argument('--exp_name',default='blurry30',type=str,choices=['disjoint', 'blurry10', 'blurry30', 'general10', 'general30', 'general40'])
    parser.add_argument('--load_dir',default='cvs_result',type=str, help='Where to load everything for the previous session.')
    args = parser.parse_args()

    arch = "resnet18" if args.dataset in ["cifar100","tinyimagenet"] else "resnet50"
    
    # get setting
    exp_short = args.exp_name
    if args.dataset == "product":
        class_list = [1343, 2043, 2743]
        ckpt_list = ["%s/jd_%s_cvs%d/ckpt.pth"%(args.load_dir,exp_short,i) for i in range(len(class_list))]
        pretrain = True
        dataset = jd.PRODUCT
    elif args.dataset == "dog":
        class_list = [60, 80, 100, 120]
        ckpt_list = ["%s/dog_%s_cvs%d/ckpt.pth"%(args.load_dir,exp_short,i) for i in range(len(class_list))]
        pretrain = True
        dataset = dog.DOG
    elif args.dataset == "inat":
        class_list = [100,125,150,175,200]
        ckpt_list = ["%s/inat_%s_cvs%d/ckpt.pth"%(args.load_dir,exp_short,i) for i in range(len(class_list))]
        pretrain = True
        dataset = inature.INAT
    elif args.dataset == "cifar100":
        class_list = [100 for _ in range(5)] if "blur" in args.exp_name else [20,40,60,80,100]
        ckpt_list = ["%s/cifar_%s_cvs%d/ckpt.pth"%(args.load_dir,exp_short,i) for i in range(len(class_list))]
        pretrain = False
        dataset = cifar.CIFAR100
    elif args.dataset == "tinyimagenet":
        class_list = [200 for _ in range(10)] if "blur" in args.exp_name else [20,40,60,80,100,120,140,160,180,200]
        ckpt_list = ["%s/tiny_%s_cvs%d/ckpt.pth"%(args.load_dir,exp_short,i) for i in range(len(class_list))]
        pretrain = False
        dataset = tinyimagenet.TINYIMAGENET
    ckpt_list[0] = "%s/%s"%(ckpt_list[0][:ckpt_list[0].rfind("_")], ckpt_list[0].split("/")[-1])
    
    print('================================================================')
    prev_gallery_features, prev_gallery_labels, prev_gallery_names = [], [], []
    AR_1, AR_2, AR_4 = [], [], []
    for session_id, (ckpt_name, n_classes) in enumerate(zip(ckpt_list,class_list)):
        print("\t[SESSION %d]\t"%(session_id))
        ### load gallery data
        eval_trainset = dataset(mode="gallery", session_id=session_id,joint_train=args.jt,exp_name=args.exp_name)
        eval_trainloader = DataLoader(eval_trainset, batch_size=100, shuffle=False, num_workers=args.kernels)
        ### load testing query data
        testset = dataset(mode="test",session_id=session_id,exp_name=args.exp_name)
        testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=args.kernels)

        # load model with trained weights
        net = select_model(arch, args.dataset, n_classes, feature_size=128, nsloss=True, pretrain=pretrain)
        w = torch.load(ckpt_name)["net"]
        w.pop('fc.bias',None)
        w.pop('old_classifier.weight',None)
        net.load_state_dict(w)
        net = net.to(device)
        
        # evaluate
        record, curr_gallery_features, curr_gallery_labels, curr_gallery_names = eval(
            net, testloader, eval_trainloader, args.kernels,
            session_id=session_id,
            reindex=args.reindex,
            prev_gallery_features=np.array(prev_gallery_features),
            prev_gallery_labels=np.array(prev_gallery_labels),
            return_curr_gallery_names=True,
        )
        
        # accumulate
        AR_1.append(record['c_recall_1'])
        AR_2.append(record['c_recall_2'])
        AR_4.append(record['c_recall_4'])
        prev_gallery_features.extend(curr_gallery_features)
        prev_gallery_labels.extend(curr_gallery_labels)
        prev_gallery_names = curr_gallery_names + prev_gallery_names
        print('================================================================')
    print("AR@1: ",sum(AR_1)/len(AR_1),"AR@2: ",sum(AR_2)/len(AR_2),"AR@4: ",sum(AR_4)/len(AR_4))










    
