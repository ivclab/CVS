import torch, os, pdb, argparse, sys, json
from loader import jd, dog, inature, cifar, tinyimagenet
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from evaluate import eval, compute_acc
from utils.train_utils import select_model, select_optimizer, icarl, load_center, savefig, savemodel
from copy import deepcopy
from models.normlayer import NormLayer
from kdloss import KDLoss
from mmdloss import MMD

# Load arguments
parser = argparse.ArgumentParser()
## Basic settting
parser.add_argument('--exp_dir',default='exp',help='Experimental directory')
parser.add_argument('--exp_name',default='disjoint',type=str,choices=['disjoint', 'blurry10', 'blurry30', 'general10', 'general30', 'general40'],help='Experimental setup')
parser.add_argument('--save_dir',required=True,type=str, help='Directory for the current session')
parser.add_argument('--load_dir',default='exp/prev_session',type=str, help='Directory for the previous session')
parser.add_argument('--lr', default=3e-2, type=float, help='Learning rate')
parser.add_argument('--n_epochs', default=256, type=int, help='Number of training epochs')
parser.add_argument('--arch', default='resnet18', type=str, choices=['resnet18','resnet50'], help='Network architecture')
parser.add_argument('--embed_dim', default=128, type=int, help='Embedding dimension of the network')
parser.add_argument('--kernels',default=8,type=int,help='Number of workers for pytorch dataloader')
parser.add_argument('--bs',default=16,type=int, help='Batch size')
parser.add_argument('--dataset',default='cifar100',type=str,choices=['cifar100', 'tinyimagenet', 'dog', 'inat', 'product'],help='Dataset')
parser.add_argument("--sched_name", type=str, default="cos", choices=['cos','none'], help='Learning rate scheduling')

## Session
parser.add_argument('--session_id',default=0,type=int,help='The i-th session (e.g. 0 for initial session)')
## Reindexing
parser.add_argument('--reindex',action='store_true',help='Enable feature reindexing')

## CVS
parser.add_argument('--loss_d',action='store_true',help="Enable inter-session data coherence loss")
parser.add_argument('--loss_m',action='store_true',help="Enable neighbor-session model coherence loss")
## Replay
parser.add_argument('--buffer',type=int,default=2000,help="Episodic memory size")
parser.add_argument('--replay',action='store_true',help="Enable memory replay")
## Joint Train
parser.add_argument('--jt',action='store_true',help="Enable joint training")
## BCT
parser.add_argument('--bct',action='store_true',help="Enable Backward Compatible Training (CVPR 2020)")
## LWF
parser.add_argument('--lwf',action='store_true',help="Enable Learning w/o Forgetting (TPAMI 2018)")
## MMD
parser.add_argument('--mmd',action='store_true',help="Enable Maximum Mean Discrepancy Loss (BMVC 2020)")

## Loss weighting
parser.add_argument('--alpha',default=1.0,type=float,help='Emphasize neighbor-session model coherence loss')
parser.add_argument('--beta',default=1.0,type=float,help='Emphasize inter-session data coherence loss')
args = parser.parse_args()

# Check arguments
os.makedirs(args.exp_dir, exist_ok=True)
assert args.exp_dir in args.save_dir, 'save_dir should contain %s'%(args.exp_dir)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.session_id > 0 and not os.path.exists(args.load_dir):
    print("%s not exists"%(args.load_dir))
    exit()
if not args.replay:
    args.buffer = 0

# Write experiment configuration
print("Write experiment information into %s"%(args.save_dir))
if os.path.isdir(args.save_dir):
    print("%s already exists! Exit!"%(args.save_dir))
    exit()
else:
    os.mkdir(args.save_dir)
with open(os.path.join(args.save_dir,"cmd.txt"),"w",encoding="utf-8") as f:
    f.write(' '.join(sys.argv))
with open(os.path.join(args.save_dir,"args.txt"),"w",encoding="utf-8") as f:
    json.dump(args.__dict__, f, indent=2)

# Get settings
nsloss = True# enable normalized softmax loss
nstemp = 0.05# temperature term for normalized softmax loss
if args.dataset == "product":
    class_list = [1343, 2043, 2743]
    dataset = jd.PRODUCT
    pretrain = True
elif args.dataset == "dog":
    class_list = [60, 80, 100, 120]
    dataset = dog.DOG
    pretrain = True
elif args.dataset == "inat":
    class_list = [100, 125, 150, 175, 200]
    dataset = inature.INAT
    pretrain = True
elif args.dataset == "cifar100":
    class_list = [100 for _ in range(5)] if "blur" in args.exp_name else [20, 40, 60, 80, 100]
    dataset = cifar.CIFAR100
    pretrain = False
elif args.dataset == "tinyimagenet":
    class_list = [200 for _ in range(10)] if "blur" in args.exp_name else [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    dataset = tinyimagenet.TINYIMAGENET
    pretrain = False

# Make sure INITIAL information exists
if args.session_id == 1:
    # In the second session, we load net, gallery, and testing query for the session 0 (i.e., first session)
    w = torch.load(os.path.join(args.load_dir,'ckpt.pth'))['net']
    w.pop('fc.bias',None)
    net = select_model(args.arch, args.dataset, num_classes=class_list[0], feature_size=args.embed_dim, nsloss=nsloss, pretrain=pretrain)
    net.load_state_dict(w)
    net = net.to(device)
    eval_trainset = dataset(mode="gallery", session_id=0, exp_name=args.exp_name)
    eval_trainloader = DataLoader(eval_trainset, batch_size=100, shuffle=False, num_workers=args.kernels)
    testset = dataset(mode="test",session_id=0,exp_name=args.exp_name)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=args.kernels)
    
    if args.replay:
        print("GENERATE EXEMPLARS FOR SESSION 0...")
        net.eval()
        prev_exem_names, prev_exem_targets = [], []
        exemplar_set = icarl(net,eval_trainset,args.buffer,device)
        for cat in exemplar_set:
            prev_exem_names.extend(exemplar_set[cat])
            prev_exem_targets.extend([cat]*len(exemplar_set[cat]))
        np.save(os.path.join(args.load_dir,'exem_names.npy'),np.array(prev_exem_names))
        np.save(os.path.join(args.load_dir,'exem_labels.npy'),np.array(prev_exem_targets))

    if not os.path.exists(os.path.join(args.load_dir,'gallery_features.npy')) or not os.path.exists(os.path.join(args.load_dir,"gallery_labels.npy")) or not os.path.exists(os.path.join(args.load_dir,"gallery_names.npy")):
        print("Reprocess previous gallery information due to file missing in %s"%(args.load_dir))
        net.eval()
        record, gallery_features, gallery_labels, gallery_names = eval(
            net, testloader, eval_trainloader, args.kernels,
            session_id=0,reindex=False,prev_gallery_features=None,prev_gallery_labels=None,return_curr_gallery_names=True
        )
        with open(os.path.join(args.load_dir,"result.json"),"w",encoding="utf-8") as f: json.dump(record,f)
        np.save(os.path.join(args.load_dir,'gallery_features.npy'),gallery_features)
        np.save(os.path.join(args.load_dir,'gallery_labels.npy'),gallery_labels)
        np.save(os.path.join(args.load_dir,'gallery_names.npy',gallery_names))

# Load data
print('==> Preparing data [%s]..'%(args.exp_name))
### load gallery data
eval_trainset = dataset(mode="gallery", session_id=args.session_id,joint_train=args.jt,exp_name=args.exp_name)
eval_trainloader = DataLoader(eval_trainset, batch_size=100, shuffle=False, num_workers=args.kernels)
### load testing query data
testset = dataset(mode="test",session_id=args.session_id,exp_name=args.exp_name)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=args.kernels)
### load validation query data
valset = dataset(mode="val",session_id=args.session_id,exp_name=args.exp_name)
valloader = DataLoader(valset, batch_size=100, shuffle=False, num_workers=args.kernels)
### load training data
trainset = dataset(mode="train",session_id=args.session_id,joint_train=args.jt,exp_name=args.exp_name)
prev_exem_names, prev_exem_labels = None, None
if args.replay:# add replayed data to train set
    prev_exem_names, prev_exem_labels = np.load(os.path.join(args.load_dir,'exem_names.npy')), np.load(os.path.join(args.load_dir, 'exem_labels.npy'))
    trainset.add_memory(prev_exem_names, prev_exem_labels)
trainloader = DataLoader(trainset,batch_size=args.bs,num_workers=args.kernels,shuffle=True,drop_last=True)
if args.loss_d:# enable inter-session data coherence loss
    src_centers, _, _ = load_center(args.load_dir)
    src_centers = src_centers.to(device)
if (not args.reindex) and args.session_id > 0:
    prev_gallery_features = np.load(os.path.join(args.load_dir,'gallery_features.npy'))
    prev_gallery_labels = np.load(os.path.join(args.load_dir,'gallery_labels.npy'))
    prev_gallery_names = np.load(os.path.join(args.load_dir,'gallery_names.npy')).tolist()
    print("finish loading previous gallery items.")
else:
    prev_gallery_features, prev_gallery_labels, prev_gallery_names = None, None, []





# Load net, loss, optimizer, scheduler
net = select_model(args.arch, args.dataset, num_classes=class_list[args.session_id],feature_size=args.embed_dim,nsloss=nsloss,pretrain=pretrain)
if args.session_id > 0:#initialized with the previous checkpoints
    w = torch.load(os.path.join(args.load_dir,"ckpt.pth"))['net']
    w.pop('fc.bias',None)
    old_n_classes = class_list[args.session_id-1]
    n_classes = class_list[args.session_id]
    prev_label_list = list(range(old_n_classes))
    curr_label_list = list(range(old_n_classes,n_classes))
    if "blur" in args.exp_name:#blurry
        net.load_state_dict(w,strict=False)
    else:#general, disjoint
        tmpfc = deepcopy(net.fc)
        net.fc = NormLayer(n_classes=old_n_classes,embed_dim=net.fc.weight.shape[-1])#new dummy layer
        net.load_state_dict(w,strict=False)
        # keep weights for the old classes
        tmpfc.weight.data[:old_n_classes] = net.fc.weight
        net.fc = tmpfc#copy it back
print("==> Build current model successfully..")

if (args.loss_m or args.bct or args.lwf or args.mmd) and args.session_id > 0:
    old_net = select_model(args.arch, args.dataset, num_classes=class_list[args.session_id-1],feature_size=args.embed_dim,nsloss=nsloss,pretrain=pretrain)
    old_net.load_state_dict(w,strict=False)
    old_classifier = deepcopy(old_net.fc)#deepcopy classifier from old model
    if args.lwf: net.old_classifier = deepcopy(old_classifier)#register old classifier to current model
    old_net, old_classifier = old_net.eval(), old_classifier.eval()
    for para in old_net.parameters(): para.requires_grad = False
    for para in old_classifier.parameters(): para.requires_grad = False
    old_net, old_classifier = old_net.to(device), old_classifier.to(device)
    print('==> Load old model successfully..')
net = net.to(device)
optimizer, scheduler = select_optimizer(opt_name="sgd", lr=args.lr, model=net, sched_name=args.sched_name)
triplet_criterion = nn.TripletMarginLoss(margin=0.1)
ce_criterion = nn.CrossEntropyLoss()











# Main
best_acc = 0  # best validation recall_at_1
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
e_recall_history, c_recall_history = [], [] # e for euclidean distance, c for cosine distance
for epoch in range(start_epoch, start_epoch+args.n_epochs):
    if scheduler is not None:
        # https://github.com/drimpossible/GDumb/blob/master/src/main.py
        # initialize for each task
        if epoch <= 0:  # Warm start of 1 epoch
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr * 0.1
        elif epoch == 1:  # Then set to maxlr
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr
        else:  # Aand go!
            scheduler.step()
    
    print("==================== [TRAIN] ====================")
    net.train()
    print('\nEpoch: %d' % epoch)
    sum_loss_m, sum_loss_c, sum_loss_d, sum_all_loss = 0., 0., 0., 0.
    final_iter = tqdm(trainloader)
    for batch_idx, (inputs, targets) in enumerate(final_iter):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        ################ Intra-session discrimination #######################################
        feats, logit = net(inputs)
        loss_c = ce_criterion(logit/nstemp, targets)
        feats_norm = F.normalize(feats,p=2,dim=1)

        #####################################################################################
        loss_m = torch.Tensor([0.]).to(device)
        loss_d = torch.Tensor([0.]).to(device)
        if args.session_id > 0:
            ################ Neighbor-session model coherence ###############################
            if args.loss_m or args.lwf or args.mmd:
                old_feats, old_logit = old_net(inputs)
                old_feats_norm = F.normalize(old_feats,p=2,dim=1)
            if args.loss_m:
                mask = [tl == targets for tl in targets]
                dist = torch.cdist(feats_norm,old_feats_norm)# cosine distance matrix
                hardest_neg = []# mining hard negative
                sort_pos = torch.argsort(dist,dim=1)
                n = targets.size(0)
                for idx in range(n):
                    for pos in sort_pos[idx]:
                        if mask[idx][pos] == 0:
                            neg_idx = pos
                            break
                    hardest_neg.append(old_feats_norm[neg_idx])
                hardest_neg = torch.stack(hardest_neg)
                loss_m += triplet_criterion(feats_norm, old_feats_norm, hardest_neg)
            if args.bct:
                # compute old class data classification loss by passing thru. old classifier
                valid_ind, o_target = [], []
                for ind, lab in enumerate(targets):
                    lab = int(lab)
                    if lab in prev_label_list:
                        o_target.append(lab)
                        valid_ind.append(ind)
                if len(valid_ind) > 0:
                    o_target = torch.LongTensor(o_target).to(device)
                    old_frozen_logit = old_classifier(feats)
                    loss_m += ce_criterion(old_frozen_logit[valid_ind]/nstemp, o_target)
            
            if args.lwf:
                head_y_o = net.old_classifier(feats)
                loss_m += KDLoss(head_y_o/nstemp,old_logit/nstemp)
            
            if args.mmd:
                loss_m += MMD(feats,old_feats)
                valid_ind = []
                for ind, lab in enumerate(targets):
                    lab = int(lab)
                    if lab in curr_label_list:#only images from new classes (i.e. current)
                        valid_ind.append(ind)
                if len(valid_ind) > 0:#only images from new classes (i.e. current)
                    loss_m += KDLoss(logit[valid_ind,:len(prev_label_list)]/nstemp,old_logit[valid_ind,:]/nstemp)

            
            ################ Inter-session data coherence ##################################
            if args.loss_d:
                pick = [targets < class_list[args.session_id-1]]
                prev_idx = targets[pick]
                if len(prev_idx) > 0:
                    loss_d = ((src_centers[prev_idx] - feats_norm[pick]).pow(2).sum(1)).mean()
        
        ################ TOTAL LOSS ########################################################
        loss_m   = loss_m   * args.alpha
        loss_d = loss_d * args.beta
        loss = loss_c + loss_m + loss_d

        # Update parameters
        loss.backward()
        optimizer.step()

        # Show logs
        sum_loss_c   += loss_c.item()
        sum_loss_m   += loss_m.item()
        sum_loss_d   += loss_d.item()
        sum_all_loss += loss.item()
        final_iter.set_description(
            'All_Loss: %.3f | Loss_C: %.3f | Loss_M: %.3f | Loss_D: %.3f | lr: %.6f'%(
                sum_all_loss/(batch_idx+1),
                sum_loss_c/(batch_idx+1),
                sum_loss_m/(batch_idx+1),
                sum_loss_d/(batch_idx+1),
                optimizer.param_groups[0]['lr'],
            )
        )
    
    print("==================== [ VAL ] ====================")
    net.eval()
    record, curr_gallery_features, curr_gallery_labels = eval(
        net, valloader, eval_trainloader, args.kernels,
        session_id=args.session_id,
        reindex=args.reindex,
        prev_gallery_features=prev_gallery_features,
        prev_gallery_labels=prev_gallery_labels,
    )
    e_recall_history.append(record['e_recall_1'])
    savefig(e_recall_history,args.save_dir,"validation e_recall_1")
    c_recall_history.append(record['c_recall_1'])
    savefig(c_recall_history,args.save_dir,"validation c_recall_1")
    if record['c_recall_1'] > best_acc:
        savemodel(net, epoch,record['c_recall_1'],args.save_dir)
        best_acc = record['c_recall_1']
        




# Evaluate using testing query
print("\n\n=============== [ TEST] ===============\n\n")
print("load the one with the highest validation recall@1 for final testing")
ckpt = torch.load('%s/ckpt.pth'%(args.save_dir))
net.load_state_dict(ckpt['net'],strict=False)
net.eval()
record, curr_gallery_features, curr_gallery_labels, curr_gallery_names = eval(
    net, testloader, eval_trainloader, args.kernels,
    session_id=args.session_id,
    reindex=args.reindex,
    prev_gallery_features=prev_gallery_features,
    prev_gallery_labels=prev_gallery_labels,
    return_curr_gallery_names=True,
)
if prev_gallery_features is not None:
    gallery_features = np.vstack((curr_gallery_features,prev_gallery_features))
    gallery_labels = np.vstack((curr_gallery_labels,prev_gallery_labels))
    gallery_names = curr_gallery_names + prev_gallery_names
else:#prev_gallery_features is None
    gallery_features = curr_gallery_features
    gallery_labels = curr_gallery_labels
    gallery_names = curr_gallery_names
with open(os.path.join(args.save_dir,"result.json"),"w",encoding="utf-8") as f: json.dump(record,f)
np.save(os.path.join(args.save_dir,"gallery_features.npy"),gallery_features)
np.save(os.path.join(args.save_dir,"gallery_labels.npy"),gallery_labels)
np.save(os.path.join(args.save_dir,"gallery_names.npy"),gallery_names)




# save icarl exemplar set before program exit
if args.replay:
    print("GENERATE EXEMPLARS FOR SESSION %d..."%(args.session_id))
    net.eval()
    if args.session_id > 0:
        eval_trainset.add_memory(prev_exem_names,prev_exem_labels)
    exemplar_set = icarl(net,eval_trainset,args.buffer,device)
    curr_exem_names, curr_exem_targets = [], []
    for cat in exemplar_set:
        curr_exem_names.extend(exemplar_set[cat])
        curr_exem_targets.extend([cat]*len(exemplar_set[cat]))
    np.save(os.path.join(args.save_dir,'exem_names.npy'),np.array(curr_exem_names))
    np.save(os.path.join(args.save_dir, 'exem_labels.npy'),np.array(curr_exem_targets))
