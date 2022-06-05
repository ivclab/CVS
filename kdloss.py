import torch, pdb
import torch.nn as nn
# See https://blog.csdn.net/Hungryof/article/details/93738717 for more clear view
# Modified from https://github.com/ngailapdi/LWF/blob/7b7a87db3d80e25cfe8590b4135a5ee25c1f2707/model.py#L16
def KDLoss(logits, labels, temperature=2.0):
    assert not labels.requires_grad, "output from teacher(old task model) should not contain gradients"
    # Compute the log of softmax values
    outputs = torch.log_softmax(logits/temperature,dim=1)
    labels  = torch.softmax(labels/temperature,dim=1)
    outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
    outputs = -torch.mean(outputs, dim=0, keepdim=False)
    return outputs
if __name__ == "__main__":
    class net(nn.Module):
        def __init__(self):
            super(net,self).__init__()
            self.layer = nn.Linear(5,3)
        def forward(self,x):
            return self.layer(x)
    student = net()
    teacher = net()
    device = "cuda" if torch.cuda.is_available else "cpu"
    student = student.to(device)
    student.train()
    teacher = teacher.to(device)
    teacher.train()
    bs = 1
    feature_dim = 5
    input = torch.rand(bs,feature_dim).to(device)
    logits = student(input)
    with torch.no_grad():
        labels = teacher(input)
    dist_loss = KDLoss(logits,labels)
    dist_loss.backward()