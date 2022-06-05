import shutil, os

srcpath="tiny-imagenet-200"

if srcpath not in os.listdir("."):
    os.chdir("dataset/")
    if not os.path.isdir(srcpath):
        print("%s not found. Please download from http://cs231n.stanford.edu/tiny-imagenet-200.zip and unzip it")
        exit()


# Read validation annotations
name_to_id = {}
with open(os.path.join(srcpath,"val","val_annotations.txt")) as f:
    for line in f:
        if not line: continue
        param = line.split("\t")
        name, imgid = param[0], param[1]
        name_to_id[name] = imgid

# Build target folder
dest = "tinyimagenet200"
for imgid in name_to_id.values():
    os.makedirs("%s/train/%s"%(dest,imgid),exist_ok=True)
    os.makedirs("%s/val/%s"%(dest,imgid),exist_ok=True)

# Move validation set
for name in os.listdir(os.path.join(srcpath,"val","images")):
    shutil.move(os.path.join(srcpath,"val","images",name), os.path.join(dest,"val",name_to_id[name],name))

# Move training set
for imgid in os.listdir(os.path.join(srcpath,"train")):
    for name in os.listdir(os.path.join(srcpath,"train",imgid,"images")):
        src = os.path.join(srcpath,"train",imgid,"images",name)
        dst = os.path.join(dest,"train",imgid,name)
        shutil.move(src,dst)

# clean
shutil.rmtree(srcpath)
