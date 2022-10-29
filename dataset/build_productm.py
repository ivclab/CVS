import os, json
from glob import glob
from shutil import copyfile
#########################
srcpath = '/media/iis/8TBsinica/MyOwnComputer/Desktop/Database/JD/train2'  # depends on the unzip folder absolute path.
collection_jsonpathlist = '/home/iis/Desktop/CVS/collections/JD/*.json' # absolute path
#########################
if srcpath not in os.listdir("."):
    if not os.path.isdir(srcpath):
        print("%s not found. Please download from https://onedrive.live.com/?authkey=%21ABwlxkUe6Gyxh4s&id=1BFDBA15301520EF%211598&cid=1BFDBA15301520EF and unzip it")
        exit()

# Build target folder
os.chdir("dataset/")
dest = 'JD'
dest_train = '%s/train'%(dest)
dest_val  = '%s/test'%(dest)
if not os.path.exists(dest):
    os.mkdir(dest)
if not os.path.exists(dest_train):
    os.mkdir(dest_train)
if not os.path.exists(dest_val):
    os.mkdir(dest_val)

# Read each annotation file and generate Product-M dataset format
for json_path in glob(collection_jsonpathlist):
    with open(json_path) as f:
        data = json.load(f)
    for item in data:
        tmp = os.path.normpath(item['file_name'])
        tmp = tmp.split(os.sep)
        fname = '/'.join(tmp[1:])
        src = os.path.join(srcpath, fname)
        splittype = tmp[0]
        
        if splittype == 'train':
            classid_tar_path = os.path.join(dest_train, tmp[1])
            if not os.path.exists(classid_tar_path):
                os.mkdir(classid_tar_path)
            copyfile(src, os.path.join(dest_train, fname))
        else:
            classid_tar_path = os.path.join(dest_val, tmp[1])
            if not os.path.exists(classid_tar_path):
                os.mkdir(classid_tar_path)
            copyfile(src, os.path.join(dest_val, fname))
print("FINISH")