set -e
echo "CIFAR100 DISJOINT"
python test.py --dataset cifar100 --exp_name disjoint
echo "CIFAR100 BLURRY10"
python test.py --dataset cifar100 --exp_name blurry10
echo "CIFAR100 GENERAL10"
python test.py --dataset cifar100 --exp_name general10
echo "TINY IMAGENET DISJOINT"
python test.py --dataset tinyimagenet --exp_name disjoint
echo "TINY IMAGENET BLURRY30"
python test.py --dataset tinyimagenet --exp_name blurry30
echo "TINY IMAGENET GENERAL30"
python test.py --dataset tinyimagenet --exp_name general30