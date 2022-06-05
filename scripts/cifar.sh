#!/bin/bash
set -e
NUM_SESSION=5
SETUP=""
METHOD="ft"
while getopts m:s: flag
do
	case "${flag}" in
		m) METHOD=${OPTARG};;
		s) SETUP=${OPTARG};;
	esac
done
SCRIPT=" train.py --exp_name $SETUP --dataset cifar100 "
EXP="exp/cifar_"$SETUP"_"$METHOD
FIRST_EXP=${EXP%_*}   # remove suffix starting with "_"



if [[ $METHOD = "cvs" ]]; then
	METHOD=" --loss_m --loss_d --alpha 10.0 --replay --buffer 2000 "
elif [[ $METHOD = "jt" ]]; then
	METHOD=" --jt --reindex "
elif [[ $METHOD = "bct" && $SETUP = "disjoint" ]]; then
	METHOD=" --bct --replay "
elif [[ $METHOD = "ft" ]]; then
	METHOD=" "
else
	METHOD=" --"$METHOD" "
fi



for ((i = 0; i < $NUM_SESSION; i++)); do
	if [[ $i = 1 ]]; then
		LOAD_DIR=$FIRST_EXP
	elif [[ $i > 1 ]]; then
		LOAD_DIR=$EXP$((i-1))
	fi
	
	if [[ $i = 0 ]]; then
		python $SCRIPT --save_dir $FIRST_EXP --session_id $i
	else
		python  $SCRIPT --load_dir $LOAD_DIR --save_dir $EXP$i $METHOD --session_id $i
	fi	
done


