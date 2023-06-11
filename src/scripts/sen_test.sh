#!/bin/bash
backbone=$1
dataset=$2
metric=$3
# bash scripts/sen_test.sh egnn synmol auc_fid_all
echo "running experiment of test_sensitivity with $metric on "$backbone"_"$dataset
nohup python test_sensitivity.py -d $dataset -b $backbone --cuda 0 --metric $metric --exp_method label_perturb0 --clf_method erm > "log/"$backbone"_"$dataset"/erm_sens0.log" 2>&1 &
nohup python test_sensitivity.py -d $dataset -b $backbone --cuda 0 --metric $metric --exp_method label_perturb1 --clf_method erm > "log/"$backbone"_"$dataset"/erm_sens1.log" 2>&1 &

nohup python test_sensitivity.py -d $dataset -b $backbone --cuda 1 --metric $metric --exp_method label_perturb0 --clf_method lri_bern > /dev/null 2>&1 &
nohup python test_sensitivity.py -d $dataset -b $backbone --cuda 1 --metric $metric --exp_method label_perturb1 --clf_method lri_bern > /dev/null 2>&1 &

nohup python test_sensitivity.py -d $dataset -b $backbone --cuda 2 --metric $metric --exp_method label_perturb0 --clf_method lri_gaussian > /dev/null 2>&1 &
nohup python test_sensitivity.py -d $dataset -b $backbone --cuda 2 --metric $metric --exp_method label_perturb1 --clf_method lri_gaussian > /dev/null 2>&1 &

nohup python test_sensitivity.py -d $dataset -b $backbone --cuda 1 --metric $metric --exp_method label_perturb0 --clf_method vgib > /dev/null 2>&1 &
nohup python test_sensitivity.py -d $dataset -b $backbone --cuda 2 --metric $metric --exp_method label_perturb1 --clf_method vgib > /dev/null 2>&1 &
