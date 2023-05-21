#!/bin/bash
#backbone=$1
#dataset=$2
#exp=$3
#echo "running experiment of "$exp" on "$backbone"_"$dataset
#nohup python test_sensitivity.py -b egnn -d synmol --exp_method label_perturb0 --clf_method ciga > log/egnn_synmol/label0_ciga.test 2>&1 &
#nohup python test_sensitivity.py -b egnn -d synmol --exp_method label_perturb1 --clf_method ciga > log/egnn_synmol/label1_ciga.test 2>&1 &
#nohup python test_sensitivity.py -b egnn -d synmol --exp_method label_perturb0 --clf_method vgib > log/egnn_synmol/label0_vgib.test 2>&1 &
nohup python test_sensitivity.py -b egnn -d synmol --exp_method label_perturb1 --clf_method vgib > log/egnn_synmol/label1_vgib.new 2>&1 &
#nohup python test_sensitivity.py -b egnn -d synmol --exp_method label_perturb0 --clf_method lri_gaussian > log/egnn_synmol/label0_lri_gaussian.test 2>&1 &
#nohup python test_sensitivity.py -b egnn -d synmol --exp_method label_perturb1 --clf_method lri_gaussian > log/egnn_synmol/label1_lri_gaussian.test 2>&1 &
#nohup python test_sensitivity.py -b egnn -d synmol --exp_method label_perturb0 --clf_method lri_bern > log/egnn_synmol/label0_lri_bern.test 2>&1 &
#nohup python test_sensitivity.py -b egnn -d synmol --exp_method label_perturb1 --clf_method lri_bern > log/egnn_synmol/label1_lri_bern.test 2>&1 &
#nohup python test_sensitivity.py -b egnn -d synmol --exp_method label_perturb0 --clf_method erm > log/egnn_synmol/label0_erm.test 2>&1 &
#nohup python test_sensitivity.py -b egnn -d synmol --exp_method label_perturb1 --clf_method erm > log/egnn_synmol/label1_erm.test 2>&1 &

#nohup python test_sensitivity.py -d $dataset -b $backbone --exp_method $exp_method --clf_method $clf_method --gpu_ratio 0.5 > "log/"$backbone"_"$dataset"/"$exp"/"$exp"0.log" 2>&1 &
#nohup python test_sensitivity.py -d $dataset -b $backbone --methods $exp --bseed 1 --cuda 1 --gpu_ratio 0.5 > "log/"$backbone"_"$dataset"/"$exp"/"$exp"1.log" 2>&1 &
#nohup python test_sensitivity.py -d $dataset -b $backbone --methods $exp --bseed 2 --cuda 2 --gpu_ratio 0.5 > "log/"$backbone"_"$dataset"/"$exp"/"$exp"2.log" 2>&1 &
#nohup python test_sensitivity.py -d $dataset -b $backbone --methods $exp --bseed 3 --cuda 3 --gpu_ratio 0.5 > "log/"$backbone"_"$dataset"/"$exp"/"$exp"3.log" 2>&1 &
#nohup python test_sensitivity.py -d $dataset -b $backbone --methods $exp --bseed 4 --cuda 4 --gpu_ratio 0.5 > "log/"$backbone"_"$dataset"/"$exp"/"$exp"4.log" 2>&1 &
#nohup python test_sensitivity.py -d $dataset -b $backbone --methods $exp --bseed 5 --cuda 0 --gpu_ratio 0.5 > "log/"$backbone"_"$dataset"/"$exp"/"$exp"5.log" 2>&1 &
#nohup python test_sensitivity.py -d $dataset -b $backbone --methods $exp --bseed 6 --cuda 1 --gpu_ratio 0.5 > "log/"$backbone"_"$dataset"/"$exp"/"$exp"6.log" 2>&1 &
#nohup python pipeline.py -d $dataset -b $backbone --methods $exp --bseed 7 --cuda 2 --gpu_ratio 0.5 > "log/"$backbone"_"$dataset"/"$exp"/"$exp"7.log" 2>&1 &
#nohup python pipeline.py -d $dataset -b $backbone --methods $exp --bseed 8 --cuda 3 --gpu_ratio 0.5 > "log/"$backbone"_"$dataset"/"$exp"/"$exp"8.log" 2>&1 &
#nohup python pipeline.py -d $dataset -b $backbone --methods $exp --bseed 9 --cuda 4 --gpu_ratio 0.5 > "log/"$backbone"_"$dataset"/"$exp"/"$exp"9.log" 2>&1 &