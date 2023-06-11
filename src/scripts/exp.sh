#!/bin/bash
exp=$1
backbone=$2
dataset=$3
method=$4
# bash scripts/exp.sh pipeline.py egnn synmol all_attribution
#nohup python pipeline.py -d synmol -b egnn --methods vgib --bseed 0 --cuda 6 --gpu_ratio 0.5 > log/egnn_synmol/vgib/vgib0.log" 2>&1 &
echo "running experiment of "$exp" on "$backbone"_"$dataset with $method
nohup python pipeline.py -d $dataset -b $backbone --methods $method --bseed 0 --cuda 0 --gpu_ratio 0.3 > "log/"$backbone"_"$dataset"/"$method"/"$method"0.log" 2>&1 &
nohup python pipeline.py -d $dataset -b $backbone --methods $method --bseed 1 --cuda 1 --gpu_ratio 0.3 > "log/"$backbone"_"$dataset"/"$method"/"$method"1.log" 2>&1 &
nohup python pipeline.py -d $dataset -b $backbone --methods $method --bseed 2 --cuda 2 --gpu_ratio 0.3 > "log/"$backbone"_"$dataset"/"$method"/"$method"2.log" 2>&1 &
nohup python pipeline.py -d $dataset -b $backbone --methods $method --bseed 3 --cuda 3 --gpu_ratio 0.3 > "log/"$backbone"_"$dataset"/"$method"/"$method"3.log" 2>&1 &
nohup python pipeline.py -d $dataset -b $backbone --methods $method --bseed 4 --cuda 4 --gpu_ratio 0.3 > "log/"$backbone"_"$dataset"/"$method"/"$method"4.log" 2>&1 &
#nohup python pipeline.py -d $dataset -b $backbone --methods $method --bseed 5 --cuda 0 --gpu_ratio 0.3 > "log/"$backbone"_"$dataset"/"$method"/"$method"5.log" 2>&1 &
#nohup python pipeline.py -d $dataset -b $backbone --methods $method --bseed 6 --cuda 1 --gpu_ratio 0.3 > "log/"$backbone"_"$dataset"/"$method"/"$method"6.log" 2>&1 &
#nohup python pipeline.py -d $dataset -b $backbone --methods $method --bseed 7 --cuda 2 --gpu_ratio 0.3 > "log/"$backbone"_"$dataset"/"$method"/"$method"7.log" 2>&1 &
#nohup python pipeline.py -d $dataset -b $backbone --methods $method --bseed 8 --cuda 3 --gpu_ratio 0.3 > "log/"$backbone"_"$dataset"/"$method"/"$method"8.log" 2>&1 &
#nohup python pipeline.py -d $dataset -b $backbone --methods $method --bseed 9 --cuda 4 --gpu_ratio 0.3 > "log/"$backbone"_"$dataset"/"$method"/"$method"9.log" 2>&1 &