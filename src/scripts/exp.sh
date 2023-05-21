#!/bin/bash
backbone=$1
dataset=$2
exp=$3
#nohup python pipeline.py -d synmol -b egnn --methods vgib --bseed 0 --cuda 6 --gpu_ratio 0.5 > log/egnn_synmol/vgib/vgib0.log" 2>&1 &
echo "running experiment of "$exp" on "$backbone"_"$dataset
#nohup python pipeline.py -d $dataset -b $backbone --methods $exp --bseed 0 --cuda 0 --gpu_ratio 0.5 > "log/"$backbone"_"$dataset"/"$exp"/"$exp"0.log" 2>&1 &
#nohup python pipeline.py -d $dataset -b $backbone --methods $exp --bseed 1 --cuda 1 --gpu_ratio 0.5 > "log/"$backbone"_"$dataset"/"$exp"/"$exp"1.log" 2>&1 &
#nohup python pipeline.py -d $dataset -b $backbone --methods $exp --bseed 2 --cuda 2 --gpu_ratio 0.5 > "log/"$backbone"_"$dataset"/"$exp"/"$exp"2.log" 2>&1 &
#nohup python pipeline.py -d $dataset -b $backbone --methods $exp --bseed 3 --cuda 0 --gpu_ratio 0.5 > "log/"$backbone"_"$dataset"/"$exp"/"$exp"3.log" 2>&1 &
#nohup python pipeline.py -d $dataset -b $backbone --methods $exp --bseed 4 --cuda 1 --gpu_ratio 0.5 > "log/"$backbone"_"$dataset"/"$exp"/"$exp"4.log" 2>&1 &
#nohup python pipeline.py -d $dataset -b $backbone --methods $exp --bseed 5 --cuda 2 --gpu_ratio 0.5 > "log/"$backbone"_"$dataset"/"$exp"/"$exp"5.log" 2>&1 &
nohup python pipeline.py -d $dataset -b $backbone --methods $exp --bseed 6 --cuda 1 --gpu_ratio 0.5 > "log/"$backbone"_"$dataset"/"$exp"/"$exp"6.log" 2>&1 &
nohup python pipeline.py -d $dataset -b $backbone --methods $exp --bseed 7 --cuda 2 --gpu_ratio 0.5 > "log/"$backbone"_"$dataset"/"$exp"/"$exp"7.log" 2>&1 &
nohup python pipeline.py -d $dataset -b $backbone --methods $exp --bseed 8 --cuda 1 --gpu_ratio 0.5 > "log/"$backbone"_"$dataset"/"$exp"/"$exp"8.log" 2>&1 &
nohup python pipeline.py -d $dataset -b $backbone --methods $exp --bseed 9 --cuda 2 --gpu_ratio 0.5 > "log/"$backbone"_"$dataset"/"$exp"/"$exp"9.log" 2>&1 &