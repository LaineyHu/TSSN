#!/bin/bash
arr=("classic5" "LIVE1")
for ds in ${arr[@]}
do
    python main.py --scale 1 --save $1 --model $2 --data_test $ds --pre_train ../experiment/$1/model/model_best.pt --test_only --save_results --GPU_id $3
done
