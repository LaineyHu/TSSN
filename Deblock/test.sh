#!/bin/bash
arr=("Set5" "Set14" "B100" "Urban100" "Manga109")
for ds in ${arr[@]}
do
    python main.py --scale 8 --save $1 --model $2 --data_test $ds --pre_train ../experiment/$1/model/model_best.pt --test_only --save_results --GPU_id $3
done
