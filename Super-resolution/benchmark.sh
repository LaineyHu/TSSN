#!/bin/bash
arr=("Set5" "Set14" "BSD100" "Urban100" "Manga109")
for ds in ${arr[@]}
do
    python main.py --scale $1 --save $2 --model $3 --data_test $ds --pre_train ../experiment/$2/model/model_best.pt --test_only --save_results --GPU_id $4
done
