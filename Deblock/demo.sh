###### train from scratch #######
python main.py --scale 1 --degree 40 --save deblock_x40 --model tssn_ir --epochs 1000 --batch_size 16 --patch_size 48 --lr_decay 200 --data_test classic5 --reset --loss "1*L1_softmax" --GPU_id "0"

###### train from pretrained model #######

# restore from the training record
# python main.py --scale 1 --degree 40 --load deblock_x40 --model tssn_ir --epochs 1000 --batch_size 16 --patch_size 48 --resume -1 --lr_decay 200 --data_test classic5 --loss "1*L1_softmax" --GPU_id "0"

# restore from certain checkpoint
# python main.py --scale 1 --degree 40 --save deblock_restore_x40 --model tssn_ir --pre_train ../experiment/deblock_x40/model/model_best.pt --epochs 1000 --batch_size 16 --patch_size 48 --resume 0 --lr_decay 200 --data_test classic5 --loss "1*L1_softmax" --GPU_id "0" --reset


###### test #######

# test benchmark datasets
# data_test: classic5 LIVE1
# python main.py --data_test classic5 --scale 1 --degree 40 --pre_train ../experiment/deblock_x40/model/model_best.pt --test_only --model tssn_ir --save_results

# test single images (put testing images in test/ directory)
# python main.py --data_test Demo --scale 1 --degree 40 --pre_train ../experiment/deblock_x40/model/model_best.pt --test_only --model tssn_ir --save_results
