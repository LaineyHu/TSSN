
###### train from scratch #######

# scale x2(patch_size 96), x3(patch_size 144), x4(patch_size 192), x8(patch_size 384)
# loss: MSE, L1, L1_softmax
python main.py --scale 2 --save tssn_x2 --model tssn --epochs 1000 --batch_size 16 --patch_size 96 --lr_decay 200 --data_test Set5 --reset --loss "1*L1_softmax" --GPU_id "0"

###### train from pretrained model #######

# restore from the training record
# python main.py --scale 2 --load tssn_x2 --model tssn --epochs 1000 --batch_size 16 --patch_size 96 --resume -1 --lr_decay 200 --data_test Set5 --loss "1*L1_softmax" --GPU_id "0"

# restore from certain checkpoint
# python main.py --scale 2 --save tssn_restore_x2 --model tssn --pre_train ../experiment/tssn_x2/model/model_best.pt --epochs 1000 --batch_size 16 --patch_size 96 --resume 0 --lr_decay 200 --data_test Set5 --loss "1*L1_softmax" --GPU_id "0"


###### test #######

# test benchmark datasets
# data_test: Set5, Set14, B100, Urban100, Manga109
# python main.py --data_test Set5 --scale 2 --pre_train ../experiment/tssn_x2/model/model_best.pt --test_only --model tssn --save_results

# test single images (put testing images in test/ directory)
# python main.py --data_test Demo --scale 2 --pre_train ../experiment/tssn_x2/model/model_best.pt --test_only --model tssn --save_results
