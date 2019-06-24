
###### train from scratch #######

# scale x2(patch_size 96), x3(patch_size 144), x4(patch_size 192), x8(patch_size 384)
# loss: MSE, L1, L1_softmax
python main.py --scale 2 --save tssn_x2 --model tssn --epochs 1000 --batch_size 16 --patch_size 96 --lr_decay 200 --data_test Set5 --reset --loss "1*L1_softmax" --GPU_id "0"


###### train from pretrained model #######

# restore from the training record
# python main.py --scale 2 --load tssn_x2 --model tssn --epochs 1000 --batch_size 16 --patch_size 96 --resume -1 --lr_decay 200 --data_test Set5 --loss "1*L1_softmax" --GPU_id "0"

###### test #######

#python main.py --data_test Set5 --save tsrn_GE_x2 --scale 2 --pre_train ../experiment/tsrn_GE_x2/model/model_best.pt --test_only --model tsrn --save_results
# python main.py --data_test Set5 --save RDN_sparse_mb123_se_fusion_x2 --scale 2 --pre_train ../experiment/RDN_sparse_mb123_se_fusion_x2/model/model_best.pt --test_only --ext sep_reset --model RDN_sparse_mb123_se_fusion --save_results --dir_data ..
# RDN BI model (x3)
#python3.6 main.py --scale 3 --save RDN_D16C8G64_BIx3 --model RDN --epochs 200 --batch_size 16 --data_range 801-805 --patch_size 96 --reset
# RDN BI model (x4)
#python3.6 main.py --scale 4 --save RDN_D16C8G64_BIx4 --model RDN --epochs 200 --batch_size 16 --data_range 801-805 --patch_size 128 --reset

# RCAN_BIX2_G10R20P48, input=48x48, output=96x96
# pretrained model can be downloaded from https://www.dropbox.com/s/mjbcqkd4nwhr6nu/models_ECCV2018RCAN.zip?dl=0
#python main.py --template RCAN --save RCAN_BIX2_G10R20P48 --scale 2 --reset --save_results --patch_size 96
# RCAN_BIX3_G10R20P48, input=48x48, output=144x144
#python main.py --template RCAN --save RCAN_BIX3_G10R20P48 --scale 3 --reset --save_results --patch_size 144 --pre_train ../experiment/model/RCAN_BIX2.pt
# RCAN_BIX4_G10R20P48, input=48x48, output=192x192
#python main.py --template RCAN --save RCAN_BIX4_G10R20P48 --scale 4 --reset --save_results --patch_size 192 --pre_train ../experiment/model/RCAN_BIX2.pt
# RCAN_BIX8_G10R20P48, input=48x48, output=384x384
#python main.py --template RCAN --save RCAN_BIX8_G10R20P48 --scale 8 --reset --save_results --patch_size 384 --pre_train ../experiment/model/RCAN_BIX2.pt

