# EDSR baseline model (x2) + JPEG augmentation
#python main.py --model EDSR --scale 2 --patch_size 96 --save edsr_baseline_x2 --reset
#python main.py --model EDSR --scale 2 --patch_size 96 --save edsr_baseline_x2 --reset --data_train DIV2K+DIV2K-Q75 --data_test DIV2K+DIV2K-Q75

# EDSR baseline model (x3) - from EDSR baseline model (x2)
#python main.py --model EDSR --scale 3 --patch_size 144 --save edsr_baseline_x3 --reset --pre_train [pre-trained EDSR_baseline_x2 model dir]

# EDSR baseline model (x4) - from EDSR baseline model (x2)
#python main.py --model EDSR --scale 4 --save edsr_baseline_x4 --reset --pre_train [pre-trained EDSR_baseline_x2 model dir]

# EDSR in the paper (x2)
#python main.py --model EDSR --scale 2 --save edsr_x2 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset

# EDSR in the paper (x3) - from EDSR (x2)
#python main.py --model EDSR --scale 3 --save edsr_x3 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train [pre-trained EDSR model dir]

# EDSR in the paper (x4) - from EDSR (x2)
#python main.py --model EDSR --scale 4 --save edsr_x4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train [pre-trained EDSR_x2 model dir]

# MDSR baseline model
#python main.py --template MDSR --model MDSR --scale 2+3+4 --save MDSR_baseline --reset --save_models

# MDSR in the paper
#python main.py --template MDSR --model MDSR --scale 2+3+4 --n_resblocks 80 --save MDSR --reset --save_models

# Standard benchmarks (Ex. EDSR_baseline_x4)
#python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 4 --pre_train download --test_only --self_ensemble

#python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train download --test_only --self_ensemble

# Test your own images
#python main.py --model tsrn --data_test Demo --scale 4 --pre_train ../experiment/tsrn_x4/model/model_best.pt --test_only --save_results

# Advanced - Test with JPEG images 
#python main.py --model MDSR --data_test Demo --scale 2+3+4 --pre_train download --test_only --save_results

# Advanced - Training with adversarial loss
#python main.py --template GAN --scale 4 --save edsr_gan --reset --patch_size 96 --loss 5*VGG54+0.15*GAN --pre_train download

# RDN BI model (x2)
# python main.py --scale 1 --load car_x40 --model tsrn_ir --epochs 1000 --batch_size 16 --patch_size 48 --lr_decay 200 --GPU_id "0" --data_test classic5 --degree 40 --loss "1*L1_softmax" --resume -1
# python main.py --scale 2 --load tsrnl2_x2 --model tsrn --data_test Set5 --GPU_id "1" --epochs 1000 --batch_size 16 --patch_size 96 --lr_decay 200 --resume -1 --loss "1*MSE"
#python main.py --scale 2 --save TSRN_x2 --model tsrn --epochs 1000 --batch_size 16 --patch_size 96 --lr_decay 200 --GPU_id "3" --resume 0 --reset --loss 1*L1_softmax --data_test Set5
#python main.py --scale 2 --save RDN_sparse_b2_se_fusion_10+18_x2 --model RDN_sparse_b2_se_fusion --epochs 1000 --batch_size 16 --patch_size 64 --lr_decay 200 --GPU_id "3" --resume 0
#python main.py --scale 4 --save RDN_sparse_b2_se_fusion_x4 --model RDN_sparse_b2_se_fusion --epochs 650 --batch_size 16 --patch_size 128 --decay 200 --GPU_id "2" --resume 0 --reset --ext sep
# python main.py --scale 2 --load RDN_epoch150_x2 --model RDN --epochs 650 --batch_size 16 --patch_size 64 --decay 200 --GPU_id "3" --resume -1
# test
# visualize python main.py --batch_size 1 --scale 2 --save RDN_x2 --model RDN --data_test Set5 --pre_train ../experiment/RDN_x2/model/model_best.pt --test_only --ext sep_reset --save_results --GPU_id "2"
# python main.py --scale 1 --save greynoise_x70 --model tsrn_ir --data_test BSD68 --pre_train ../experiment/greynoise_x70/model/model_latest.pt --test_only --GPU_id "0" --degree 70 --batch_size 1 --n_colors 1
python main.py --scale 1 --save car_x40 --model tsrn_ir --data_test LIVE1 --pre_train ../experiment/car_x40/model/model_latest.pt --test_only --GPU_id "0" --degree 40 --batch_size 1 --n_colors 3
# python main.py --scale 2 --save se_x2 --model tsrn --data_test Manga109 --pre_train ../experiment/se_x2/model/model_best.pt --test_only --GPU_id "2"
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

