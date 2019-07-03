# Two-stream Sparse Network for Accurate Image Restoration (pytorch)

## Requirement and Dependency
Python 3.6  
Pytorch == 0.4.1  
cuda 9.0  
numpy  
skimage  
imageio  
matplotlib  
tqdm  

## Dataset
We use DIV2K as our training dataset and it can be downloaded from [here](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar) (7.1GB).  
For testing, we use Set5, Set14, BSD100, Urban100 and Manga109 as benchmark datasets for SISR.  
We use Kodak24, BSD68, and Urban100 for color and gray image denoising.  
Classic5 and LIVE1 are used for JPEG deblocking.  
All the benchmarks can be downloaded from [Baidu Pan](https://pan.baidu.com/s/19yv7fFDRXu5GrH1_0wIoOw) (905v).

## Preparation
1. Put downloaded DIV2K and benckmark in dataset/ directory.  
```sh
dataset/
	DIV2K/
	benchmark/
```
2. For color and gray image denoising, please run genDN.m and genGreyDN.m in scripts/ directory to generate noisy images for DIV2K.	  
```	sh
$ matlab -r -nodisplay scripts/genDN
$ matlab -r -nodisplay scripts/genGreyDN
```
3. For JPEG deblocking, please run genJPEG.m in scripts/ directory to generate JPEG blocking images for DIV2K.   
```sh
$ matlab -r -nodisplay scripts/genJPEG.m
```

## Pretrained models
Pretrained models can be downloaded from [Baidu Pan](https://pan.baidu.com/s/1UVlma-l_GPnreKeMWnpb0w) (ztl6).

## Super-resolution
Train/Test
```sh
$ cd TSSN/Super-resolution
$ sh demo.sh
```
To test all the benchmarks
```sh
$ cd TSSN/Super-resolution
$ ./benchmark.sh 2 tssn_x2 tssn 0   [scale model_dir model gpu_id]
```
## Denoising
Color image denoising Train/Test
```sh
$ cd TSSN/ColorDN
$ sh demo.sh
```
To test all the benchmarks
```sh
$ cd TSSN/ColorDN
$ ./benchmark.sh 50 colorDN_x50 tssn_ir 0   [degree model_dir model gpu_id]
```
Grey-scale image denoising Train/Test
```sh
$ cd TSSN/GreyDN
$ sh demo.sh
```
To test all the benchmarks
```sh
$ cd TSSN/GreyDN
$ ./benchmark.sh 50 greyDN_x50 tssn_ir 0   [degree model_dir model gpu_id]
```
## JPEG deblocking
Train/Test
```sh
$ cd TSSN/Deblock
$ sh demo.sh
```
To test all the benchmarks
```sh
$ cd TSSN/Deblock
$ ./benchmark.sh 10 deblock_x10 tssn_ir 0   [degree model_dir model gpu_id]
```
## Acknowledgements
We thank [Sanghyun Son](https://github.com/thstkdgus35) for the code base in [`EDSR-PyTorch`](https://github.com/thstkdgus35/EDSR-PyTorch).

