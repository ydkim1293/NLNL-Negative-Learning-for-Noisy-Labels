# NLNL-Negative-Learning-for-Noisy-Labels

Pytorch implementation for paper NLNL: Negative Learning for Noisy Labels, ICCV 2019

Paper: https://arxiv.org/abs/1908.07387

## Requirements
- python3
- pytorch
- matplotlib

## Generating noisy data
```
python3 noise_generator.py --noise_type val_split_symm_exc
```

## Start training
Simply run sh file: run.sh
```
GPU=0 setting='--dataset cifar10_wo_val --model resnet34 --noise 0.2 --noise_type val_split_symm_exc'
CUDA_VISIBLE_DEVICES=$GPU python3 main_NL.py $setting
CUDA_VISIBLE_DEVICES=$GPU python3 main_PL.py $setting --max_epochs 720
CUDA_VISIBLE_DEVICES=$GPU python3 main_pseudo1.py $setting --lr 0.1 --max_epochs 480 --epoch_step 192 288
CUDA_VISIBLE_DEVICES=$GPU python3 main_pseudo2.py $setting --lr 0.1 --max_epochs 480 --epoch_step 192 288
```
### Citation
```
@inproceedings{kim2019nlnl,
  title={Nlnl: Negative learning for noisy labels},
  author={Kim, Youngdong and Yim, Junho and Yun, Juseung and Kim, Junmo},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={101--110},
  year={2019}
}
```
