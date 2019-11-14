GPU=3 setting='--dataset cifar10_wo_val --model resnet34 --noise 0.2 --noise_type val_split_asymm'
CUDA_VISIBLE_DEVICES=$GPU python3 main_NL.py $setting
CUDA_VISIBLE_DEVICES=$GPU python3 main_PL.py $setting --max_epochs 720
CUDA_VISIBLE_DEVICES=$GPU python3 main_pseudo1.py $setting --lr 0.1 --max_epochs 480 --epoch_step 192 288
CUDA_VISIBLE_DEVICES=$GPU python3 main_pseudo2.py $setting --lr 0.1 --max_epochs 480 --epoch_step 192 288