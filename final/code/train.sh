# TEyeD_seg

# python3 train_seg.py --epochs 20 --warmup_epochs 2 --model deeplab --save_model ./checkpoint/TEyeD \
#                --batch_size 48 --workers 8 --scheduler linearwarmup --device 1 --val_freq 1 \
#                --root ../dataset --dataset TEyeD --lr 2e-4 --adam --img_size 320 200

# python3 train_seg.py --epochs 10 --warmup_epochs 2 --model segformer --save_model ./checkpoint/TEyeD \
#                --batch_size 112 --workers 8 --scheduler linearwarmup --device 0,1 --val_freq 1 \
#                --root ../dataset --img_size 320 200 --dataset TEyeD --adam --lr 2e-4

# python3 train_seg.py --epochs 16 --warmup_epochs 3 --model sfnet --save_model ./checkpoint/TEyeD \
#                --batch_size 128 --workers 8 --scheduler linearwarmup --device 0 --val_freq 2 \
#                --root ../dataset --img_size 320 200 --dataset TEyeD --adam --lr 2e-4

# python3 train_seg.py --epochs 4 --model lawin --save_model ./checkpoint/TEyeD \
#                --batch_size 16 --workers 8 --scheduler cosine --device 0 --val_freq 2 \
#                --root ../dataset --img_size 384 256 --dataset TEyeD --adam --lr 1e-4


# Ganzin valid
python3 train_valid.py --epochs 30 --warmup_epochs 3  --saved_model ./checkpoint/ganzin/valid \
               --batch_size 96 --workers 4 --scheduler linearwarmup --device 0 --lr 7e-4 \
               --root ../dataset --img_size 640 480 --load ./valid_pretrain.pth \
               --threshold 0.4
 
 # Ganzin_seg
CUDA_VISIBLE_DEVICES=1 python3 train_seg.py --epochs 40 --warmup_epochs 4 --model deeplab --save_model ./checkpoint/ganzin_validonly \
               --batch_size 16 --workers 8 --scheduler linearwarmup --device 1 --val_freq 2 \
               --root ../dataset --img_size 640 480 --dataset ganzin --lr 3e-4 --adam\
               --load ./deeplab_pretrain.pth \
               --train_txt ../dataset/train.txt --valid_txt ../dataset/valid.txt

CUDA_VISIBLE_DEVICES=0 python3 train_seg.py --epochs 40 --warmup_epochs 4 --model deeplab --save_model ./checkpoint/ganzin_alldata \
               --batch_size 16 --workers 8 --scheduler linearwarmup --device 0 --val_freq 2 \
               --root ../dataset --img_size 640 480 --dataset ganzin --lr 3e-4 --adam\
               --load ./deeplab_pretrain.pth --non_valid \
               --train_txt ../dataset/train.txt --valid_txt ../dataset/valid.txt

# python3 train_seg.py --epochs 40 --warmup_epochs 4 --model segformer --save_model ./checkpoint/ganzin \
#                --batch_size 16 --workers 8 --scheduler linearwarmup --device 1 --val_freq 2 \
#                --root ../dataset --img_size 640 480 --dataset ganzin \
#                --load ./checkpoint/TEyeD/fcn/model_best.pth \
#                --train_txt ../dataset/train90.txt --valid_txt ../dataset/valid90.txt

# python3 train_seg.py --epochs 40 --warmup_epochs 4 --model segformer --save_model ./checkpoint/ganzin_alldata \
#                --batch_size 16 --workers 8 --scheduler linearwarmup --device 1 --val_freq 2 \
#                --root ../dataset --img_size 640 480 --dataset ganzin \
#                --load ./checkpoint/TEyeD/fcn/model_best.pth --non_valid \
#                --train_txt ../dataset/train90.txt --valid_txt ../dataset/valid90.txt

# python3 train_seg.py --epochs 40 --warmup_epochs 4 --model sfnet --save_model ./checkpoint/ganzin \
#                --batch_size 16 --workers 8 --scheduler linearwarmup --device 1 --val_freq 2 \
#                --root ../dataset --img_size 640 480 --dataset ganzin \
#                --load ./checkpoint/TEyeD/fcn/model_best.pth \
#                --train_txt ../dataset/train90.txt --valid_txt ../dataset/valid90.txt

# python3 train_seg.py --epochs 40 --warmup_epochs 4 --model sfnet --save_model ./checkpoint/ganzin_alldata \
#                --batch_size 16 --workers 8 --scheduler linearwarmup --device 1 --val_freq 2 \
#                --root ../dataset --img_size 640 480 --dataset ganzin \
#                --load ./checkpoint/TEyeD/fcn/model_best.pth --non_valid \
#                --train_txt ../dataset/train90.txt --valid_txt ../dataset/valid90.txt