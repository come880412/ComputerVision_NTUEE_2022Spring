# Final project -- Pupil Tracking
Project website: https://codalab.lisn.upsaclay.fr/competitions/5120#results

# Data root
``` 
final 
  ├── code/ 
  ├── dataset/ 
    ├── public 
    ├── HM 
    ├── KL 
    ├── non_valid_eye.txt 
    ├── train.txt 
    ├── valid.txt 
    ├── valid_eye.txt
``` 

# Computer Equipment
- System: Ubuntu20.04
- Pytorch version: Pytorch 1.7 or higher
- Python version: Python 3.7
- Testing: \
CPU: Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz \
RAM: 32 GB \
GPU: NVIDIA GeForce RTX 1080ti 12GB

- Training: \
CPU: Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz \
RAM: 256GB \
GPU: NVIDIA GeForce RTX 3090 24GB


# Get pretrained models
Please use the following commnad to get the pretrained models
``` bash
bash get_model.sh
```
- There are four models will be downloaded, including the model fine-tuning on the ganzin dataset and the model pretraining on the TEyeD dataset.

# Inference
After preparing the datasets and the pretrained models, use the following command to generate the predicted masks
``` bash
bash inference.sh
```
- Note: We set the parameters for the `public/S5` folder. If you want to test other subjects, please modify the script's config `--subject`.
- The predicted masks will be generated on the folder `public_mask` automatically.

# Training
- We have three models, including valid prediction model, segmentation w/ non-valid eyes model, and segmentation w/o non-valid eyes model. All models are pretrained on the TEyeD dataset [1], you can use the pretrained models provided by us to obtain the same results by fine-tuning the models on the ganzin dataset.
- Please use the following command to train these three models
``` bash
bash train.sh
```
- After training, models will be saved on the folder `checkpoint`.

# Reference
[1] Fuhl, W., Kasneci, G., & Kasneci, E. (2021, October). Teyed: Over 20 million real-world eye images with pupil, eyelid, and iris 2d and 3d segmentations, 2d and 3d landmarks, 3d eyeball, gaze vector, and eye movement types. In 2021 IEEE International Symposium on Mixed and Augmented Reality (ISMAR) (pp. 367-375). IEEE.
