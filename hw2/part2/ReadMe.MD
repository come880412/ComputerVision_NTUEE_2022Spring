To generate the weak model for cleaning data:
python train --clean_data 0 --semi 0 --save_model ./checkpoints/weak_model
The generated.pth will be saved on the path "./checkpoints/weak_model/myResnet", and I take the model saved on epoch 7 to clean the data.

To generate the model for semi-supervised learning:
python train --clean_data 1 --semi 0 --weak_model_load ./checkpoints/weak_model/myResnet/generated.pth --save_model ./checkpoints/semi_model
The generated.pth will be saved on the path "./checkpoints/semi_model", and I take the model having the best performance to generate pseudo labels.
(You should change the name of generated.pth by yourself)

To generate the model for the final result:
python train --clean_data 1 --semi 1 --weak_model_load ./checkpoints/weak_model/myResnet/generated.pth \\
	     --semi_model_load ./checkpoints/semi_model/myResnet/model_best.pth
(You should change the name of generated.pth by yourself)


