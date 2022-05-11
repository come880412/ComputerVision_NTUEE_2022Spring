from turtle import forward
import torchvision.models as models
import torch.nn as nn
import torch

class myLeNet(nn.Module):
    def __init__(self, num_out):
        super(myLeNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3,6,kernel_size=5, stride=1),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=2, stride=2),
                             )
        self.conv2 = nn.Sequential(nn.Conv2d(6,16,kernel_size=5),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=2, stride=2),)
        
        self.fc1 = nn.Sequential(nn.Linear(400, 120), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(120,84), nn.ReLU())
        self.fc3 = nn.Linear(84, num_out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        
        # It is important to check your shape here so that you know how manys nodes are there in first FC in_features
        #print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)        
        out = x
        return out

class residual_block(nn.Module):
    def __init__(self, in_channels):
        super(residual_block, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(in_channels))

        self.relu = nn.ReLU()
        
    def forward(self,x):
        ## TO DO ## 
        # Perform residaul network. 
        # You can refer to our ppt to build the block. It's ok if you want to do much more complicated one. 
        # i.e. pass identity to final result before activation function 
        return self.relu(self.conv1(x)) + x

        
class myResnet(nn.Module):
    def __init__(self, in_channels=3, num_out=10):
        super(myResnet, self).__init__()
        
        self.stem_conv = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        self.layer1 = residual_block(64)

        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2, 0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer3 = residual_block(64)

        self.cnn_layer4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.MaxPool2d(2, 2, 0),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.layer5 = residual_block(128)

        self.fc1 = nn.Sequential(
            nn.Linear(8192, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.fc2 = nn.Linear(512, num_out)
        
        ## TO DO ##
        # Define your own residual network here. 
        # Note: You need to use the residual block you design. It can help you a lot in training.
        # If you have no idea how to design a model, check myLeNet provided by TA above.
                
    def forward(self,x):
        ## TO DO ## 
        # Define the data path yourself by using the network member you define.
        # Note : It's important to print the shape before you flatten all of your nodes into fc layers.
        # It help you to design your model a lot. 
        x = self.stem_conv(x)
        x = self.layer1(x)
        x = self.cnn_layer2(x)
        x = self.layer3(x)
        x = self.cnn_layer4(x)
        x = self.layer5(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)

        return self.fc2(x)

