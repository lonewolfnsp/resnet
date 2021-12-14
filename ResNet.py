import torch 
import torch.nn as nn 
import torch.nn.functional as F 

"""
for CNN, output dimension is calculated as: (W+2P-F)//s where W is width, P is padding, F is filter size and S is stride
note that division is integer division. Eg: 7//2=3. 

this implementation is based on ResNet paper. ResNet 16 and 32 skips 2 layers. 50, 101 and 152 skip 3 layers
"""
# used by ResNets below 50 layers. skip 2 layers
class IdentityBlock2(nn.Module):
    def __init__(self, inChannels,  filters):
        super().__init__() 
        F1, F2 = filters
        self.trunk=nn.Sequential(
            nn.Conv2d(inChannels, F1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(F1),
            nn.ReLU(),
            nn.Conv2d(F1, F2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(F2),            
            )                
    
    def forward(self, x):        
        return F.relu(self.trunk(x)+x)
        


#used by ResNet 50 onwards 
class IdentityBlock3(nn.Module):
    def __init__(self, inChannels,  filters):
        super().__init__() 
        F1, F2, F3 = filters
        self.trunk=nn.Sequential(
            nn.Conv2d(inChannels, F1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F1),
            nn.ReLU(),
            nn.Conv2d(F1, F2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(F2),
            nn.ReLU(),
            nn.Conv2d(F2, F3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F3)
            )                
    
    def forward(self, x):        
        return F.relu(self.trunk(x)+x)


class ConvBlock2(nn.Module):
    def __init__(self, inChannels,filters):
        super().__init__() 
        F1, F2 = filters
        self.trunk=nn.Sequential(
            nn.Conv2d(inChannels, F1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(F1),
            nn.ReLU(), # width= ((W-1)/s)+1
            nn.Conv2d(F1, F2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(F2),            
            )
        self.res=nn.Conv2d(inChannels, F2, kernel_size=3, stride=2, padding=1)        
    
    def forward(self, x):        
        return F.relu(self.trunk(x)+self.res(x))


class ConvBlock3(nn.Module):
    def __init__(self, inChannels,filters):
        super().__init__() 
        F1, F2, F3 = filters
        self.trunk=nn.Sequential(
            nn.Conv2d(inChannels, F1, kernel_size=1, stride=2, padding=0),
            nn.BatchNorm2d(F1),
            nn.ReLU(), # width= ((W-1)/s)+1
            nn.Conv2d(F1, F2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(F2),
            nn.ReLU(), 
            nn.Conv2d(F2, F3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F3)
            )
        self.res=nn.Conv2d(inChannels, F3, kernel_size=1, stride=2, padding=0)
        
    
    def forward(self, x):        
        return F.relu(self.trunk(x)+self.res(x))


# Residual Network implementation based on paper
from ResNet import *

class ResNet(nn.Module):
    
    def __init__(self,image, num_classes, types):        
        super().__init__()  
        resnettypes=[18,32,50,101,152]
        layers=resnettypes[types]
        channels=image.shape[0]
        width=image.shape[1]
        resnetLayers=[]
        final_width=width        
        n=1
        
        #stage 1: width//2        
        if width<64:
            resnetLayers.append(nn.Conv2d(channels, 64, kernel_size=3, stride=2, padding=1))  #width/2 
            resnetLayers.append(nn.Identity()) #width//4
            final_width=width//2
        else:
                resnetLayers.append(nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3))  #width/2 
                resnetLayers.append(nn.MaxPool2d(3, stride=2)) #width//4
                final_width=width//4 -1
        channels=64        
        print(f'stage 1: width={final_width}, channel={channels}')
                
        # stage 2 wudth//4 NO CHANGE IN DIMENSION, only channels increased
        if layers <50: #resnet 18 and 32            
            resnetLayers.append(IdentityBlock2(64,[64,64]))            
            if layers>18: 
                resnetLayers.append(IdentityBlock2(64,[64,64])) #resnet 32
            channels=64
        else: #resnet 50 onwards in channel:64 out channel: 256            
            resnetLayers.append(nn.Conv2d(64,256,kernel_size=3, stride=1, padding=1))
            resnetLayers.append(IdentityBlock3(256,[64,64,256]))
            resnetLayers.append(IdentityBlock3(256,[64,64,256]))
            channels=256        
        print(f'stage 2: width={final_width}, channel={channels}')
        
        
        #stage 3 width//8  
        if layers<50: #resnet 18 & 32 in channel: 64 out channel:128
            resnetLayers.append(ConvBlock2(64,[128,128]))            
            n=2 if layers>18 else 1            
            for i in range(n):
                resnetLayers.append(IdentityBlock2(128,[128,128]))            
            channels=128
        else: #resnet 50 onwards in channel:256 out channel:512
            resnetLayers.append(ConvBlock3(256,[128,128,512]))
            n=7 if layers>50 else 3            
            for i in range(n):
                resnetLayers.append(IdentityBlock3(512,[128,128,512]))                            
            channels=512
        final_width=width//8 if width>63 else width//4
        print(f'stage 3: width={final_width}, channel={channels}')
                
        
        #stage 4  width//16
        if layers <50: #resnet 18 & 32 in channel:128 out channel:256
            resnetLayers.append(ConvBlock2(128,[256,256]))
            n= 5 if layers>18 else 1                    
            for i in range(n):
                resnetLayers.append(IdentityBlock2(256,[256,256]))                
            channels=256
        else: #resnet 50 onwards in channnel:512 out channel:1024
            resnetLayers.append(ConvBlock3(512,[256,256,1024]))
            if layers==50: n=5
            elif layers==101: n=22
            else: n=35
            for i in range(n):
                resnetLayers.append(IdentityBlock3(1024,[256,256,1024]))
            channels=1024
        final_width=width//16 if width>63 else width//8
        print(f'stage 4: width={final_width}, channel={channels}')
        
        
        #stage 5 only if widrh>=64        
        if layers<50:
            resnetLayers.append(ConvBlock2(256,[512,512]))
            resnetLayers.append(IdentityBlock2(512,[512,512]))
            if layers>18:
                resnetLayers.append(IdentityBlock2(512,[512,512]))
            channels=512
        else:
            resnetLayers.append(ConvBlock3(1024,[512,512,2048]))
            resnetLayers.append(IdentityBlock3(2048,[512,512,2048]))
            resnetLayers.append(IdentityBlock3(2048,[512,512,2048]))        
            channels=2048
        final_width=width//32 if width>63 else width//16
        print(f'stage 5: width={final_width}, channel={channels}')
        
        
        # Linear classification layer
        resnetLayers.append(nn.AvgPool2d(2))
        final_width=final_width//2 
        print(f'linear layer after maxpool 2x2: width={final_width}, channel={channels}')        
        
        resnetLayers.append(nn.Flatten())                                                
        resnetLayers.append(nn.Linear((final_width**2)*channels, 512))
        resnetLayers.append(nn.ReLU())        
        resnetLayers.append( nn.Linear(512, num_classes))                                                
        
                        
        # Now we combine the layers into a single network
        self.resnet=nn.Sequential(*resnetLayers)
              
    
    def forward(self, x):
        return self.resnet(x)


