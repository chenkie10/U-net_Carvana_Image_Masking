import numpy as np
import os
import torch
import yaml
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.optim as optim
import matplotlib.pyplot as plt
from torch import nn
import cv2 as cv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import random_split
from tqdm.auto import tqdm
import imageio
import torchvision.transforms as T
from Network_parts import UNet
from torch.utils.tensorboard import SummaryWriter 

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Configs:
# archive:
dataroot = config["dataroot"]
images=config['image_path']
masks=config['mask_path']
data_dim=config['data_dim']
#dataloader
split_ratio=config['split_ratio']
batch_size=config['batch_size']

#training:
epochs=config['epochs']
lr=config['Train_step']['lr']
in_channels=config['U-net']['in_channels']
out_channels=config['U-net']['out_channels']
device=config['Train_step']['device']
criterion=config['Train_step']['criterion']
optimizer=config['Train_step']['optimizer']

 

class MakeDataset(Dataset):
    def __init__(self,images_path,masks_path,transform=None):
        self.images_path=images_path
        self.masks_path=masks_path
        self.transform=transform

        self.images_names=sorted(os.listdir(images_path))
        self.masks_names=sorted(os.listdir(masks_path))
    
    def __len__(self):
        return len(self.images_names)
    
    def __getitem__(self, index):
        image_path=os.path.join(self.images_path,self.images_names[index])

        mask_path=os.path.join(self.masks_path,self.masks_names[index])

        image=cv.imread(image_path)

        gif_reader=imageio.get_reader(mask_path)
        
        single_image=gif_reader.get_data(0)

        mask=cv.cvtColor(single_image,cv.COLOR_RGB2BGR)

        mask=mask[:,:,0]

        mask[mask==255.0]=1.0

        image=(image).astype('float32')

        mask=(mask).astype('float32')

        if (self.transform):
            augmented=self.transform(image=image,mask=mask)
            image=augmented['image']
            mask=augmented['mask']
        
        width=data_dim['width']
        height=data_dim['height']
        dimensions=(width,height)

        image=cv.resize(image,dimensions,interpolation=cv.INTER_AREA)
        mask=cv.resize(mask,dimensions,interpolation=cv.INTER_AREA)

        image=transforms.ToTensor()(image)
        mask=transforms.ToTensor()(mask)

        return image,mask
    
dataset=MakeDataset(images_path=dataroot+images,masks_path=dataroot+masks )


split=random_split(dataset,[split_ratio,1-split_ratio])

train_dataloader=DataLoader(split[0],batch_size=batch_size,shuffle=True)
test_dataloader=DataLoader(split[1],batch_size=batch_size,shuffle=True)



def dice_score(preds,targets):
    preds=F.sigmoid(preds)
    preds=(preds>0.5).float()
    score=(2.*(preds*targets).sum())/(preds+targets).sum()
    return torch.mean(score).item()

unet_model=UNet(in_channels,out_channels).to(device)
if torch.cuda.device_count()>1:
    unet_model=nn.DataParallel(unet_model,device_ids=[i for i in range(torch.cuda.device_count())])

loss_cls=getattr(nn,criterion)
loss_fn=loss_cls()
Optimizer=getattr(optim,optimizer)(params=unet_model.parameters(),lr=lr)

for epoch in tqdm(range(epochs)):
    dice=0
    val_dice=0
    for batch,(x,y) in tqdm(enumerate(train_dataloader)):
        unet_model.train()
        x=x.to(device)
        y=y.to(device)

        y_pred=unet_model(x).to(device)

        score=dice_score(y_pred,y)
        loss=loss_fn(y_pred,y)
        dice+=score

        Optimizer.zero_grad()
        
        loss.backward()

        Optimizer.step()

    unet_model.eval()
    print("Train Finished | Processing to test")
    with torch.inference_mode():
        for (x_val,y_val) in test_dataloader:
            x_val=x_val.to(device)
            y_val=y_val.to(device)
            y_val_pred=unet_model(x_val)
            val_dice+=dice_score(y_val_pred,y_val)
    
    dice/=(len(train_dataloader))
    val_dice/=len(test_dataloader)

    print(f'Epoch:{epoch+1} | Train Dice Score:{dice} | Val Dice Score={val_dice}')
    
    
    
with torch.inference_mode():
    x_test,y_test=next(iter(test_dataloader))
    x_test=x_test.to(device)
    y_pred=unet_model(x_test)
    y_pred=torch.nn.functional.sigmoid(y_pred)
    x_test=x_test.to('cpu').numpy()
    y_test=y_test.numpy()
    y_pred=y_pred.to('cpu').numpy()



