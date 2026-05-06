import torch
import torchvision
import pandas as pd
import os
import pydicom
import numpy as np
import PIL

#setting
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS=5
IMG_SIZE=256


#data Agumentation
train_transforms=torchvision.transforms.Compose([
    torchvision.transforms.Resize((IMG_SIZE,IMG_SIZE)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5],[0.5])
])

val_transforms=torchvision.transforms.Compose([
    torchvision.transforms.Resize((IMG_SIZE,IMG_SIZE)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5],[0.5])
])

#Dataset
class RSNADataset(torch.utils.data.Dataset):
    def __init__(self,img_dir,csv_path,transform=None):
        super().__init__()

        self.img_dir=img_dir
        self.csv_path=csv_path
        self.transform=transform
        df=pd.read_csv(csv_path)

        self.patient_ids=df['patient_ids']
        self.grouped=df.groupby('patient_ids')

    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, index):
        patient_ids=self.patient_ids[index]
        path=os.path.join(self.img_dir,patient_ids+'.dcm')

        image=pydicom.dcmread(path).pixel_array
        image=image.squeeze()
        if image.ndim==3:
            image=image[0]
        max_val=image.max()

        if max_val>0:
            image=(image/max_val*255).astype(np.uint8)
        else:
            image=(image*0).astype(np.uint8)
        image=PIL.Image.fromarray(image).covert('L')

        mask=np.zeros((1024,1024),dtype=np.uint8)
        record=self.grouped.get_group(patient_ids)

        for _,row in record.iterrows:
            x=int(row['x'])
            y=int(row['y'])
            w=int(row['width'])
            h=int(row['height'])

            mask[y:y+h,x:x+w]=1

        mask = PIL.Image.fromarray(mask)
        mask = mask.resize((IMG_SIZE, IMG_SIZE), PIL.Image.NEAREST)
        mask = np.array(mask, dtype=np.float32)
 
        if self.transform:
            image = self.transform(image)
 
        mask = torch.from_numpy(mask).unsqueeze(0)
 
        return image, mask

TRAIN_DIR = r'D:\CODE\PROJECTS\rsna-chest-xray-analysis\rsna-pneumonia-detection-challenge\stage_2_train_images'
CSV_PATH  = r'D:\CODE\PROJECTS\rsna-chest-xray-analysis\rsna-pneumonia-detection-challenge\stage_2_train_labels.csv'

#Train and val split
train_dataset = RSNADataset(
    img_dir  = TRAIN_DIR,
    csv_path = CSV_PATH,
    transform = train_transforms
)
 
total      = len(train_dataset)
train_size = int(0.8 * total)
val_size   = total - train_size
 
train_data, val_data = torch.utils.data.random_split(
    train_dataset, [train_size, val_size]
)
 
train_loader = torch.utils.data.DataLoader(train_data, batch_size=8,  shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_data,   batch_size=8,  shuffle=False)


#U-Net Architecture
class DoubleConv(torch.nn.Module):
    def __init__(self, in_ch,out_ch):
        super().__init__()

        self.Double=torch.nn.Sequential(
            torch.nn.Conv2d(in_ch,out_ch,kernel_size=3,padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(out_ch,out_ch,kernel_size=3,padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.Double(x)

#Encoder Block
class EncoderBlock(torch.nn.Module):
    def __init__(self, in_ch,out_ch):
        super().__init__()

        self.conv=DoubleConv(in_ch,out_ch)
        self.pool=torch.nn.MaxPool2d(2)
    
    def forward(self,x):
        feature=self.conv(x)
        pooled=self.pool(feature)

        return feature,pooled
    
#Decoder
class DecoderBock(torch.nn.Module):
    def __init__(self, in_ch,out_ch):
        super().__init__()

        self.upsample=torch.nn.ConvTranspose2d(in_ch,out_ch,kernel_size=2,stride=2)
        self.conv=DoubleConv(2*out_ch,out_ch)

    def forward(self,skip,x):

        x=self.upsample(x)
        x=torch.cat([skip,x],dim=1)
        return self.conv(x)

class UNet(torch.nn.Module):
    def __init__(self,in_ch=1,num_classes=1):
        super().__init__()

        self.enc1=EncoderBlock(in_ch,64)
        self.enc2=EncoderBlock(64,128)
        self.enc3=EncoderBlock(128,256)
        self.enc4=EncoderBlock(256,512)
        self.bottleneck=DoubleConv(512,1024)
        self.dec4=DecoderBock(1024,512)
        self.dec3=DecoderBock(512,256)
        self.dec2=DecoderBock(256,128)
        self.dec1=DecoderBock(128,64)

        self.out=torch.nn.Conv2d(64,num_classes,1)

    def forward(self,x):

        skip1,x=self.enc1(x)
        skip2,x=self.enc2(x)
        skip3,x=self.enc3(x)
        skip4,x=self.enc4(x)
        x=self.bottleneck(x)
        x=self.dec1(x,skip1)
        x=self.dec2(x,skip2)
        x=self.dec3(x,skip3)
        x=self.dec4(x,skip4)
        return self.out(x)
    
class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth=smooth

    def forward(self,logits,targets):
        probs=torch.sigmoid(logits)
        probs=probs.view(probs.size(0),-1)
        targets=targets.view(targets.size(0),-1)
        inter=(probs*targets).sum(dim=1)
        dice=(2.0*inter+self.smooth)/(
            probs.sum(dim=1)+targets.sum(dim=1)+self.smooth
        )
        return 1.0-dice.mean()

#Dice score
def dice_score(logits,targets,threshold=0.5,smooth=1e-5):
    probs=(torch.sigmoid(torch)>threshold).float()
    probs=probs.view(probs.size(0),-1)
    targets=targets.view(targets.size(0),-1)
    inter=(probs*targets).sum(dim=1)
    dice=(2.0*inter+smooth)/(
        probs.sum(dim=1)+targets.sum(dim=1)+smooth
    )
    return dice.mean().item()

#Model
model=UNet(in_ch=1,num_classes=1).to(DEVICE)

#Optimize,Loss and scheduler
bce_fn=torch.nn.BCEWithLogitsLoss()
dice_fn=DiceLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,mode='min',patience=3,factor=0.5
)

best_val_loss = float('inf')