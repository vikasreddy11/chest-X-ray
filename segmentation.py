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
train_losses, val_losses = [], []
train_dices,  val_dices  = [], []


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

        self.patient_ids = df['patientId'].unique()
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
        image=PIL.Image.fromarray(image).convert('L')

        mask=np.zeros((1024,1024),dtype=np.uint8)
        record=self.grouped.get_group(patient_ids)

        for _,row in record.iterrows():
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
        x=self.dec1(x,skip4)
        x=self.dec2(x,skip3)
        x=self.dec3(x,skip2)
        x=self.dec4(x,skip1)
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
    probs=(torch.sigmoid(logits)>threshold).float()
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

#training Epoch
for epoch in range(EPOCHS):
    running_loss=0
    running_dice=0

    model.train()
    for image,mask in train_loader:
        image,mask=image.to(DEVICE),mask.to(DEVICE)

        logits=model(image)
        loss=bce_fn(logits,mask)+dice_fn(logits,mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_dice+=dice_score(logits.detach(),mask)
        running_loss+=loss.item()

    train_loss=running_loss/len(train_loader)
    train_dice=running_dice/len(train_loader)

    model.eval()
    val_loss=0
    val_dice=0

    with torch.no_grad():
        for image,mask in val_loader:
            image,mask=image.to(DEVICE),mask.to(DEVICE)

            logits=model(image)
            loss=bce_fn(logits,mask)+dice_fn(logits,mask)

            val_loss+=loss.item()
            val_dice+=dice_score(logits.detach(),mask)

    val_loss=val_loss/len(val_loader)
    val_dice=val_dice/len(val_loader)

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
 
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
    print(f"Val Loss:   {val_loss:.4f} | Val Dice:   {val_dice:.4f}")
 
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_dices.append(train_dice)
    val_dices.append(val_dice)
 
print(f"\nBest Val Loss: {best_val_loss:.4f}")

#plots

import matplotlib.pyplot as plt

def plot_training(train_losses,val_losses,train_dices,val_dices):
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4))

    ax1.plot(train_losses,label="Train")
    ax1.plot(val_losses,label="Val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(train_dices,label="Train")
    ax2.plot(val_dices,label="Val")
    ax2.set_title("Dice Score")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice loss")
    ax2.legend()

    plt.tight_layout()
    plt.savefig('segmentation_training.png')
    plt.show()
    print('Saved training curves')


def plot_predictions(model,val_loader,n=4):
    model.eval()
    images, masks = next(iter(val_loader))

    with torch.no_grad():
        images_gpu = images.to(DEVICE)
        logits     = model(images_gpu)
        preds      = (torch.sigmoid(logits) > 0.5).float().cpu()
 
    fig, axes = plt.subplots(3, n, figsize=(3*n, 9))
 
    for i in range(n):
        img = images[i].squeeze().numpy()
        img = (img * 0.5 + 0.5)  
        img = np.clip(img, 0, 1)
 
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title('X-Ray')
        axes[0, i].axis('off')
 
        axes[1, i].imshow(masks[i].squeeze().numpy(), cmap='gray')
        axes[1, i].set_title('True Mask')
        axes[1, i].axis('off')
 
        axes[2, i].imshow(preds[i].squeeze().numpy(), cmap='gray')
        axes[2, i].set_title('Predicted')
        axes[2, i].axis('off')
 
    plt.tight_layout()
    plt.savefig('segmentation_predictions.png')
    plt.show()
    print('Saved predictions')
 
 
plot_training(train_losses, val_losses, train_dices, val_dices)
plot_predictions(model, val_loader)