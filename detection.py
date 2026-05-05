import torch
import torchvision
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,precision_score,recall_score

#setting
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS=10


train_accs,val_accs=[],[]
train_losses,val_losses=[],[]

#data loading
train_transforms=torchvision.transforms.Compose([
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(10),
    torchvision.transforms.ColorJitter(brightness=0.2,saturation=0.2,contrast=0.2),
    torchvision.transforms.RandomCrop(224,padding=2),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485,0.456,0.406],
                                     [0.229,0.224,0.225])
])

val_transforms=torchvision.transforms.Compose([
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485,0.456,0.406],
                                     [0.229,0.224,0.225])
])

import torch 
import pydicom
import numpy as np
import pandas as pd
import PIL
import os 

def collate_fn(batch):
    images  = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets

class RSNADataset(torch.utils.data.Dataset):
    def __init__(self,img_dir,csv_path,transform=None):
        self.img_dir=img_dir
        self.transform=transform

        if csv_path:
            self.df=pd.read_csv(csv_path)
            self.df=self.df[self.df['Target']==1]
        else :
            self.images=os.listdir(img_dir)

    def __len__(self):
        if hasattr(self,'df'):
            return len(self.df)
        else:
            return len(self.images)
        
    def __getitem__(self, index):
        row=self.df.iloc[index]
        path=os.path.join(self.img_dir,row['patientId']+'.dcm')

        dcm=pydicom.dcmread(path)
        image=dcm.pixel_array

        image=np.stack([image]*3,axis=-1)
        image=PIL.Image.fromarray(image)
        if self.transform:
            image=self.transform(image)

        x=row['x']
        y=row['y']
        width=row['width']
        height=row['height']

        x1=x
        y1=y
        x2=x1+width
        y2=y1+height

        target={
            "boxes":torch.tensor([[x1,y1,x2,y2]],dtype=torch.float32),
            "labels":torch.tensor([1],dtype=torch.int64)
        }

        return image,target
    

TRAIN_DIR = r'D:\CODE\PROJECTS\rsna-chest-xray-analysis\rsna-pneumonia-detection-challenge\stage_2_train_images'
TEST_DIR  = r'D:\CODE\PROJECTS\rsna-chest-xray-analysis\rsna-pneumonia-detection-challenge\stage_2_test_images'
CSV_PATH  = r'D:\CODE\PROJECTS\rsna-chest-xray-analysis\rsna-pneumonia-detection-challenge\stage_2_train_labels.csv'


train_dataset = RSNADataset(
    img_dir = TRAIN_DIR,
    csv_path  = CSV_PATH,
    transform = train_transforms
)

total      = len(train_dataset)
train_size = int(0.8 * total)
val_size   = total - train_size

train_data, val_data = torch.utils.data.random_split(
    train_dataset, [train_size, val_size]
)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True,collate_fn = collate_fn)
val_loader   = torch.utils.data.DataLoader(val_data,   batch_size=4, shuffle=False,collate_fn = collate_fn)


test_dataset = RSNADataset(
    img_dir = TEST_DIR,
    csv_path  = None,
    transform = val_transforms
)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)

#Model
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')

for param in model.backbone.parameters():
    param.requires_grad = False

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)

model = model.to(DEVICE)

optimizer = torch.optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)
best_val_loss = float('inf')

for epoch in range(EPOCHS):

    #train
    model.train()
    running_loss = 0.0

    for images, targets in train_loader:
        images  = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)  # model returns dict of losses
        loss      = sum(loss_dict.values()) # add them all together

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

   
    model.train()  
    val_loss = 0.0

    with torch.no_grad():
        for images, targets in val_loader:
            images  = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss      = sum(loss_dict.values())
            val_loss += loss.item()

    val_loss = val_loss / len(val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')  # save best model

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss:   {val_loss:.4f}")

    train_losses.append(train_loss)
    val_losses.append(val_loss)

import matplotlib.pyplot as plt
import seaborn as sns


#training cruve
def plot_training(train_losses, val_losses):
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses,   label='Val')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('detection_training.png')
    plt.show()

plot_training(train_losses, val_losses)
