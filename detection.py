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
    

TRAIN_DIR = r'D:\CODE\PROJECTS\CHESTxray\rsna-pneumonia-detection-challenge\stage_2_train_images'
TEST_DIR  = r'D:\CODE\PROJECTS\CHESTxray\rsna-pneumonia-detection-challenge\stage_2_test_images'
CSV_PATH  = r'D:\CODE\PROJECTS\CHESTxray\rsna-pneumonia-detection-challenge\stage_2_train_labels.csv'


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
