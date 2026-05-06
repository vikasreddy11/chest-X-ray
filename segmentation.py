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
train_transform=torchvision.transforms.Compose([
    torchvision.transforms.Resize((IMG_SIZE,IMG_SIZE)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5],[0.5])
])

val_transform=torchvision.transforms.Compose([
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

