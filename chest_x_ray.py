import torch
import torchvision
import pydicom
import pandas as pd
import numpy as np
import os
import PIL
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,precision_score,recall_score

#setting
Batch=32
Architecture='vgg16'
num_classes=2
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Epochs=10

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


class RSNADataset(torch.utils.data.Dataset):
    def __init__(self,img_dir,csv_path,transform =None):
        super().__init__()

        self.img_dir=img_dir
        self.has_labels=csv_path
        self.transform = transform
        self.classes = ['Normal', 'Pneumonia']
        self.patient_ids=[]

        if self.has_labels:
            self.df = pd.read_csv(csv_path)
            self.patient_ids = self.df['patientId'].unique().tolist()
        else:
            for file in os.listdir(img_dir):
                if file.endswith('.dcm'):
                    self.patient_ids.append(file.replace('.dcm',''))
            self.grouped=None
    
    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, index):
        patient_id=self.patient_ids[index]
        path=os.path.join(self.img_dir,patient_id+'.dcm')

        #load grayscale to array to rbg
        image=pydicom.dcmread(path).pixel_array
        image=image.squeeze()
        if image.ndim==3:
            image=image[0]
        max_val=image.max()

        if max_val>0:
            image=(image/max_val*255).astype(np.uint8)
        else:
            image=(image*0).astype(np.uint8)
        image=PIL.Image.fromarray(image).convert('RGB')

        if self.transform:
            image=self.transform(image)
        
        if not self.has_labels:
            return image, patient_id
        record = self.df[self.df['patientId'] == patient_id]

        label = 1 if record['Target'].max() == 1 else 0

        return image, torch.tensor(label, dtype=torch.long)
    
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


train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_data,   batch_size=4, shuffle=False)


test_dataset = RSNADataset(
    img_dir = TEST_DIR,
    csv_path  = None,
    transform = val_transforms
)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)

#Model
def build_model(Architecture,num_classes):

    if Architecture=='resnet50':
        model=torchvision.models.resnet50(weights='IMAGENET1K_V2')

        for param in model.parameters():
            param.requires_grad=False
        
        for param in model.layer4.parameters():
            param.requires_grad=True
        
        in_features=model.fc.in_features
        model.fc=torch.nn.Sequential(
            torch.nn.Linear(in_features,512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(512,num_classes)
        )
    
    elif Architecture=='vgg16':
        model=torchvision.models.vgg16(weights='IMAGENET1K_V1')

        for param in model.features.parameters():
            param.requires_grad=False
        
        for layer in model.features[-4:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        in_features=model.classifier[6].in_features
        model.classifier[6]=torch.nn.Sequential(
            torch.nn.Linear(in_features,256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(256,num_classes)
        )

    elif Architecture=='mobilenet_v2':
        model=torchvision.models.mobilenet_v2(weights='IMAGENET1K_V2')

        for param in model.features.parameters():
            param.requires_grad=False

        for layer in model.features[-3:]:
            for param in layer.parameters():
                param.requires_grad = True

        in_features=model.classifier[1].in_features
        model.classifier[1]=torch.nn.Sequential(
            torch.nn.Linear(in_features,256),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(256,num_classes)
        )

    else :
        raise ValueError(f'Invalid Architecture{Architecture}')
    
    return model

model=build_model(Architecture,num_classes)
model=model.to(DEVICE)


#optimizer,loss and scheduler
optimizer=torch.optim.Adam(
    filter(lambda p: p.requires_grad,model.parameters()),lr=1e-4
)

criterion=torch.nn.CrossEntropyLoss()

scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    patience=2,
    factor=0.2
)

# Evaluate

def evaluate(model,loader):
    model.eval()

    all_preds=[]
    all_labels=[]

    with torch.no_grad():
        for images,labels in loader:
            images,labels=images.to(DEVICE),labels.to(DEVICE)

            outputs=model(images)
            preds=outputs.argmax(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy=accuracy_score(all_labels,all_preds)
    precision=precision_score(all_labels,all_preds)
    recall=recall_score(all_labels,all_preds)
    f1=f1_score(all_labels,all_preds)

    return accuracy,precision,recall,f1

#train
best_val_acc=0

for epoch in range(Epochs):
    model.train()
    running_loss,correct,total=0.0,0,0
    for images,labels in train_loader:
        images,labels=images.to(DEVICE),labels.to(DEVICE)
        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        predicted=outputs.argmax(1)
        correct+=(predicted==labels).sum().item()
        total+=labels.size(0)
        running_loss+=loss.item()
    
    train_acc=(correct/total)*100
    train_loss=running_loss/len(train_loader)

    model.eval()
    with torch.no_grad():
       
        running_loss,correct,total=0.0,0,0
        for images,labels in val_loader:
            images,labels=images.to(DEVICE),labels.to(DEVICE)
            outputs=model(images)
            loss=criterion(outputs,labels)

            predicted=outputs.argmax(1)
            correct+=(predicted==labels).sum().item()
            total+=labels.size(0)
            running_loss+=loss.item()
        
    val_acc=(correct/total)*100
    val_loss=running_loss/len(val_loader)
    scheduler.step(val_acc) 

    if best_val_acc<val_acc:
        best_val_acc=val_acc

    print(f"Epoch {epoch+1}/{Epochs}")
    print(f'Train Accuracy : {train_acc:.2f}%')
    print(f'Val Accuracy   : {val_acc:.2f}%')
    print(f'Train Loss     : {train_loss:.4f}')
    print(f'Val Loss       : {val_loss:.4f}')


    train_accs.append(train_acc)
    train_losses.append(train_loss)
    val_accs.append(val_acc)
    val_losses.append(val_loss)

print(f"Best Accuracy:{best_val_acc:.2f}")
print("\n----Validation Results----")
val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader)

print(f"Accuracy  : {val_acc*100:.2f}%")
print(f" Precision : {val_prec:.4f}")
print(f" Recall    : {val_rec:.4f}")
print(f" F1 Score  : {val_f1:.4f}")

import matplotlib.pyplot as plt
import seaborn as sns


#training cruve
def plot_training(train_accs,val_accs,train_losses,val_losses):
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4))

    ax1.plot(train_accs,label='Train')
    ax1.plot(val_accs,label='Val ')
    ax1.set_title('Accuracy')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("%")
    ax1.legend()

    ax2.plot(train_losses,label='Train')
    ax2.plot(val_losses,label='Val')
    ax2.set_title("loss")
    ax2.set_xlabel("Epoch")  
    ax2.set_ylabel("Loss")    
    ax2.legend()

    plt.tight_layout()
    plt.savefig('chesx_ray_training.png')
    plt.show()
    print('Saved training cruves')


#confusion matrix
def plot_confusion(model,loader,class_names):
    all_preds,all_labels=[],[]
    model.eval()
    with torch.no_grad():
        for images,labels in loader:
            images,labels=images.to(DEVICE),labels.to(DEVICE)
            outputs=model(images)
            preds=outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    cm=confusion_matrix(all_labels,all_preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    
    plt.title("Confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("confusion_matrix.png")
    plt.show()
    print("Saved confusion matrix")

#sample prediction
def plot_predicted(model, val_loader, class_names, n=8):
    model.eval()
    images, labels = next(iter(val_loader))
    n = min(n, len(images))

    with torch.no_grad():
        images_device = images.to(DEVICE)
        preds = model(images_device).argmax(1).cpu().numpy()

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

        images = (images * std + mean).clamp(0,1)
        fig, axes = plt.subplots(1, n, figsize=(2*n, 3))

        if n == 1:
            axes = [axes]

        for i in range(n):
            ax = axes[i]

            ax.imshow(images[i].permute(1,2,0))

            color = 'green' if preds[i] == labels[i].item() else 'red'

            ax.set_title(
                f'P: {class_names[preds[i]]}\nT: {class_names[labels[i].item()]}',
                color=color,
                fontsize=8
            )

            ax.axis('off')

        plt.tight_layout()
        plt.savefig('Predicted.png')
        plt.show()

        print('Saved predicted figure')

class_names=train_dataset.classes

plot_training(train_accs,val_accs,train_losses,val_losses)
plot_confusion(model,val_loader,class_names)
plot_predicted(model,val_loader,class_names)