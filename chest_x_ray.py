import torch
import torchvision

#setting
Batch=32
Architecture='resnet50'
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

train_datasets = torchvision.datasets.ImageFolder(root='archive/chest_xray/train', transform=train_transforms)
val_datasets = torchvision.datasets.ImageFolder(root='archive/chest_xray/val', transform=val_transforms)

train_loader=torch.utils.data.DataLoader(dataset=train_datasets,batch_size=Batch,shuffle=True)
val_loader=torch.utils.data.DataLoader(dataset=val_datasets,shuffle=False,batch_size=Batch)


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
        
        for param in list(model.features.children())[:-4]:
            param.requires_grad=True
        
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

        for param in list(model.features.children())[:-3]:
            param.requires_grad=True

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
    filter(lambda p: p.requires_grad,model.parameters()),lr=1e-3
)

criterion=torch.nn.CrossEntropyLoss()

scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    patience=2,
    factor=0.2
)

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
        print(f"Trainning Accuracy:{train_acc:.2f}")
        print(f"Val accuracy:{val_acc:.2f}")



    train_accs.append(train_acc)
    train_losses.append(train_loss)
    val_accs.append(val_acc)
    val_losses.append(val_loss)

