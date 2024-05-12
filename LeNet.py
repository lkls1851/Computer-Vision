import os
import glob
import numpy as np
import cv2
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision 
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

train_path='Data_Hub/intel_image_classification/seg_train/seg_train'
val_path='Data_Hub/intel_image_classification/seg_test/seg_test'
INPUT_SIZE=32
EPOCHS=20
BATCH_SIZE=32


def class_encoder(class_name):
    class_names=[]
    for class_file in os.listdir(train_path):
        if class_file not in class_names:
            class_names.append(class_file)
    
    return class_names.index(class_name)


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.LazyConv2d(out_channels=6, kernel_size=5, padding=2)
        self.conv2=nn.LazyConv2d(out_channels=16, kernel_size=5)
        self.pool1=nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool2=nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1=nn.LazyLinear(out_features=120)
        self.fc2=nn.LazyLinear(out_features=84)
        self.fc3=nn.LazyLinear(out_features=6)
        self.flatten=nn.Flatten()

    def forward(self, x):
        x=x.float()
        x=self.conv1(x)
        x=torch.sigmoid(x)
        x=self.pool1(x)
        x=self.conv2(x)
        x=torch.sigmoid(x)
        x=self.pool2(x)
        x=self.flatten(x)
        x=self.fc1(x)
        x=torch.sigmoid(x)
        x=self.fc2(x)
        x=torch.sigmoid(x)
        x=self.fc3(x)
        x=torch.sigmoid(x)

        return x


class MyDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.image_path=path
        files=os.listdir(self.image_path)
        self.data=[]
        for f in files:
            class_name=str(f)
            class_path=os.path.join(train_path, f)
            for file in os.listdir(class_path):
                self.data.append([os.path.join(class_path, file), class_name])
        
        self.transform=transforms.Compose([transforms.Resize((INPUT_SIZE,INPUT_SIZE))])
    
    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):
        image_path=self.data[idx][0]
        image=cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = image.reshape(image.shape[1], image.shape[1], 1)
        print("SHAPE IS: ", image.shape)        

        image=torch.from_numpy(image)
        image=self.transform(image)
        class_name=self.data[idx][1]

        return image, class_encoder(class_name)
    
train_dataloader=DataLoader(dataset=MyDataset(path=train_path), batch_size=BATCH_SIZE, shuffle=True)
val_dataloader=DataLoader(dataset=MyDataset(path=val_path), batch_size=BATCH_SIZE, shuffle=False)


model=LeNet()
device=('cuda' if torch.cuda.is_available() else 'cpu')
criterion=nn.BCELoss()
optimiser=optim.Adam(model.parameters(), lr=1e-4)

losses=[]
accuracies=[]
val_losses=[]
val_accuracies=[]

model.to(device)

for epoch in range(EPOCHS):
    for i, (image, label) in enumerate(train_dataloader):
        image=image.to(device)
        label=label.to(device)
        label=f.one_hot(label, num_classes=6).float()
        output=model(image)
        loss=criterion(output, label)
        loss.backward()
        optimiser.step()
        _, predicted=torch.max(output.data, 1)
    acc=(predicted==label).sum().item()/label.size(0)
    accuracies.append(acc)
    losses.append(loss.item())

    val_loss=0.0
    val_accuracy=0.0

    with torch.no_grad():
        for i, (img, lab) in enumerate(val_dataloader):
            lab=lab.to(device)
            img=img.to(device)
            out=model(img)
            lab=f.one_hot(lab, num_classes=6).float()

            loss=criterion(out, lab)
            val_loss+=loss.item()
            _, pred=torch.max(out.data, 1)

        acc_val=(pred==lab).sum().item()/lab.size(0)
        val_accuracy+=acc_val
        val_accuracies.append(acc_val)
        val_losses.append(val_loss)


    print('Epoch [{}/{}],Loss:{:.4f},Validation Loss:{:.4f},Accuracy:{:.2f},Validation Accuracy:{:.2f}'.format( 
        epoch+1, EPOCHS, loss.item(), val_loss, acc ,val_accuracy))     









