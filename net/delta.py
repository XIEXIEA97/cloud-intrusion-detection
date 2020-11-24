import torch

import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch import optim
from torch.autograd import Variable

import matplotlib.pyplot as plt

CLASS_NUM = 5

class NN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(6000*2000*1, 40)
        self.hidden2 = nn.Linear(40, 200)
        self.dp = nn.Dropout(0)
        self.actv = nn.Sigmoid()
        self.output = nn.Linear(200, CLASS_NUM)

    def forward(self, x):
        x = x.view(1, -1)
        x = self.hidden(x)
        x = self.dp(x)
        x = self.actv(x)
        
        x = self.hidden2(x)
        x = self.dp(x)
        x = self.actv(x)
        x = self.output(x)
        return x
# class NN_Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.hidden = nn.Linear(6000*2000*1, 40)
#         self.dp = nn.Dropout(0.5)
#         self.actv = nn.Sigmoid()
#         self.output = nn.Linear(40, CLASS_NUM)

#     def forward(self, x):
#         x = x.view(1, -1)
#         x = self.hidden(x)
#         x = self.dp(x)
#         x = self.actv(x)
#         x = self.output(x)
#         return x

if __name__ == '__main__':
    plt.ion()
    if_cuda = True

    transform = transforms.Compose([
        # transforms.Resize(),
        transforms.Grayscale(),
        transforms.ToTensor(),      # 转换为Tensor&归一化至[0,1]
        transforms.Normalize(mean = [.5], std = [.5]) # 标准化
    ])

    train_dataset = ImageFolder('data2/train/', transform = transform)
    val_dataset = ImageFolder('data2/val/', transform = transform)
    test_dataset = ImageFolder('data2/test/', transform = transform) + val_dataset
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)
    print(train_dataset.class_to_idx)

model = NN_Model()
    epo = 20

    if if_cuda: 
        model.cuda()
        loss_function = nn.CrossEntropyLoss().cuda()
    else:
        loss_function = nn.CrossEntropyLoss()

#     optimizer = optim.SGD(model.parameters(), lr=1e-4, weight_decay= 1e-5, momentum = 0.1, nesterov = True)
    lr = 4e-4
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    train_loss_list, valid_loss_list, train_acc_list, valid_acc_list = [], [], [], []

    for epoch in range(epo):
        train_loss, valid_loss = [], []
        train_acc, valid_acc = [], []

        model.train()
        for data, target in train_loader:
            if if_cuda: 
                data = data.cuda()
            optimizer.zero_grad()
            output = model(data)
            if if_cuda: 
                target = target.cuda()
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            pre = output.argmax(dim=1)
            train_acc.append(pre.item() == target.item())

        model.eval()
        for data, target in val_loader:
            if if_cuda: 
                data = data.cuda()
            output = model(data)
            if if_cuda: 
                target = target.cuda()
            loss = loss_function(output, target)
            valid_loss.append(loss.item())
            pre = output.argmax(dim=1)
            valid_acc.append(pre.item() == target.item())
        print ("Epoch:", epoch, "Training Loss: ", np.mean(train_loss), "Valid Loss: ", np.mean(valid_loss), "Train Accuracy: ", np.mean(train_acc), "Valid Accuracy: ", np.mean(valid_acc))
        train_loss_list.append(np.mean(train_loss))
        valid_loss_list.append(np.mean(valid_loss))
        train_acc_list.append(np.mean(train_acc))
        valid_acc_list.append(np.mean(valid_acc))
        if epoch > 5:
            if epoch % 5 == 0:
                lr /= 2
                print("lr/=2, lr= ", lr)
                optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

x = range(epo)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
l1, = plt.plot(x, train_loss_list, '-b')
l2, = plt.plot(x, valid_loss_list, '-r')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend([l1, l2], ['train loss', 'validation loss'], loc='best')
plt.yticks([0, 2])


plt.subplot(1, 2, 2)
l1, = plt.plot(x, train_acc_list, '-b')
l2, = plt.plot(x, valid_acc_list, '-r')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend([l1, l2], ['train acc', 'validation acc'], loc='best')
plt.yticks([0, 1])

plt.show()

accuracy = []
for data, target in val_loader:
    if if_cuda: 
        data = data.cuda()
        target = target.cuda()
    output = model(data.float())
    pre = output.argmax(dim=1)
    print(pre.item(), target.item(), pre.item() == target.item())
    accuracy.append(pre.item() == target.item())
print(np.mean(accuracy))