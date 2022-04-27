import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import dataset
from model import MobileNetV3

MAX_EPOCH = 50
BATCH_SIZE = 64
LR = 0.0001
log_interval = 3
val_interval = 1

split_dir = os.path.join(".", "data", "splitData")
train_dir = os.path.join(split_dir, "train")
valid_dir = os.path.join(split_dir, "valid")

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_data = dataset.FlowerDataset(data_dir=train_dir, transform=train_transform)
valid_data = dataset.FlowerDataset(data_dir=valid_dir, transform=valid_transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True)

net = MobileNetV3.MobileNetV3_Large(17)
if torch.cuda.is_available():
    net.cuda()

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.99))

train_curve = list()
valid_curve = list()
net.train()
accurancy_global = 0.0
for epoch in range(MAX_EPOCH):
    loss_mean = 0.
    correct = 0.
    total = 0.
    running_loss = 0.0

    for i, data in enumerate(train_loader):
        img, label = data
        img = Variable(img)
        label = Variable(label)
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()
        out = net(img)
        optimizer.zero_grad()
        loss = criterion(out, label)

        print_loss = loss.data.item()

        loss.backward()
        optimizer.step()
        if (i+1) % log_interval == 0:
            print('epoch:{},loss:{:.4f}'.format(epoch+1, loss.data.item()))
        _, predicted = torch.max(out.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum()
    print("=====================================================")
    accurancy = correct / total
    if accurancy > accurancy_global:
        torch.save(net.state_dict(), './weights/best.pkl')
        print("准确率由：", accurancy_global, "上升至：", accurancy, "已更新并保存权值为weights/best.pkl")
        accurancy_global = accurancy
    print('第%d个epoch的识别准确率为：%d%%' % (epoch + 1, 100*accurancy))
torch.save(net.state_dict(), './weights/last.pkl')
print("训练完毕，权重已保存为：weights/last.pkl")
