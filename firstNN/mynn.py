# Python 3.8.6
from time import process_time
start = process_time()

# torch 1.7.0
# torchvision 0.8.1
# matplotlib 3.3.3
# numpy 1.19.4
# opencv-python 4.4.0
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2

fout = open('rtx2060.txt', 'w')

# File location to save to or load from
MODEL_SAVE_PATH = './cifar_net.pth'
# Set to zero to use above saved model
TRAIN_EPOCHS = 50
# If you want to save the model at every epoch in a subfolder set to 'True'
SAVE_EPOCHS = False
# If you just want to save the final output in current folder, set to 'True'
SAVE_LAST = False
BATCH_SIZE_TRAIN = 4
BATCH_SIZE_TEST = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(device, file=fout)

print("[INFO] Done importing packages.")
print("[INFO] Done importing packages.", file=fout)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # Input: 32 x 32 x 3 = 3072

        # Kernel: 5 x 5, Stride: (1, 1), output 6 layers, padding = 0 px
        # So Output size = 28 x 28 x 6 = 4704
        ### Inputs to Conv2d: Incoming layers, outgoing layers, Frame size
        # Other values are all keyword arguments
        # Ex. to use stride = 2 and padding = 1 we would do:
        # nn.Conv2d(3, 6, 5, stride = 2, padding = 1)

        self.conv1 = nn.Conv2d(3, 64, 3)

        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(64, 128, 3)

        # self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(128, 256, 3, padding = 2)

        self.pool3 = nn.MaxPool2d(3, 3)

        # self.conv4 = nn.Conv2d(256, 512, 3)

        #28x28x256

        # self.dropout10 = nn.Dropout(p=0.1)
        # self.dropout20 = nn.Dropout(p=0.2)
        # self.dropout50 = nn.Dropout(p=0.5)

        # Activation function to use
        self.activation = F.relu

        # Batch Normalization functions
        self.batchNormalization1 = nn.BatchNorm2d(64)
        self.batchNormalization2 = nn.BatchNorm2d(128)
        self.batchNormalization3 = nn.BatchNorm2d(256)

        self.batchNormalization4 = nn.BatchNorm1d(2400)
        self.batchNormalization5 = nn.BatchNorm1d(1200)
        self.batchNormalization6 = nn.BatchNorm1d(600)
        self.batchNormalization7 = nn.BatchNorm1d(300)
        self.batchNormalization8 = nn.BatchNorm1d(120)
        self.batchNormalization9 = nn.BatchNorm1d(60)
        self.batchNormalization10 = nn.BatchNorm1d(30)

        self.fc1 = nn.Linear(6400, 2400)
        self.fc2 = nn.Linear(2400, 1200)
        self.fc3 = nn.Linear(1200, 600)
        self.fc4 = nn.Linear(600, 300)
        self.fc5 = nn.Linear(300, 120)
        self.fc6 = nn.Linear(120, 60)
        self.fc7 = nn.Linear(60, 30)
        self.fc8 = nn.Linear(30, 10)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.batchNormalization1(x)
        x = self.pool1(x)
        # x = self.dropout50(x)

        x = self.activation(self.conv2(x))
        x = self.batchNormalization2(x)
        # x = self.pool2(x)
        # x = self.dropout50(x)

        x = self.activation(self.conv3(x))
        x = self.batchNormalization3(x)
        x = self.pool3(x)
        # x = self.dropout50(x)

        # x = self.activation(self.conv4(x))

        x = x.view(-1, 6400)

        x = self.activation(self.fc1(x))
        x = self.batchNormalization4(x)

        x = self.activation(self.fc2(x))
        x = self.batchNormalization5(x)

        x = self.activation(self.fc3(x))
        x = self.batchNormalization6(x)

        x = self.activation(self.fc4(x))
        x = self.batchNormalization7(x)

        x = self.activation(self.fc5(x))
        x = self.batchNormalization8(x)

        x = self.activation(self.fc6(x))
        x = self.batchNormalization9(x)

        x = self.activation(self.fc7(x))
        x = self.batchNormalization10(x)

        x = self.fc8(x)
        return x

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

print("[INFO] Loading Traning and Test Datasets.")
print("[INFO] Loading Traning and Test Datasets.", file=fout)

# transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
transform = transforms.ToTensor()
trainset = torchvision.datasets.CIFAR10(root = './data', train = True,
    download = True, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset,
    batch_size = BATCH_SIZE_TRAIN, shuffle = True)
testset = torchvision.datasets.CIFAR10(root = './data', train = False,
    download = True, transform = transform)
testloader = torch.utils.data.DataLoader(testset,
    batch_size = BATCH_SIZE_TEST, shuffle = True)

print("[INFO] Done loading data.")
print("[INFO] Done loading data.", file=fout)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# trainiter = iter(trainloader)
# images, labels = trainiter.next()
#
# print('  '.join(f"{classes[labels[j]]}" for j in range(4)))
# imshow(torchvision.utils.make_grid(images))

net = Net()
net.to(device)
print("Network:", net)
print("Network:", net, file=fout)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

for epoch in range(TRAIN_EPOCHS):
    now = process_time()
    print(f"[TIMER] Process Time so far: {now - start:.6} seconds")
    print(f"Beginning Epoch {epoch + 1}...")
    print(f"[TIMER] Process Time so far: {now - start:.6} seconds", file=fout)
    print(f"Beginning Epoch {epoch + 1}...", file=fout, flush=True)
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        # inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 500 == 499:
            # print(f"Epoch: {epoch + 1}, Mini-Batches Processed: {i + 1:5}, Loss: {running_loss/2000:3.5}")
            print(f"Epoch: {epoch + 1}, Mini-Batches Processed: {i + 1:5}, Loss: {running_loss/2000:3.5}", file=fout, flush=True)
            running_loss = 0.0

    now = process_time()
    print(f"[TIMER] Process Time so far: {now - start:.6} seconds")
    print("Starting validation...")
    print(f"[TIMER] Process Time so far: {now - start:.6} seconds", file=fout)
    print("Starting validation...", file=fout, flush=True)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in trainloader:
            images, labels = data[0].to(device), data[1].to(device)
            # images, labels = data
            outputs = net(images)
            # For overall accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"[TRAINING] {correct} out of {total}")
    print(f"[TRAINING] {correct} out of {total}", file=fout)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            # images, labels = data
            outputs = net(images)
            # For overall accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"[VALIDATION] {correct} out of {total}")
    print(f"[VALIDATION] {correct} out of {total}", file=fout, flush=True)
    if SAVE_EPOCHS:
        torch.save(net.state_dict(), f"./saves/cifar_net_{epoch + 1}.pth")


if TRAIN_EPOCHS:
    print("[INFO] Finished training.")
    print("[INFO] Finished training.", file=fout, flush=True)
    if SAVE_LAST:
        torch.save(net.state_dict(), MODEL_SAVE_PATH)
else:
    net.load_state_dict(torch.load(MODEL_SAVE_PATH))

# testiter = iter(testloader)
# images, labels = testiter.next()
#
# print('Ground Truth:',' '.join(f"{classes[labels[j]]:5}" for j in range(4)))
#
# outputs = net(images)
#
# _, predicted = torch.max(outputs, 1)
#
# print('Predicted:',' '.join(f"{classes[predicted[j]]:5}" for j in range(4)))
# imshow(torchvision.utils.make_grid(images))

correct = 0
total = 0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        # images, labels = data
        outputs = net(images)
        # For overall accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # For class-by-class accuracy
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(BATCH_SIZE_TEST):
            label = labels[i]
            try:
                class_correct[label] += c[i].item()
            except:
                class_correct[label] += c.item()
            class_total[label] += 1

print(f"Accuracy of the network on the 10000 test items: {100 * correct / total:.4}%")
print(f"Accuracy of the network on the 10000 test items: {100 * correct / total:.4}%", file=fout)

for i in range(10):
    print(f"Accuracy of {classes[i]}: {100 * class_correct[i] / class_total[i]:.3}%")
    print(f"Accuracy of {classes[i]}: {100 * class_correct[i] / class_total[i]:.3}%", file=fout)

now = process_time()
print(f"[TIMER] Total Process Time: {now - start:.8} seconds")
print(f"[TIMER] Total Process Time: {now - start:.8} seconds", file=fout, flush=True)
fout.close()

# print(images)

# sj = cv2.imread('Serena.jpg')
# sj = cv2.resize(sj, (32, 32), interpolation = cv2.INTER_AREA)
# tmp = np.zeros((1, sj.shape[2], sj.shape[0], sj.shape[0]), dtype=float)
# for i, row in enumerate(sj):
#     for j, col in enumerate(row):
#         for k, entry in enumerate(col):
#             tmp[0][k][i][j] = entry/255.0
# sjTensor = torch.Tensor(tmp)
# outputs = net(sjTensor)
# _, predicted = torch.max(outputs, 1)
# print(classes[predicted])
