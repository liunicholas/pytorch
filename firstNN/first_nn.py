# Python 3.8.6
from time import process_time

start = process_time()

# torch 1.7.0
# torchvision 0.8.1
# matplotlib 3.3.3
# numpy 1.19.4
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# File location to save to or load from
MODEL_SAVE_PATH = './saves/cifar_net_56.pth'
# Set to zero to use above saved model
TRAIN_EPOCHS = 0
# If you want to save the model at every epoch set to 'True'
SAVE_EPOCHS = False

print("[INFO] Done importing packages.")

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel
        # 6 output channels
        # 3x3 square convolution

        # Kernel:
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a 2x2 window
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

print("[INFO] Loading Traning and Test Datasets.")

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
trainset = torchvision.datasets.CIFAR10(root = './data', train = True,
    download = True, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 4,
    shuffle = True)
testset = torchvision.datasets.CIFAR10(root = './data', train = False,
    download = True, transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size = 4,
    shuffle = True)

print("[INFO] Done loading data.")

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# trainiter = iter(trainloader)
# images, labels = trainiter.next()
#
# print('  '.join(f"{classes[labels[j]]}" for j in range(4)))
# imshow(torchvision.utils.make_grid(images))

net = Net()
print("Network:", net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

for epoch in range(TRAIN_EPOCHS):
    now = process_time()
    print(f"[TIMER] Process Time so far: {now - start:.6} seconds")
    print(f"Beginning Epoch {epoch + 1}...")
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print(f"Epoch: {epoch + 1}, Mini-Batches Processed: {i + 1:5}, Loss: {running_loss/2000:3.5}")
            running_loss = 0.0

    now = process_time()
    print(f"[TIMER] Process Time so far: {now - start:.6} seconds")
    print("Starting validation...")
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            # For overall accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"[VALIDATION] {correct} out of {total}")
    if SAVE_EPOCHS:
        torch.save(net.state_dict(), f"./saves/cifar_net_{epoch + 1}.pth")


if TRAIN_EPOCHS:
    print("[INFO] Finished training.")
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
        images, labels = data
        outputs = net(images)
        # For overall accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # For class-by-class accuracy
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

print(f"Accuracy of the network on the 10000 test items: {100 * correct / total:.4}%")

for i in range(10):
    print(f"Accuracy of {classes[i]}: {100 * class_correct[i] / class_total[i]:.3}%")

now = process_time()
print(f"[TIMER] Total Process Time: {now - start:.8} seconds")
