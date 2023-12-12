import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transform

device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transform.Compose([transform.ToTensor(), transform.Normalize((0.5,),(0.5,))])

trainset= torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

testset= torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1,32,3)

        self.pool = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(32,64,4)

        self.fc1= nn.Linear(64*5*5,128)

        self.fc2 = nn.Linear(128,10)
    def forward(self,x):
        x= self.pool(nn.ReLU()(self.conv1(x)))
        x= self.pool(nn.ReLU()(self.conv2(x)))

        x= x.view(-1,64*5*5)

        x = nn.ReLU()(self.fc1(x))
        x= self.fc2(x)

        return x

net = Net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(5):
    running_loss=0.0
    for i, data in enumerate(trainloader,0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        running_loss+= loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d loss: %.3f' % (epoch+1,(i+1)*4,running_loss/2000))
            running_loss = 0.0

print("training bitti")

PATH= './mnist_net.pth'
torch.save(net.state_dict(), PATH)

correct=0
total=0
with torch.no_grad():
    for data in testloader:
        images, labels=data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)

        _,predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()

print('Dogruluk degeri(10.000 foto uzerinde)=  %% %d' % (100*correct/total))














