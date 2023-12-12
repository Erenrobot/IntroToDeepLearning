import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)

X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)


train_data = TensorDataset(X_train,y_train)
train_loader = DataLoader(dataset= train_data, batch_size=10, shuffle=True)

class IrisNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(IrisNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, num_classes)
        self.relu=nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.softmax(out)
        return out

model = IrisNet(input_size=4, hidden_size=10, num_classes=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epoch=50

for epoch in range(num_epoch):
    for i, (inputs, labels) in enumerate(train_loader):
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if(epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epoch}], Loss {loss.item():.4f}')

with torch.no_grad():
    y_pred = model(X_test)
    _, predicted = torch.max(y_pred,1)
    accuracy = (predicted == y_test).float().mean()
    print(f'Accuracy: {accuracy:.2f}')
















