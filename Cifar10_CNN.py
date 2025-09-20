import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np



# Download training data from open datasets.
training_data = torchvision.datasets.CIFAR10(
    root="data",
    train=True,
    download=False,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=False,
    transform=ToTensor(),
)

batch_size = 64 #no of batches

# Create data loaders.
train_set = DataLoader(training_data, batch_size=batch_size,shuffle=True)
test_set = DataLoader(test_data, batch_size=batch_size,shuffle=True)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#For data shape and type
for X, y in test_set:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Define the transformation to convert PIL Image to tensor and normalize
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# functions to show an image
batch_size_to_view=4
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(train_set)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size_to_view)))


#Define a CNN architecture

class Cifar10CNN(nn.Module):
    def __init__(self):
        super(Cifar10CNN,self).__init__()
        self.conv1=nn.Conv2d(3,6,kernel_size=5)
        self.conv2=nn.Conv2d(6,16,kernel_size=5)
        self.pool=nn.MaxPool2d(2,2)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)
        self.relu = nn.ReLU()

    def forward(self,x):
        x=self.relu(self.conv1(x))
        x=self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x=torch.flatten(x,1) #to flatten
        x=self.relu(self.fc1(x)) #start of fully connected layer
        x=self.relu(self.fc2(x))
        x=self.fc3(x)
        return x

#Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=Cifar10CNN().to(device)
loss=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.001, momentum=0.9)

#Training loop
def train(model,train_set,loss,optimizer,epochs=100):
    model.train() #setting the model to training mode
    for epoch in range (epochs):
        running_loss=0.0 #to store the loss values of each batch
        correct = 0
        total = 0
        for images, labels in train_set:
            images, labels=images.to(device), labels.to(device)
            optimizer.zero_grad()
            output=model(images)
            loss_val=loss(output,labels)
            loss_val.backward()
            optimizer.step() #Updates the model parameters using the optimizer (Adam) & Uses the gradients from loss_val.backward() to move parameters in the direction that reduces the loss
            running_loss += loss_val.item() #.item() extracts the Python float from the PyTorch tensor.
            # --- Track accuracy ---
            preds = output.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(train_set)
        epoch_acc = 100 * correct / total
        if (epoch + 1) % 10 == 0:
            print(f"For Epoch {epoch+1}, Loss: {running_loss/len(train_set):.4f},Train Acc: {epoch_acc:.2f}%") #average loss per batch
    print('Finished Training')

#Training the model
train(model, train_set, loss, optimizer)


#Testing loop
def test(model,test_set):
    model.eval() #seta model to evaluation mode
    correct=0 #counts the no of correctly predicted labels
    total_loss=0 #contain total loss across all batches
    with torch.no_grad():
        for images,labels in test_set:
            images,labels= images.to(device), labels.to(device)
            output=model(images)
            test_loss=loss(output,labels)
            total_loss += test_loss.item() #adds batch's loss to total_loss
            pred= output.argmax(dim=1, keepdims=True) #Finds the predicted class with the highest logit (probability-like score), here-dim=1 means “across classes” & keepdims=True keeps the output shape consistent for comparison
            correct +=pred.eq(labels.view_as(pred)).sum().item() #Compares predictions (pred) with true labels (labels) , counts how many predictions are correct and add the count to correct
    accuracy= 100 * correct/ len(test_set.dataset) #.dataset is used to get total number of samples (if not used then gives number of batches)
    print(f"Test loss: {total_loss/len(test_set)}, Accuracy: {accuracy: .4f}%")

#Testing the model
test(model,test_set)

