import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, models
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
import numpy as np

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Data Preparation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet expects 224x224
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# -------------------------------
# Feature Extractor Model
# -------------------------------
def get_feature_extractor():
    model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
    # Remove the classification layer -> output 512-dim features
    model.fc = nn.Identity()
    model.eval()
    return model

# -------------------------------
# Extract features using ResNet
# -------------------------------
def extract_features(model, dataloader):
    features, labels = [], []
    with torch.no_grad():
        for images, lbls in dataloader:
            images = images.to(device)
            outputs = model(images)  # shape: (batch_size, 512)
            features.append(outputs.cpu().numpy())
            labels.append(lbls.numpy())
    return np.vstack(features), np.hstack(labels)

# -------------------------------
# Main
# -------------------------------
print("\n=== Extracting Features with ResNet18 ===")
model = get_feature_extractor()

X_train, y_train = extract_features(model, trainloader)
X_test, y_test = extract_features(model, testloader)


print("Train Features Shape:", X_train.shape)
print("Test Features Shape:", X_test.shape)

print("\nNumber of training samples:", X_train.shape[0])
print("Number of features per sample:", X_train.shape[1])

# -------------------------------
# Train SVM
# -------------------------------
print("\n=== Training SVM on ResNet Features ===")
svm_clf = SVC(kernel="rbf", C=10, random_state=42)
svm_clf.fit(X_train, y_train)

# Evaluate
y_pred = svm_clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\nSVM Accuracy on CIFAR-10 test set:", acc)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=classes))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# === Extracting Features with ResNet18 ===
# Train Features Shape: (50000, 512)
# Test Features Shape: (10000, 512)
#
# Number of training samples: 50000
# Number of features per sample: 512
#
# === Training SVM on ResNet Features ===
#
# SVM Accuracy on CIFAR-10 test set: 0.8692
#
# Classification Report:
#               precision    recall  f1-score   support
#
#        plane       0.88      0.92      0.90      1000
#          car       0.92      0.92      0.92      1000
#         bird       0.83      0.82      0.83      1000
#          cat       0.76      0.77      0.76      1000
#         deer       0.82      0.86      0.84      1000
#          dog       0.83      0.80      0.82      1000
#         frog       0.89      0.91      0.90      1000
#        horse       0.92      0.86      0.89      1000
#         ship       0.92      0.91      0.92      1000
#        truck       0.93      0.92      0.92      1000
#
#     accuracy                           0.87     10000
#    macro avg       0.87      0.87      0.87     10000
# weighted avg       0.87      0.87      0.87     10000
