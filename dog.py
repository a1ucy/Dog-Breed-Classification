# 狗狗/猫咪/……品种识别器：创建一个狗狗品种识别器，通过给定的狗狗图片识别狗/猫……的品种
import os
import torch
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import f1_score

# constants
train_path = './train/'
val_path = './val/'
num_breeds = len(os.listdir(train_path))
num_epochs = 10
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Transformations
train_transform = v2.Compose([
    v2.Resize(256),
    v2.RandomRotation(15),
    v2.ColorJitter(),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.CenterCrop(224),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = v2.Compose([
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load Dataset
train_dataset = ImageFolder(root=train_path, transform=train_transform)
val_dataset = ImageFolder(root=val_path, transform=val_transform)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

model = resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, num_breeds)
model_transfer_grad_paramaters = filter(lambda p: p.requires_grad, model.parameters())

# Loss and Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model_transfer_grad_paramaters, lr=0.001)

model.to(device)

# Early stopping parameters
best_val_loss = float('inf')
patience = 2
patience_counter = 0

# Training Loop
training_losses = []
testing_losses = []
f1_scores = []
for epoch in range(num_epochs):
    # set model to training mode
    model.train()
    total_train_loss = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        
        loss = loss_fn(outputs, labels)
        total_train_loss += loss.item()
        
        # backpropagation & update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    avg_train_loss = total_train_loss / len(train_loader)
    training_losses.append(avg_train_loss)

    model.eval()
    total_val_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    # Disable gradient calculations for validation.
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            loss = loss_fn(outputs, labels)
            total_val_loss += loss.item()
            
            # given prediction based on 1 dimension tensor data
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)
    testing_losses.append(avg_val_loss)
    
    # Check if validation loss improved
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
    else:
        patience_counter += 1

    # Early stopping check
    if patience_counter >= patience:
        print("Stopping early.")
        break
    
    val_accuracy = 100 * correct / total
    f1 = f1_score(all_labels, all_predictions, average='macro')
    f1_scores.append(f1)
    print(f'Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%, F1 Score: {f1:.4f}')

# graph for loss over epochs
plt.figure(figsize=(10, 5))
plt.plot(training_losses, label='Training Loss')
plt.plot(testing_losses, label='Testing Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Testing Loss Over Epochs')
plt.legend()
plt.show()

# graph for f1 score over epochs
plt.figure(figsize=(10, 5))
plt.plot(f1_scores, label='F1 scores')
plt.xlabel('Epoch')
plt.ylabel('Scores')
plt.title('F1 Scores Over Epochs')
plt.legend()
plt.show()

# Save the Model
torch.save(model.state_dict(), 'model.pth')

# validate using local data
idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
img_path = input('输入本地图片位置：')
# img_path = './1.jpg'
image = Image.open(img_path).convert('RGB')
image = val_transform(image)
image = image.unsqueeze(0)
image = image.to(device)

model.eval()
with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    predicted_breed = idx_to_class[predicted.item()]
    predicted_breed = predicted_breed.replace("_", " ")
    predicted_breed = predicted_breed.title()
    
    print(f"图片里的狗狗是: {predicted_breed}")

