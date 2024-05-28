#!/usr/bin/env python
# coding: utf-8

# ## AIN 422 - Programming Assignment 2
# 
# * You can add as many cells as you want in-between each question.
# * Please add comments to your code to explain your work.  
# * Please add Markdown cells to answer the (non-coding) questions in the homework text. You can, however, refer to the outputs of code cells without adding them as images to the Markdown cell unless you are requested to do otherwise.
# * Please be careful about the order of runs of cells. Doing the homework, it is likely that you will be running the cells in different orders, however, they will be evaluated in the order they appear. Hence, please try running the cells in this order before submission to make sure they work.    
# * Please refer to the homework text for any implementation detail. Though you are somewhat expected to abide by the comments in the below cells, they are mainly just provided for guidance. That is, as long as you are not completely off this structure and your work pattern is understandable and traceable, it is fine. For instance, you do not have to implement a particular function within a cell just because the comment directs you to do so.
# * This document is also your report. Show your work.

# ###  Kazım Halil KESMÜK 2200765031

# ## 1. Implementing a CNN from Scratch (60 points)

# ### 1.1. Introduction
# * Brief overview of the task.
# * Answer the questions like, What are the main components of a CNN architecture?, Why we use this in image classification?, etc.
# * Description of the dataset used for classification.

# ### 1.2. Data Loading and Preprocessing (5 points)

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns


# In[2]:


## Load the dataset using PyTorch's data loading utilities
## Apply necessary preprocessing such as resizing and normalization
## Divide the dataset into training, validation, and testing subsets


# In[3]:


data_dir = 'pa3_subset'


# In[4]:


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# In[5]:


dataset = datasets.ImageFolder(root=data_dir, transform=transform)


# In[6]:


train_size = int(0.6 * len(dataset))  # 60% for training
val_size = int(0.2 * len(dataset))    # 20% for validation
test_size = len(dataset) - train_size - val_size  # Remaining for testing


# In[7]:


train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])


# In[8]:


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# ### 1.3. Define your CNN architecture (10 points)
# * Explain the reason behind your architecture.
# * Explain your choice of activation functions.

# In[9]:


## Design a CNN architecture with at least 5 convolutional layers
## Add activation functions (e.g., ReLU) after each convolutional layer
## Intersperse pooling layers (e.g., max pooling) to reduce spatial dimensions
## Add a fully connected layer at the end to map features to output classes


# In[10]:


class VegetableCNN(nn.Module):
    def __init__(self):
        super(VegetableCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Linear(512, 15)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


# conv_layers: This section defines a series of convolutional layers with ReLU activation and max pooling for feature extraction from images.
# 
# fc_layers: This section defines fully connected layers for classification, taking the extracted features and outputting predictions for 15 classes (presumably vegetable types).
# 
# forward: This function defines the forward pass of the network, passing the input through the convolutional and fully connected layers to produce the output.

# ### 1.4 Prepare the model for training (5 points)
# * Explain your choice of loss functions and optimization algorithms.

# In[12]:


## Define appropriate loss function for multi-class classification (e.g., cross-entropy loss)


# In[13]:


## Choose an optimizer (e.g., SGD, Adam) and set its parameters (e.g., learning rate)


# In[14]:


model = VegetableCNN()


# In[15]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# In[16]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# ### 1.5 Train and Validate the CNN model (15 points)

# In[17]:


## Iterate over the training dataset in mini-batches
## Implement forward pass, compute loss, and backward pass for gradient computation
## Update model parameters using the optimizer based on computed gradients
## Validate the model on the validation set periodically and plot the validation loss
## Repeat the training process for a suitable number of epochs (at least 30epohs)


# In[18]:


## Conduct experiments with different hyperparameters.
## Experiment with at least 3 different learning rates and 2 different batch sizes.


# In[19]:


## Visualize the accuracy and loss change of the experiments across training and validation datasets.
## Select your best model with respect to validation accuracy


# In[20]:


num_epochs = 30


# In[37]:


# Hyperparameters
learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [32, 64]
best_val_acc = 0.0
best_model_weights = None

for lr in learning_rates:
    for batch_size in batch_sizes:
        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Model, loss and optimizer
        model = VegetableCNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # train and val loss and acc saving
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            train_acc = 0.0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_acc += (predicted == labels).sum().item()

            train_loss /= len(train_dataset)
            train_acc /= len(train_dataset)

            model.eval()
            val_loss = 0.0
            val_acc = 0.0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    val_acc += (predicted == labels).sum().item()

            val_loss /= len(val_dataset)
            val_acc /= len(val_dataset)


            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_weights = model.state_dict()

            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

         
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Training Accuracy')
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.suptitle(f'Learning Rate: {lr}, Batch Size: {batch_size}')
        plt.tight_layout()
        plt.show()

        model.load_state_dict(best_model_weights)


# ### 1.6 Evaluate the trained model on the test set (15 points)

# In[39]:


## Test the trained model on the test set to evaluate its performance
## Compute metrics such as accuracy, precision, recall, and F1-score
## Visualize confusion matrix to understand the model's behavior across different classes
## Comment on the results


# In[40]:


# Test the trained model on the test set
model.eval()
test_loss = 0.0
test_acc = 0.0
test_predictions = []
test_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        test_acc += (predicted == labels).sum().item()

        test_predictions.extend(predicted.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

test_loss /= len(test_dataset)
test_acc /= len(test_dataset)

# accuracy, precision, recall, and F1-score
test_precision = precision_score(test_labels, test_predictions, average='weighted')
test_recall = recall_score(test_labels, test_predictions, average='weighted')
test_f1 = f1_score(test_labels, test_predictions, average='weighted')

print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
print(f'Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1-score: {test_f1:.4f}')

# confusion matrix
cm = confusion_matrix(test_labels, test_predictions)
class_names = test_dataset.dataset.classes

plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# Low Test Loss (0.5239): This indicates the model's predictions are close to the true labels.
# High Test Accuracy (91.56%): The model correctly classifies a large majority of the vegetable images.
# Precision (91.97%): When the model predicts a specific vegetable, it is correct most of the time.
# Recall (91.56%): The model successfully identifies most instances of each vegetable class.
# F1-score (91.60%): This balanced metric confirms the model's overall effectiveness, considering both precision and recall.

# ### 1.7 Conclusion and interpretation (10 points)
# * Summarize the performance of the model on the test set
# * Discuss any challenges encountered during training and potential areas for improvement
# * Reflect on the overall effectiveness of the chosen CNN architecture and training approach

# In[42]:


print("The model achieved a test accuracy of {:.2f}%.".format(test_acc * 100))


# ## 2. Exploring Transfer Learning with ResNet50 (40 points)

# ### 2.1. Introduction
# * Brief overview of the task.
# * Answer the questions like, What is fine-tuning? Why should we do this? Why do we freeze the rest and train only last layers?

# ### 2.2. Load the pre-trained ResNet50 model (5 points)
# 

# In[44]:


## Utilize torchvision library to load the pre-trained ResNet50 model
## Ensure that the model's architecture matches ResNet50, by checking the model summary.


# In[45]:


# Load the pre-trained ResNet50 model
resnet50 = models.resnet50(pretrained=True)

# Verify the model architecture
print(resnet50)


# ### 2.3 Modify the ResNet50 model for transfer learning (10 points)

# In[46]:


## Freeze all layers of the ResNet50 model.
## Replace the final fully connected layer with a new FC layer matching the number of classes
## Unfreeze the final FC layer


# In[47]:


## Define appropriate loss function and optimizer for training


# In[48]:


## Train the modified ResNet50 model on the vegetable image dataset.
## Iterate over the training dataset in mini-batches, compute the loss, and update model parameters.
## Monitor the training process and evaluate the model's performance on the validation set periodically.
## Visualize the accuracy and loss changes of the model across training and validation datasets.


# In[49]:


# Freeze all layers of the ResNet50 model
for param in resnet50.parameters():
    param.requires_grad = False

# Replace the final fully connected layer
num_classes = 15  # Number of classes in the vegetable dataset
resnet50.fc = nn.Linear(resnet50.fc.in_features, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet50.fc.parameters(), lr=0.001)

# Training loop
num_epochs = 30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet50.to(device)


# In[50]:


for epoch in range(num_epochs):
    resnet50.train()
    train_loss = 0.0
    train_acc = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = resnet50(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_acc += (predicted == labels).sum().item()

    train_loss /= len(train_dataset)
    train_acc /= len(train_dataset)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    resnet50.eval()
    val_loss = 0.0
    val_acc = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = resnet50(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_acc += (predicted == labels).sum().item()

    val_loss /= len(val_dataset)
    val_acc /= len(val_dataset)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()


# ### 2.4 Evaluate the fine-tuned ResNet50 model (15 points)

# In[51]:


## Test the model on the test set to evaluate its performance.
## Compute metrics such as accuracy, precision, recall, and F1-score to assess classification performance.
## Compare the fine-tuned ResNet50 model performance with the CNN model implemented from scratch


# In[52]:


# Testing and evaluation
resnet50.eval()
test_loss = 0.0
test_acc = 0.0
test_predictions = []
test_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = resnet50(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        test_acc += (predicted == labels).sum().item()

        test_predictions.extend(predicted.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

test_loss /= len(test_dataset)
test_acc /= len(test_dataset)

test_precision = precision_score(test_labels, test_predictions, average='weighted')
test_recall = recall_score(test_labels, test_predictions, average='weighted')
test_f1 = f1_score(test_labels, test_predictions, average='weighted')

print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
print(f'Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1-score: {test_f1:.4f}')


# I have these result in from scratch part : 
# 
# Test Loss: 0.5239, Test Acc: 0.9156
# Test Precision: 0.9197, Test Recall: 0.9156, Test F1-score: 0.9160

# ### 2.5 Analyze advantages and disadvantages (10 points)
# 
# * Provide insights on the advantages and disadvantages of transfer learning vs. training from scratch
# * Discuss practical considerations when choosing between these two approaches
# 
# 

# When it comes to choosing between transfer learning and training from scratch, there are several factors to consider. Transfer learning has its advantages, such as leveraging pre-trained models that have learned rich features from large-scale datasets, resulting in faster convergence and higher accuracy, especially when working with limited labeled data. It also allows for efficient utilization of computational resources. However, pre-trained models may not capture specific features required for the target task, and there is limited flexibility in designing custom architectures.
# 
# 
# On the other hand, training from scratch provides complete flexibility to design custom architectures tailored to specific tasks and allows for full control over the training process. All model parameters can be optimized specifically for the target task. However, training from scratch requires more computational resources and time, especially with large datasets. It also poses a higher risk of overfitting when working with limited labeled data and may result in slower convergence and potentially lower accuracy compared to transfer learning.
# 
# 
# Practical considerations when choosing between transfer learning and training from scratch include the size of the dataset, similarity of the target task to the original task of the pre-trained model, available computational resources, and requirements for custom architectures. If the target task is similar to the original task and computational resources are limited, transfer learning is often more effective. However, if the task is significantly different or custom architectures are required, training from scratch may be more suitable.
# 
# 
# Ultimately, the choice between transfer learning and training from scratch depends on the specific requirements and constraints of the project. It's essential to consider the trade-offs and align the approach with the available resources and desired outcomes.
