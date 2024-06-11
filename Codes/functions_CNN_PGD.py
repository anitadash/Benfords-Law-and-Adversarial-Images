import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
import matplotlib.pyplot as plt
import numpy as np
from functions_BenfordLaw import *

use_cuda=True
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

'''
The functions in this file are used to 
(i) Train and test a simple CNN model 
(ii) Generate adversarial images using the PGD attack
(iii) Perform Benford's Law analysis on the adversarial images
'''

# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self, input_channels):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)  # Adjusted to match the actual size
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Function to process images
def image_processing(dataset):
    # returns processing, train_loader, test_loader
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    if dataset == 'mnist':
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset == 'fashion_mnist':
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    input_channels = 1 if dataset in ['mnist', 'fashion_mnist'] else 3
    return train_loader, test_loader, input_channels

# Function to fit the model
def train_model(dataset_name):
    train_loader, _ , input_channels = image_processing(dataset_name)
    model = SimpleCNN(input_channels).to(device)  
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.0001, betas=(0.9, 0.999), eps=1e-08)
    history = {'loss': [], 'accuracy': []}
    for epoch in range(10):  
        running_loss = 0.0
        total = 0
        correct = 0
        for i, data in enumerate(tqdm(train_loader), 0):  
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        average_loss = running_loss / len(train_loader)
        print('[%d] loss: %.3f' % (epoch + 1, average_loss))
        history['loss'].append(average_loss)
        accuracy = 100 * correct / total
        history['accuracy'].append(accuracy)
        print('Accuracy of the network on the train images: %d %%' % accuracy)

    print('Finished Training')
    return model, history

# test the model
def test_model(model, dataset_name):
    _, test_loader, input_channels = image_processing(dataset_name)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print('Accuracy of the network on the test images: %d %%' % accuracy)
    return accuracy

# Function to generate a sample for conducting the adversarial attack
def create_loader(dataset, num_samples=1000):
    # Randomly selecting a subset of the dataset
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    sampler = SubsetRandomSampler(indices)
    loader = DataLoader(dataset=dataset, batch_size=1, sampler=sampler)
    return loader

# Function to perform the PGD attack
def pgd_attack(model, images, labels, eps, alpha=2/255, iters=40) :
    images = images.to(device)
    labels = labels.to(device)
    loss = nn.CrossEntropyLoss()
        
    ori_images = images.data
        
    for i in range(iters) :    
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()

        adv_images = images + alpha*images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
            
    return images

# Function to test the PGD attack
def test_pgd(model, device, data_loader, epsilon):
    print('Performing PGD attack with epsilon =', epsilon)
    model.eval()
    correct = 0
    total = 0
    adv_images = []
    orig_images = []  
    for images, labels in tqdm(data_loader):
        # Save original images
        orig_images.append(images.clone().detach())
        
        # Generate adversarial images
        images = pgd_attack(model, images, labels, epsilon)
        adv_images.append(images.clone().detach())
        
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        
        _, pre = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (pre == labels).sum().item()
        
        if total >= 1000:
            break

    accuracy = 100 * correct / total
    return adv_images, orig_images, accuracy

# Function to plot the results
def plot_images_and_prediction(model, adv_images, orig_images, index=None):
    # Predicting the class of a random adversarial image
    if index is None:
        index = np.random.randint(0, len(adv_images))
    adv_image = adv_images[index]
    orig_image = orig_images[index]
    output = model(adv_image)
    _, pred = torch.max(output.data, 1)
    plt.figure(figsize=(4, 1))
    plt.subplot(1, 2, 1)
    plt.imshow(orig_image[0][0], cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(adv_image[0][0].detach().cpu().numpy(), cmap='gray')
    plt.title('Adversarial Image')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    print('Predicted:', pred.item())

# Function to perform benford's law test on the adversarial image
def process_and_test_image(adv_image, transformation):  
    # reshape adv_image to 28x28
    test_adv_image = adv_image[0][0].detach().cpu().numpy()
    test_adv_image = cv2.resize(test_adv_image, (28, 28))
    if transformation == 'gradient_magnitude':
        _, distribution = run_code(test_adv_image, 'gradient_magnitude')
        plot_run_code(test_adv_image, 'gradient_magnitude') 
    elif transformation == 'dct':
        _, distribution = run_code(test_adv_image, 'dct')
        plot_run_code(test_adv_image, 'dct')
    ks_statistic, p_value = ks_test(distribution, benfords_law())
    print('KS statistic:', ks_statistic)
    print('P-value:', p_value)