from functions_BenfordLaw import *
from functions_CNN_PGD import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 

'''
Functions in this file are used to perform adversarial
image detection using Benford's Law analysis.
'''

# Function to process images
def process_images(orig_images, adv_images, num_images=1000):
    original_images = [orig_images[i] for i in range(num_images)]
    adversarial_images = [adv_images[i] for i in range(num_images)]
    original_images = np.array(original_images)
    adversarial_images = np.array(adversarial_images)
    original_images = np.array([cv2.resize(image.reshape(28, 28), (28, 28)) for image in original_images])
    adversarial_images = np.array([cv2.resize(image.reshape(28, 28), (28, 28)) for image in adversarial_images])
    original_images = np.array([image.reshape(28, 28) for image in original_images])
    adversarial_images = np.array([image.reshape(28, 28) for image in adversarial_images])
    return original_images, adversarial_images

# Function to classify images using Logistic Regression
def train_and_evaluate_model(original_images, adversarial_images, transformation):
    # Computing the KS statistics
    ks_stats = []
    for img in original_images:
        if transformation == 'gradient_magnitude':
            _, distribution = run_code(img, 'gradient_magnitude')
        elif transformation == 'dct':
            _, distribution = run_code(img, 'dct')
        ks_statistic, _ = ks_test(distribution, benfords_law())
        ks_stats.append(ks_statistic)

    for img in adversarial_images:
        if transformation == 'gradient_magnitude':
            _, distribution = run_code(img, 'gradient_magnitude')
        elif transformation == 'dct':
            _, distribution = run_code(img, 'dct')
        ks_statistic, _ = ks_test(distribution, benfords_law())
        ks_stats.append(ks_statistic)
    ks_stats = np.array(ks_stats).reshape(-1, 1)

    orig_labels = np.zeros(len(original_images)) # original images are labeled as 0
    adv_labels = np.ones(len(adversarial_images)) # adversarial images are labeled as 1
    y = np.concatenate([orig_labels, adv_labels])
    y = y.reshape(-1, 1)

    # Splitting the dataset into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(ks_stats, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Training the Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    # Evaluating the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return ks_stats, y, model, accuracy, precision, recall, f1

# Function to plot the KS statistics of the images
def plot_ks_stats(y_shuffle, ks_stats_shuffle, transformation):
    plt.figure(figsize=(6, 4))
    plt.plot(np.where(y_shuffle == 0)[0], ks_stats_shuffle[y_shuffle == 0], 'co', label='Original')
    plt.plot(np.where(y_shuffle == 1)[0], ks_stats_shuffle[y_shuffle == 1], 'yo', label='Adversarial')
    plt.ylabel('KS Statistic')
    plt.suptitle('KS Statistic of Original and Adversarial Images', y = 0.95)
    if transformation == 'gradient_magnitude':
        plt.title('Gradient Magnitude Transformation')
    elif transformation == 'dct':
        plt.title('DCT Transformation')
    plt.legend()
    plt.tight_layout()
    plt.show()