import numpy as np
import matplotlib.pyplot as plt
import cv2
import collections
from scipy import fftpack
from scipy.stats import ks_2samp

'''
The functions in this file are used to perform 
Benford's Law analysis on images.
'''

# Function that performs gradient magnitude transformation
def gradient_magnitude_transformation(image):
    # Compute the x and y gradients using the Sobel operator
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    # Compute the gradient magnitude
    gradient_magnitude = cv2.magnitude(gradient_x, gradient_y)
    return gradient_magnitude

# function that performs dct transformation
def dct_transform(image):
    dct_image = fftpack.dct(fftpack.dct(image.T, norm='ortho').T, norm='ortho')
    return dct_image

# Function to extract the first non-zero digit of a number
def first_non_zero_digit(number):
    str_number = str(abs(number)).lstrip('0')
    for char in str_number:
        if char.isdigit() and int(char) != 0:
            return int(char)
    return None

# Function to compute the first digit distribution
def first_digit_distribution(numbers):
    first_digits = [first_non_zero_digit(number) for number in numbers if number != 0]
    distribution = collections.Counter(first_digits)
    distribution_probabilities = {digit: count / len(first_digits) for digit, count in distribution.items() if digit is not None}
    
    # Sort the dictionary in ascending order by keys
    distribution_probabilities = {k: v for k, v in sorted(distribution_probabilities.items())}
    
    return distribution_probabilities

# Function to compute Benford's law probabilities
def benfords_law():
    benfords_law_probabilities = {i: np.log10(1 + 1/i) for i in range(1, 10)}
    return benfords_law_probabilities

# Function to perform the Kolmogorov-Smirnov test between two distributions
def ks_test(distribution_probabilities, benfords_law_probabilities):
    ks_statistic, p_value = ks_2samp(list(distribution_probabilities.values()), list(benfords_law_probabilities.values()))
    return ks_statistic, p_value

# Function to run the analysis
def run_code(image, transformation):
    # Performing the transformation
    if transformation == 'gradient_magnitude':
        transformed_image = gradient_magnitude_transformation(image)
    elif transformation == 'dct':
        transformed_image = dct_transform(image)
    transformed_image = transformed_image.flatten()
    distribution_probabilities = first_digit_distribution(transformed_image)
    return transformed_image, distribution_probabilities

# Function to plot the results
def plot_run_code(image, transformation):
    if transformation == 'gradient_magnitude':
        transformed_image = gradient_magnitude_transformation(image)
    elif transformation == 'dct':
        transformed_image = dct_transform(image)
    # Displaying the original and transformed images
    norm_image = image.astype(float)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(norm_image)
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(transformed_image, cmap='gray')
    plt.title('Transformed Image')
    plt.axis('off')
    plt.show()
    # First digit distribution of the transformed image
    transformed_image = transformed_image.flatten()
    distribution_probabilities = first_digit_distribution(transformed_image)
    benfords_law_probabilities = benfords_law()
    # Plotting the distribution
    plt.figure()
    plt.plot(list(distribution_probabilities.keys()), list(distribution_probabilities.values()), 'bo-', label='Transformed Image')
    plt.plot(list(benfords_law_probabilities.keys()), list(benfords_law_probabilities.values()), 'ro-', label="Benford's Law")
    plt.xlabel('First Digit')
    plt.ylabel('Probability')
    plt.title('First Digit Distribution')
    plt.legend()
    plt.show()