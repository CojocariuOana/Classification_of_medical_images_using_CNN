import os
import cv2
import numpy as np

def load_medical_data(data_directory):
    classes = os.listdir(data_directory)
    num_classes = len(classes)
    images = []
    labels = []
    class_mapping = {}  # To store class labels
    for i, class_name in enumerate(classes):
        class_mapping[i] = class_name
        class_directory = os.path.join(data_directory, class_name)
        for image_file in os.listdir(class_directory):
            image_path = os.path.join(class_directory, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read images in grayscale
            # You might need to resize or preprocess the images here
            images.append(image)
            labels.append(i)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels, class_mapping

# Example usage:
data_directory = r'C:\Users\Oana\Desktop\Licență\brain'
images, labels, class_mapping = load_medical_data(data_directory)
print("Loaded", len(images), "images.")
print("Classes:", class_mapping)
