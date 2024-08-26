# import cv2
# import os

# def load_images_from_folder(folder):
#     foldersNames = []
#     for filename in os.listdir(folder):
#         img = cv2.imread(os.path.join(folder,filename))
#         if img is not None:
#             foldersNames.append("brain/acute_infarct")
#             foldersNames.append("brain/arteriovenous_anomaly")
#             foldersNames.append("brain/chronic_infarct")
#             foldersNames.append("brain/edema")
#             foldersNames.append("brain/extra")
#             foldersNames.append("brain/focal_flair_hyper")
#             foldersNames.append("brain/intra")
#             foldersNames.append("brain/white_matter_changes")
#             foldersNames.append("brain/normal")
#     return foldersNames



import os
import cv2
import numpy as np

def load_images_from_folder(folder_path):
    images = []
    labels = []
    for label, class_folder in enumerate(os.listdir(folder_path)):
        class_folder_path = os.path.join(folder_path, class_folder)
        if os.path.isdir(class_folder_path):
            for image_file in os.listdir(class_folder_path):
                image_path = os.path.join(class_folder_path, image_file)
                if os.path.isfile(image_path):  # Check if it's a file
                    image = cv2.imread(image_path)
                    if image is not None:
                        images.append(image.flatten())  # You may need to flatten or resize the images as needed
                        labels.append(label)
                else:
                    print("Warning: {} is not a valid file path.".format(image_path))
        else:
            print("Warning: {} is not a valid directory.".format(class_folder_path))
    return np.array(images), np.array(labels)