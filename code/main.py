from imageAugmentation import image_augmentation
from imageRandom import image_random

foldersNames = []

foldersNames.append("brain/acute_infarct")
foldersNames.append("brain/arteriovenous_anomaly")
foldersNames.append("brain/chronic_infarct")
foldersNames.append("brain/edema")
foldersNames.append("brain/extra")
foldersNames.append("brain/focal_flair_hyper")
foldersNames.append("brain/intra")
foldersNames.append("brain/white_matter_changes")
foldersNames.append("brain/normal")


for i in foldersNames:
   image_augmentation(i)

for i in foldersNames:
   image_random(i)