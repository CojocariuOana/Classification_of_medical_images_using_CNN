import os
import cv2
import keras
import numpy as np
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import gc
import pandas as pd 

# se eliberează memoria
gc.collect()


# încarcă datele din directorul specificat
def load_images_from_folders(folder_paths):
    images = []           # inițializează liste pentru imagini și etichete
    labels = []
    for label, folder_path in enumerate(folder_paths):
        if os.path.isdir(folder_path):         # se verifică dacă directorul este valid
            for image_file in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_file)
                if os.path.isfile(image_path):
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)    # citește imaginea în nuanțe de gri
                    if image is not None:
                        images.append(image)      # dacă imaginea e încărcată cu succes este adăugată listei 
                        labels.append(label)      # eticheta corespunzătoare este adăugată și ea 
                else:
                    print(f"Atenție: {image_path} nu este o cale validă.")
        else:
            print(f"Atenție: {folder_path} nu este un director valid.")
    return np.array(images), np.array(labels)

# definește calea pentru fiecare director al fiecărei clase
base_folder = r"toate_datele"
foldersNames = [
    "acute_infarct",
    "arteriovenous_anomaly", 
    "chronic_infarct",
    "edema",
    "extra",
    "focal_flair_hyper",
    "intra",
    "normal",
    "white_matter_changes"
]

folder_paths = [os.path.join(base_folder, folder_name) for folder_name in foldersNames]

print("Se încarcă datele...")
X, y = load_images_from_folders(folder_paths)

# normalizează valorile pixelilor pentru a fi cumprinse în intervalul [0, 1]
X = X / 255.0

# adaugă o dimensiune suplimentară matricei X, necesară straturilor convoluționare 2D
X = np.expand_dims(X, axis=-1)

# împarte datele în 70% date de antrenare și 30% date temporare
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

# împarte datele temporare în date de validare (1/3 of 30% = 10%) și date de testare (2/3 of 30% = 20%)
X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42)

# redimensionarea datelor
scaler = StandardScaler()

# aplatizarea datelor - 2D
train_X_flatten = X_train.reshape(X_train.shape[0], -1)
test_X_flatten = X_test.reshape(X_test.shape[0], -1)
val_X_flatten = X_validation.reshape(X_validation.shape[0], -1)

# scalarea datelor
train_X_scaled = scaler.fit_transform(train_X_flatten)
test_X_scaled = scaler.transform(test_X_flatten)
val_X_scaled = scaler.transform(val_X_flatten)

# remodelarea datelor scalate la forma inițială - 4D
train_X_scaled = train_X_scaled.reshape(X_train.shape)
test_X_scaled = test_X_scaled.reshape(X_test.shape)
val_X_scaled = val_X_scaled.reshape(X_validation.shape)

# codificarea datelor pentru a putea fi procesate
train_y = to_categorical(y_train, num_classes=9)
val_y = to_categorical(y_validation, num_classes=9)

# crearea modelului utilizând rețele neuronale convoluționale
def create_model(dropout_rate):
    model = Sequential([
        Conv2D(filters=4, kernel_size=3, activation='relu', padding='same', input_shape=(224, 224, 1)),
        MaxPool2D(pool_size=2),
        Dropout(dropout_rate),
        
        Conv2D(filters=16, kernel_size=3, activation='relu', padding='same'),
        MaxPool2D(pool_size=2),
        Dropout(dropout_rate),
        
        Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'),
        MaxPool2D(pool_size=2),
        Dropout(dropout_rate),
        
        Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
        MaxPool2D(pool_size=2),
        Dropout(dropout_rate),
        
        Conv2D(filters=126, kernel_size=3, activation='relu', padding='same'),
        MaxPool2D(pool_size=2),
        Dropout(dropout_rate),
        
        Conv2D(filters=252, kernel_size=3, activation='relu', padding='same'),
        MaxPool2D(pool_size=2),
        Dropout(dropout_rate),
        
        Conv2D(filters=504, kernel_size=3, activation='relu', padding='same'),
        MaxPool2D(pool_size=2),
        Dropout(dropout_rate),
        
        Flatten(),
        Dense(units=256, activation='relu'),
        Dropout(dropout_rate),
        
        Dense(units=128, activation='relu'),
        Dropout(dropout_rate),
        
        Dense(units=64, activation='relu'),
        Dropout(dropout_rate),
        
        Dense(units=32, activation='relu'),
        Dropout(dropout_rate),
        
        Dense(units=9, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model



# # testarea modelului cu diferite valori pentru dropout
# dropout_rates = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5]  
# best_dropout_rate = None
# best_accuracy = 0

# for dropout_rate in dropout_rates:
#     print("Testing dropout rate:", dropout_rate)
    
#     model = create_model(dropout_rate)
#     history = model.fit(train_X_scaled, train_y, epochs=130, batch_size=32, validation_data=(val_X_scaled, val_y), verbose=1)
    
#     # preia valoarea pentru acuratețe obținută în urma validării
#     val_accuracy = history.history['val_accuracy'][-1]
#     print("Validation accuracy:", val_accuracy)
          
#     if val_accuracy > best_accuracy:
#         best_accuracy = val_accuracy
#         best_dropout_rate = dropout_rate

# print("Best dropout rate:", best_dropout_rate)



best_dropout_rate = 0.1

def compile_model(model, optimizer='adam', loss='categorical_crossentropy'):
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    
def fitting_model(model, train_X_scaled, train_y, val_X_scaled, val_y, epoch):
    history = model.fit(train_X_scaled, train_y, validation_data=(val_X_scaled, val_y), batch_size=32, epochs=epoch)
    return history

# crearea modelului
model = create_model(best_dropout_rate)

# optimizarea modelului
compile_model(model, 'adam', 'categorical_crossentropy')

# antrenarea modelului
history = fitting_model(model, train_X_scaled, train_y, val_X_scaled, val_y, epoch=130)
 # modelul va fi salvat în folderul rădăcină pentru a putea fi apelat ulterior pentru predicție
model.save("cnn_digitclass.keras") 


# vizualizarea performanței modelului
f = plt.figure(figsize=(20, 8))

# acuratețea
plt1 = f.add_subplot(121)
plt1.plot(history.history['accuracy'], label='Training accuracy')
plt1.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.legend()
plt.title('Accuracy')

# pierderile
plt2 = f.add_subplot(122)
plt2.plot(history.history['loss'], label='Training loss')
plt2.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.title('Loss')

plt.show()


# realizarea predicțiilor
predictions = model.predict(test_X_scaled)

# conversia predicțiilor la o clasă de etichete
predicted_labels = np.argmax(predictions, axis=1)

# calcularea acurateții generale
overall_accuracy = np.mean(predicted_labels == y_test)
print("Overall Accuracy:", overall_accuracy)


# generează raportul de clasificare
report = classification_report(y_test, predicted_labels, target_names=foldersNames)
print("Classification Report:\n", report)

# generează matricea de confuzie
confusion = confusion_matrix(y_test, predicted_labels)
plt.figure(figsize=(15,15))
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=foldersNames, yticklabels=foldersNames)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Confusion Matrix")
plt.show()
 


# salvează fișierele în Excel
results = {
    'Overall accuracy on training data': [overall_accuracy]
}


report_dict = metrics.classification_report(y_test, predicted_labels, target_names=foldersNames, output_dict=True)


with pd.ExcelWriter('cnn_results.xlsx') as writer:
    pd.DataFrame(results).to_excel(writer, sheet_name='Summary', index=False)
    pd.DataFrame(report_dict).T.to_excel(writer, sheet_name='Classification Report')