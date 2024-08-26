import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from joblib import Memory
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
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

# aplatizează datele înainte de împărțire
num_samples, img_height, img_width = X.shape
X_flatten = X.reshape(num_samples, -1)

# împarte datele în 80% date de antrenare și 20% date de testare 
X_train, X_test, y_train, y_test = train_test_split(X_flatten, y, test_size=0.2, random_state=42)

# redimensionarea datelor
scaler = StandardScaler()

# scalarea datelor
train_X_scaled = scaler.fit_transform(X_train)
test_X_scaled = scaler.transform(X_test)

print("Se antrenează modelul SVM...")


# # definirea parametrilor pentru validarea încrucișată
# param_grid = {'C': [0.1, 1, 10, 100, 200, 300, 1000],
#               'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
#               'kernel': ['rbf']}

# print("Se calculează cei mai buni parametri...")

# # metoda validării încrucișate
# def calculate_best_params(grid):
#     svm  = SVC()        #  creează o instanță pentru clasificatorul SVM
#     svm_cv = GridSearchCV(svm, grid)        # creează o instanță care efectuează căutarea pentru validarea încrucișată
#     svm_cv.fit(X_train,train_y)             # potrivește modelul creat cu datele de antrenare
#     best_svc = svm_cv.best_estimator_       # preia cea mai bună combinație de parametri
#     print("Best Parameters:",best_svc)
#     print("Train Score:",svm_cv.best_score_)      # afișează acuratețea
#     print("Test Score:",svm_cv.score(X_test,test_y))
 
# calculate_best_params(param_grid)


# crearea modelului SVM
c_svm = 10
gamma_svm = 0.0001
kernel_type = 'rbf'

svm_model = svm.SVC(kernel=kernel_type, C=c_svm, gamma=gamma_svm)


# # efectuarea validării încrucișate
# cv_scores = cross_val_score(svm_model, train_X_scaled, y_train, cv=9)

# print("Cross-validation scores:", cv_scores)
# print("Mean cross-validation score:", cv_scores.mean())

# antrenarea modelului
svm_model.fit(train_X_scaled, y_train)

# realizarea predicțiilor pe datele de test
pred = svm_model.predict(test_X_scaled)

# calculează acuratețea
accuracy = accuracy_score(y_test, pred)
print("Accuracy:", accuracy)

# evaluează modelul pe întreg setul de date
overall_accuracy = svm_model.score(train_X_scaled, y_train)
print("Overall accuracy on training data:", overall_accuracy)

# generează raportul de clasificare
report = classification_report(y_test, pred, target_names=foldersNames)
print("Classification Report:\n", report)

# generează matricea de confuzie
confusion = confusion_matrix(y_test, pred)
plt.figure(figsize=(15,15))
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=foldersNames, yticklabels=foldersNames)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Confusion Matrix")
plt.show()



# salvează rezultatele în Excel
results = {
    # 'Cross-validation scores': cv_scores,
    # 'Mean cross-validation score': [cv_scores.mean()],
    'Accuracy': [accuracy],
    'Overall accuracy on training data': [overall_accuracy]
}

report_dict = metrics.classification_report(y_test, pred, target_names=foldersNames, output_dict=True)

with pd.ExcelWriter('svm_results.xlsx') as writer:
    pd.DataFrame(results).to_excel(writer, sheet_name='Summary', index=False)
    pd.DataFrame(report_dict).T.to_excel(writer, sheet_name='Classification Report')