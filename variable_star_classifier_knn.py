# classifier.py
from src.data_loader import load_data, clean_data, getDataSetPath, getOutputPath
from src.perform_pca import performPca
from src.preprocessing import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
import joblib
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#Caricamento  del dataset
df = load_data(getDataSetPath())  # path corretto per il dataset

#Pulizia del dataset 
df = clean_data(df)

#Preprocessing dei dati
X, y = preprocess_data(df)

#Divisione del dataset in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pca
X_train_pca, X_test_pca = performPca(X_train,X_test , 0.95)

for  n in range(50):
    # Creazione del modello
    knn = KNeighborsClassifier(n_neighbors=10)

    # Allena il modello sui dati di addestramento
    knn.fit(X_train, y_train)

    # Previsione sul test set
    y_pred = knn.predict(X_test)

    #Valutazione del modello
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy Logistic Regression: {accuracy:.2f}\n")
    #print(classification_report(y_test, y_pred, zero_division=1))


#Salvataggio del modello
joblib.dump(knn, getOutputPath())
    

