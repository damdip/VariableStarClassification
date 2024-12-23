# classifier.py
import joblib
import os
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from src.data_loader import load_data, clean_data
from src.preprocessing import preprocess_data

# Passo 0: Caricamento  del dataset
current_dir = os.getcwd()  
local_file_path  = "\data\PLV_LINEAR.csv"
fullDataSetPath = current_dir + local_file_path
df = load_data(fullDataSetPath)  # path corretto per il dataset

# Passo 1: Pulizia del dataset 
df = clean_data(df)

# Passo 2: Preprocessing dei dati
X, y = preprocess_data(df)

# Passo 3: Divisione del dataset in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Passo 4: Creazione del modello
model = RandomForestClassifier(random_state=42)

# Definisci lo spazio di ricerca degli iperparametri
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Esegui il tuning iperparametrico con Grid Search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Stampa i migliori parametri trovati
print("Best Hyperparameters:", grid_search.best_params_)

# Modello ottimizzato
best_model = grid_search.best_estimator_

# Passo 5: Valutazione del modello
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}\n")
print(classification_report(y_test, y_pred, zero_division=1))

# Passo 6: Salvataggio del modello
outputPath = current_dir + "\models\\final_model.pkl"
joblib.dump(best_model, outputPath)
