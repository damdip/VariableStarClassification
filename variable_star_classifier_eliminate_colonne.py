# classifier.py
from src.data_loader import load_data, clean_data, getDataSetPath, getOutputPath
from src.preprocessing import preprocess_data
from src.cross_validation import perform_k_fold_cross_validation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from sklearn.decomposition import PCA

#Caricamento  del dataset
df = load_data(getDataSetPath())  # path corretto per il dataset
# Rimuove qualsiasi tipo di carattere di spazio dai nomi delle colonne


print("\nAnteprima dei dati caricati:\n", df.head())

#Preprocessing dei dati
X, y = preprocess_data(df)

#Divisione del dataset in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Creazione del modello
model = RandomForestClassifier(random_state=42)

# Allena il modello sul training set completo
model.fit(X_train, y_train)

#Valutazione del modello
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}\n")
print(classification_report(y_test, y_pred, zero_division=1))

# Passo 6: Salvataggio del modello
joblib.dump(model, getOutputPath())
