# classifier.py
from src.data_loader import load_data, clean_data
from src.preprocessing import preprocess_data
from src.cross_validation import perform_k_fold_cross_validation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from sklearn.decomposition import PCA

# Passo 0: Caricamento  del dataset
current_dir = os.getcwd()  
local_file_path  = "\VariableStarClassification\data\PLV_LINEAR.csv"
fullDataSetPath = current_dir + local_file_path
df = load_data(fullDataSetPath)  # path corretto per il dataset
# Rimuove qualsiasi tipo di carattere di spazio dai nomi delle colonne


df.columns = df.columns.str.replace(r'\s+', '', regex=True)

df = df.drop(["#","LR"], axis=1) # la rimozione di una colonna correlata non porta a differenze, random forest resiste bene 
# alle correlazioni


#Rimozione di alcune classi
# Filtra il dataframe escludendo questi valori
df = df[~df['LCtype'].isin([3, 9, 11, 8])]


print("\nAnteprima dei dati caricati:\n", df.head())

# Passo 2: Preprocessing dei dati
X, y = preprocess_data(df)

# Passo 3: Divisione del dataset in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Passo 4: Creazione del modello
model = RandomForestClassifier(random_state=42)

# Esegui la k-fold cross-validation solo su X_train e y_train
#k = 5  # Ad esempio, 5-fold cross-validation
#perform_k_fold_cross_validation(model, X_train, y_train, k)

# Allena il modello sul training set completo
model.fit(X_train, y_train)

# Passo 5: Valutazione del modello
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}\n")
print(classification_report(y_test, y_pred, zero_division=1))

# Passo 6: Salvataggio del modello
outputPath = current_dir + "\VariableStarClassification\models\\final_model.pkl"
joblib.dump(model, outputPath)
