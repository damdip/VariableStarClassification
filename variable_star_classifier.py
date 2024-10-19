# classifier.py
from src.data_loader import load_data, clean_data
from src.preprocessing import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Passo 1: Caricamento e pulizia del dataset
df = load_data('C:/Users/hp/ClassificationOfVariableStars/VariableStarClassification/data/PLV_LINEAR.csv')  # path corretto per il dataset
df = clean_data(df)

# Passo 2: Preprocessing dei dati
X, y = preprocess_data(df)

# Passo 3: Divisione del dataset in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Passo 4: Addestramento del modello
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Passo 5: Valutazione del modello
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}\n")
print(classification_report(y_test, y_pred, zero_division=1))

# Passo 6: Salvataggio del modello
joblib.dump(model, 'C:/Users/hp/ClassificationOfVariableStars/VariableStarClassification/models/final_model.pkl')
