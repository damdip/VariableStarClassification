# classifier.py
from src.data_loader import load_data, clean_data, getDataSetPath, getOutputPath
from src.preprocessing import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
import joblib
import os

#Caricamento  del dataset
df = load_data(getDataSetPath())  # path corretto per il dataset

#Pulizia del dataset 
df = clean_data(df)

#Preprocessing dei dati
X, y = preprocess_data(df)

#Divisione del dataset in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Logistic regression
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)


# Allena il modello sui dati di addestramento
model.fit(X_train, y_train)

# Fai previsioni sul set di test
y_pred = model.predict(X_test)

#Valutazione del modello
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Logistic Regression: {accuracy:.2f}\n")
print(classification_report(y_test, y_pred, zero_division=1))


#Salvataggio del modello
joblib.dump(model, getOutputPath())

    

