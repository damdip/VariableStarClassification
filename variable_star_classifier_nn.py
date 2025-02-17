from src.data_loader import load_data, clean_data, getDataSetPath, getOutputPath
from src.preprocessing import preprocess_data
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tensorflow import keras
from keras import layers
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np


#Caricamento  del dataset
df = load_data(getDataSetPath())  # path corretto per il dataset

#Pulizia del dataset 
df = clean_data(df)

#Preprocessing dei dati
X, y = preprocess_data(df)

# One-hot encoding delle classi
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y.shape(-1, 1))

#Divisione del dataset in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Creare modello fully connected
model = keras.Sequential([
    layers.Dense(32, activation="relu", input_shape=(X.shape[1],)),  # Primo hidden layer
    layers.Dense(16, activation="relu"),  # Secondo hidden layer
    layers.Dense(y.shape[1], activation="softmax")  # Output layer per multi-classe
])

# Compilare il modello
model.compile(optimizer="adam",
              loss="categorical_crossentropy",  # Per multi-class classification
              metrics=['accuracy', "precision", "recall", "AUC"])

# Addestrare il modello
model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test))

# Valutare il modello
test_loss, test_acc, test_prec, test_rec, test_f1 = model.evaluate(X_test, y_test)
print(f"Accuratezza sul test set: {test_acc:.2f}")
print(f"Precision sul test set: {test_acc:.2f}")
print(f"Recall sul test set: {test_acc:.2f}")
print(f"F1 sul test set: {test_acc:.2f}")


