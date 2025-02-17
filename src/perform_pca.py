from sklearn.decomposition import PCA



def performPca( X_train , X_test, varianzaSpiegata):

    pca = PCA(n_components=0.95)  #percentuale di varianza spiegata che vogliamo mantenere dopo la pca, di solito 80 / 90%
    X_train_pca = pca.fit_transform(X_train)  # Fitta la PCA e trasforma i dati di training
    X_test_pca = pca.transform(X_test)


    return X_train_pca,X_test_pca





