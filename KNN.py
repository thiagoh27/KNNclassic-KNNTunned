import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_digits, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report

# Carregar os datasets
iris = load_iris()
digits = load_digits()
breast_cancer = load_breast_cancer()
wine = load_wine()

datasets = {
    "Iris": iris,
    "Digits": digits,
    "Breast Cancer": breast_cancer,
    "Wine": wine
}

# Função para treinar e avaliar o classificador KDE
def kde_classifier(dataset, bandwidth):
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, random_state=42)
    
    # Treinar um KDE para cada classe
    kde_models = {}
    for class_label in np.unique(y_train):
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
        kde.fit(X_train[y_train == class_label])
        kde_models[class_label] = kde
    
    # Função para prever a classe de uma nova amostra
    def predict(X):
        log_probs = np.array([kde_models[class_label].score_samples(X) for class_label in kde_models])
        return np.argmax(log_probs, axis=0)
    
    # Prever no conjunto de teste
    y_pred = predict(X_test)
    
    # Avaliar o classificador
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"KDE Classifier - {dataset['DESCR'].splitlines()[0]} dataset with bandwidth {bandwidth}:")
    print(f"Accuracy: {accuracy:.2f}")
    print(report)
    
    return accuracy

# Função para treinar e avaliar o classificador Naive Bayes
def nb_classifier(dataset):
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, random_state=42)
    
    # Treinar o Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    
    # Prever no conjunto de teste
    y_pred = nb.predict(X_test)
    
    # Avaliar o classificador
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Naive Bayes Classifier - {dataset['DESCR'].splitlines()[0]} dataset:")
    print(f"Accuracy: {accuracy:.2f}")
    print(report)
    
    return accuracy

# Função para treinar e avaliar o classificador Perceptron
def perceptron_classifier(dataset):
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, random_state=42)
    
    # Treinar o Perceptron
    perceptron = Perceptron()
    perceptron.fit(X_train, y_train)
    
    # Prever no conjunto de teste
    y_pred = perceptron.predict(X_test)
    
    # Avaliar o classificador
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Perceptron Classifier - {dataset['DESCR'].splitlines()[0]} dataset:")
    print(f"Accuracy: {accuracy:.2f}")
    print(report)
    
    return accuracy

# Ajuste do valor da largura da gaussiana
bandwidth = 1.0  # Você pode alterar este valor

# Tabela de acurácia
accuracy_table = pd.DataFrame(index=['KDE', 'Naive Bayes', 'Perceptron'], columns=datasets.keys())

# Aplicar para cada dataset
for name, dataset in datasets.items():
    print(f"Evaluating {name} dataset:")

    # KDE Classifier
    accuracy_kde = kde_classifier(dataset, bandwidth)
    accuracy_table.loc['KDE', name] = accuracy_kde
    
    # Naive Bayes Classifier
    accuracy_nb = nb_classifier(dataset)
    accuracy_table.loc['Naive Bayes', name] = accuracy_nb
    
    # Perceptron Classifier
    accuracy_perceptron = perceptron_classifier(dataset)
    accuracy_table.loc['Perceptron', name] = accuracy_perceptron

# Mostrar a tabela de acurácia
print("\nAccuracy Table:")
print(accuracy_table)
