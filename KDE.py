import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KernelDensity
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Função para treinar e avaliar o classificador KDE
def kde_classifier(X_train, X_test, y_train, y_test, bandwidth):
    kde_models = {}
    for class_label in np.unique(y_train):
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
        kde.fit(X_train[y_train == class_label])
        kde_models[class_label] = kde
    
    def predict(X):
        log_probs = np.array([kde_models[class_label].score_samples(X) for class_label in kde_models])
        return np.argmax(log_probs, axis=0)
    
    y_pred = predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))
    
    return accuracy

# Função para treinar e avaliar o classificador Naive Bayes
def nb_classifier(X_train, X_test, y_train, y_test):
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))
    return accuracy

# Função para treinar e avaliar o classificador Perceptron
def perceptron_classifier(X_train, X_test, y_train, y_test):
    perceptron = Perceptron()
    perceptron.fit(X_train, y_train)
    y_pred = perceptron.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))
    return accuracy

# Função para treinar e avaliar o modelo KNN clássico
def train_and_evaluate_knn_classic(X_train, X_test, y_train, y_test):
    knn_classic = KNeighborsClassifier(n_neighbors=3)
    knn_classic.fit(X_train, y_train)
    y_pred_classic = knn_classic.predict(X_test)
    accuracy_classic = accuracy_score(y_test, y_pred_classic)
    print(f"Accuracy: {accuracy_classic * 100:.2f}%")
    print(classification_report(y_test, y_pred_classic))
    return accuracy_classic

# Função para treinar e avaliar o modelo KNN ponderado com GridSearchCV
def train_and_evaluate_knn_weighted(X_train, X_test, y_train, y_test):
    param_grid = {
        'n_neighbors': np.arange(1, 32),
        'weights': ['uniform', 'distance']
    }
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")
    knn_best = KNeighborsClassifier(**best_params)
    knn_best.fit(X_train, y_train)
    y_pred_weighted = knn_best.predict(X_test)
    accuracy_weighted = accuracy_score(y_test, y_pred_weighted)
    print(f"Accuracy: {accuracy_weighted * 100:.2f}%")
    print(classification_report(y_test, y_pred_weighted))
    return accuracy_weighted

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

# Ajuste do valor da largura da gaussiana
bandwidth = 1.0  # Você pode alterar este valor

# Tabela de acurácia
accuracy_table = pd.DataFrame(index=['KDE', 'Naive Bayes', 'Perceptron', 'KNN Clássico', 'KNN Ponderado'], columns=datasets.keys())

# Aplicar para cada dataset
for name, dataset in datasets.items():
    print(f"Evaluating {name} dataset:")
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # KDE Classifier
    print(f"KDE Classifier - {name} dataset with bandwidth {bandwidth}:")
    accuracy_kde = kde_classifier(X_train, X_test, y_train, y_test, bandwidth)
    accuracy_table.loc['KDE', name] = accuracy_kde
    
    # Naive Bayes Classifier
    print(f"Naive Bayes Classifier - {name} dataset:")
    accuracy_nb = nb_classifier(X_train, X_test, y_train, y_test)
    accuracy_table.loc['Naive Bayes', name] = accuracy_nb
    
    # Perceptron Classifier
    print(f"Perceptron Classifier - {name} dataset:")
    accuracy_perceptron = perceptron_classifier(X_train, X_test, y_train, y_test)
    accuracy_table.loc['Perceptron', name] = accuracy_perceptron
    
    # KNN Clássico
    print(f"KNN Clássico - {name} dataset:")
    accuracy_knn_classic = train_and_evaluate_knn_classic(X_train, X_test, y_train, y_test)
    accuracy_table.loc['KNN Clássico', name] = accuracy_knn_classic
    
    # KNN Ponderado
    print(f"KNN Ponderado - {name} dataset:")
    accuracy_knn_weighted = train_and_evaluate_knn_weighted(X_train, X_test, y_train, y_test)
    accuracy_table.loc['KNN Ponderado', name] = accuracy_knn_weighted

# Mostrar a tabela de acurácia
print("\nAccuracy Table:")
print(accuracy_table)
