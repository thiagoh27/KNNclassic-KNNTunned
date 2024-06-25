import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KernelDensity
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Função para treinar e avaliar o classificador KDE
def kde_classifier(X_train, X_test, y_train, y_test, bandwidth):
    start_time = time.time()
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
    end_time = time.time()
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))
    
    return accuracy, end_time - start_time

# Função para treinar e avaliar o classificador Naive Bayes
def nb_classifier(X_train, X_test, y_train, y_test):
    start_time = time.time()
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    end_time = time.time()
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))
    return accuracy, end_time - start_time

# Função para treinar e avaliar o classificador Perceptron
def perceptron_classifier(X_train, X_test, y_train, y_test):
    start_time = time.time()
    perceptron = Perceptron()
    perceptron.fit(X_train, y_train)
    y_pred = perceptron.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    end_time = time.time()
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))
    return accuracy, end_time - start_time

# Função para treinar e avaliar o modelo KNN clássico
def train_and_evaluate_knn_classic(X_train, X_test, y_train, y_test):
    start_time = time.time()
    knn_classic = KNeighborsClassifier(n_neighbors=3)
    knn_classic.fit(X_train, y_train)
    y_pred_classic = knn_classic.predict(X_test)
    accuracy_classic = accuracy_score(y_test, y_pred_classic)
    end_time = time.time()
    print(f"Accuracy: {accuracy_classic * 100:.2f}%")
    print(classification_report(y_test, y_pred_classic))
    return accuracy_classic, end_time - start_time

# Função para treinar e avaliar o modelo KNN ponderado com GridSearchCV
def train_and_evaluate_knn_weighted(X_train, X_test, y_train, y_test):
    start_time = time.time()
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
    end_time = time.time()
    print(f"Accuracy: {accuracy_weighted * 100:.2f}%")
    print(classification_report(y_test, y_pred_weighted))
    return accuracy_weighted, end_time - start_time

# Função para treinar e avaliar o modelo KNN ponderado com pesos de densidades KDE
def train_and_evaluate_knn_kde_weighted(X_train, X_test, y_train, y_test, bandwidth):
    start_time = time.time()
    kde_models = {}
    for class_label in np.unique(y_train):
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
        kde.fit(X_train[y_train == class_label])
        kde_models[class_label] = kde

    # Calcular as densidades dos pontos de teste
    def kde_density_weights(X):
        return np.exp(np.array([kde_models[class_label].score_samples(X) for class_label in kde_models]).T)

    test_weights = kde_density_weights(X_test)
    train_weights = kde_density_weights(X_train)

    # Ajustar manualmente as distâncias usando as densidades como pesos
    class CustomKNN(KNeighborsClassifier):
        def predict(self, X):
            distances, indices = self.kneighbors(X)
            y_pred = np.zeros(X.shape[0], dtype=int)
            for i in range(X.shape[0]):
                weighted_votes = np.zeros(len(self.classes_))
                for j, idx in enumerate(indices[i]):
                    class_idx = self._y[idx]
                    distance = distances[i][j]
                    weight = train_weights[idx, class_idx]  # usar pesos de densidade
                    weighted_votes[class_idx] += weight / distance
                y_pred[i] = np.argmax(weighted_votes)
            return y_pred

    knn = CustomKNN(n_neighbors=7)
    knn.fit(X_train, y_train)
    y_pred_weighted = knn.predict(X_test)
    
    accuracy_weighted = accuracy_score(y_test, y_pred_weighted)
    end_time = time.time()
    print(f"Accuracy: {accuracy_weighted * 100:.2f}%")
    print(classification_report(y_test, y_pred_weighted))
    
    # Matriz de confusão
    conf_matrix = confusion_matrix(y_test, y_pred_weighted)
    print(f"Confusion Matrix:\n{conf_matrix}")
    
    return accuracy_weighted, end_time - start_time, conf_matrix

# Função para plotar as densidades em 2D do KDE
def plot_kde_2d(X_train, y_train, bandwidth):
    plt.figure(figsize=(10, 7))
    colors = ['r', 'g', 'b']
    labels = np.unique(y_train)
    
    for class_label, color in zip(labels, colors):
        X = X_train[y_train == class_label]
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
        kde.fit(X)
        
        # Gerar um grid de pontos
        X_range = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100)
        Y_range = np.linspace(X_train[:, 1].min(), X_train[:, 1].max(), 100)
        X_grid, Y_grid = np.meshgrid(X_range, Y_range)
        grid_samples = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T
        
        Z = np.exp(kde.score_samples(grid_samples)).reshape(X_grid.shape)
        
        plt.contourf(X_grid, Y_grid, Z, alpha=0.5, cmap='coolwarm')
        plt.scatter(X[:, 0], X[:, 1], c=color, label=f'Class {class_label}')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title('Densidade 2D KDE')
    plt.colorbar()
    plt.show()

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
bandwidth = 1.0  # Ajuste conforme necessário

# Tabela de resultados
results_table = pd.DataFrame(index=['KDE', 'Naive Bayes', 'Perceptron', 'KNN Clássico', 'KNN Ponderado', 'KNN KDE-Ponderado'],
                             columns=pd.MultiIndex.from_product([datasets.keys(), ['Accuracy', 'Time (s)']]))

# Aplicar para cada dataset
for name, dataset in datasets.items():
    print(f"Evaluating {name} dataset:")
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    if X_train.shape[1] > 2:
        print(f"Dataset {name} tem mais de 2 características. Apenas 2 serão usadas para a plotagem do KDE 2D.")
        X_train_2d = X_train[:, :2]
        X_test_2d = X_test[:, :2]
    else:
        X_train_2d = X_train
        X_test_2d = X_test

    # KDE Classifier
    print(f"KDE Classifier - {name} dataset with bandwidth {bandwidth}:")
    accuracy_kde, time_kde = kde_classifier(X_train, X_test, y_train, y_test, bandwidth)
    results_table.loc['KDE', (name, 'Accuracy')] = accuracy_kde
    results_table.loc['KDE', (name, 'Time (s)')] = time_kde
    
    # Naive Bayes Classifier
    print(f"Naive Bayes Classifier - {name} dataset:")
    accuracy_nb, time_nb = nb_classifier(X_train, X_test, y_train, y_test)
    results_table.loc['Naive Bayes', (name, 'Accuracy')] = accuracy_nb
    results_table.loc['Naive Bayes', (name, 'Time (s)')] = time_nb
    
    # Perceptron Classifier
    print(f"Perceptron Classifier - {name} dataset:")
    accuracy_perceptron, time_perceptron = perceptron_classifier(X_train, X_test, y_train, y_test)
    results_table.loc['Perceptron', (name, 'Accuracy')] = accuracy_perceptron
    results_table.loc['Perceptron', (name, 'Time (s)')] = time_perceptron
    
    # KNN Clássico
    print(f"KNN Clássico - {name} dataset:")
    accuracy_knn_classic, time_knn_classic = train_and_evaluate_knn_classic(X_train, X_test, y_train, y_test)
    results_table.loc['KNN Clássico', (name, 'Accuracy')] = accuracy_knn_classic
    results_table.loc['KNN Clássico', (name, 'Time (s)')] = time_knn_classic
    
    # KNN Ponderado
    print(f"KNN Ponderado - {name} dataset:")
    accuracy_knn_weighted, time_knn_weighted = train_and_evaluate_knn_weighted(X_train, X_test, y_train, y_test)
    results_table.loc['KNN Ponderado', (name, 'Accuracy')] = accuracy_knn_weighted
    results_table.loc['KNN Ponderado', (name, 'Time (s)')] = time_knn_weighted
    
    # KNN KDE-Ponderado
    print(f"KNN KDE-Ponderado - {name} dataset with bandwidth {bandwidth}:")
    accuracy_knn_kde_weighted, time_knn_kde_weighted, conf_matrix = train_and_evaluate_knn_kde_weighted(X_train, X_test, y_train, y_test, bandwidth)
    results_table.loc['KNN KDE-Ponderado', (name, 'Accuracy')] = accuracy_knn_kde_weighted
    results_table.loc['KNN KDE-Ponderado', (name, 'Time (s)')] = time_knn_kde_weighted

    # Plotar as densidades em 2D do KDE
    plot_kde_2d(X_train_2d, y_train, bandwidth)

    # Plotar a matriz de confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title(f'Confusion Matrix - {name} dataset (KNN KDE-Ponderado)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Calcular a média de acurácia e tempo para cada classificador
results_table['Mean Accuracy'] = results_table.xs('Accuracy', axis=1, level=1).mean(axis=1)
results_table['Mean Time (s)'] = results_table.xs('Time (s)', axis=1, level=1).mean(axis=1)

# Ordenar os classificadores pela média de acurácia
results_table = results_table.sort_values(by='Mean Accuracy', ascending=False)

## Mostrar a tabela de resultados
print("\nResults Table:")
print(results_table)
