# Passo 1: Importar as bibliotecas necessárias
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Função para plotar gráficos de dispersão
def plot_data(X, y, title):
    plt.figure(figsize=(12, 6))
    for i in range(1, 3):
        for j in range(i+1, 4):
            plt.subplot(1, 3, i)
            sns.scatterplot(x=X[:, i-1], y=X[:, j-1], hue=y, palette='viridis')
            plt.xlabel(f'Feature {i}')
            plt.ylabel(f'Feature {j}')
            plt.title(f'{title} - Feature {i} vs Feature {j}')
    plt.tight_layout()
    plt.show()

# Função para treinar e avaliar o modelo KNN
def train_and_evaluate_knn(X, y, dataset_name):
    # Dividir os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Normalizar os dados
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Treinar o modelo KNN
    knn = KNeighborsClassifier(n_neighbors=3)  # k=3
    knn.fit(X_train, y_train)

    # Fazer previsões
    y_pred = knn.predict(X_test)

    # Avaliar o modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia do modelo no conjunto de dados {dataset_name}: {accuracy * 100:.2f}%")
    print("Relatório de Classificação:")
    print(classification_report(y_test, y_pred))

    # Plotar matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusão - {dataset_name}')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.show()

# Carregar e plotar dados do conjunto Iris
iris = load_iris()
X_iris, y_iris = iris.data, iris.target
plot_data(X_iris, y_iris, 'Iris')

# Treinar e avaliar modelo KNN no conjunto Iris
train_and_evaluate_knn(X_iris, y_iris, 'Iris')

# Carregar e plotar dados do conjunto Wine
wine = load_wine()
X_wine, y_wine = wine.data, wine.target
plot_data(X_wine, y_wine, 'Wine')

# Treinar e avaliar modelo KNN no conjunto Wine
train_and_evaluate_knn(X_wine, y_wine, 'Wine')
