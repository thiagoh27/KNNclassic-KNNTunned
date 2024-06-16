import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Função para plotar gráficos 3D
def plot_3d(X, y_true, y_pred, title, feature_names):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    incorrect = (y_true != y_pred)
    correct = (y_true == y_pred)

    # Plotar pontos corretos
    scatter_correct = ax.scatter(X[correct, 0], X[correct, 1], X[correct, 2], c=y_true[correct], cmap='viridis', marker='o')

    # Plotar pontos incorretos em vermelho
    scatter_incorrect = ax.scatter(X[incorrect, 0], X[incorrect, 1], X[incorrect, 2], c='red', marker='x', label='Incorreto')

    # Adicionar legenda
    legend_correct = ax.legend(*scatter_correct.legend_elements(), title="Classes", loc="upper right")
    ax.add_artist(legend_correct)
    legend_incorrect = ax.legend(['Incorreto'], loc='upper left')
    ax.add_artist(legend_incorrect)

    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_zlabel(feature_names[2])
    ax.set_title(title)
    plt.show()

# Função para treinar e avaliar o modelo KNN clássico
def train_and_evaluate_knn_classic(X_train, X_test, y_train, y_test, dataset_name):
    # Treinar o modelo KNN clássico
    knn_classic = KNeighborsClassifier(n_neighbors=3)
    knn_classic.fit(X_train, y_train)

    # Fazer previsões
    y_pred_classic = knn_classic.predict(X_test)

    # Avaliar o modelo
    accuracy_classic = accuracy_score(y_test, y_pred_classic)
    print(f"Acurácia do KNN Clássico no conjunto de dados {dataset_name}: {accuracy_classic * 100:.2f}%")
    print("Relatório de Classificação do KNN Clássico:")
    print(classification_report(y_test, y_pred_classic))

    return y_pred_classic, accuracy_classic

# Função para treinar e avaliar o modelo KNN ponderado com GridSearchCV
def train_and_evaluate_knn_weighted(X_train, X_test, y_train, y_test, dataset_name):
    # Definir os hiperparâmetros para o GridSearchCV
    param_grid = {
        'n_neighbors': np.arange(1, 32),
        'weights': ['uniform', 'distance']
    }

    # Inicializar o modelo KNN
    knn = KNeighborsClassifier()

    # Inicializar o GridSearchCV
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Melhor combinação de hiperparâmetros
    best_params = grid_search.best_params_
    print(f"Melhores parâmetros para o KNN Ponderado no conjunto de dados {dataset_name}: {best_params}")

    # Treinar o modelo KNN com os melhores parâmetros
    knn_best = KNeighborsClassifier(**best_params)
    knn_best.fit(X_train, y_train)

    # Fazer previsões
    y_pred_weighted = knn_best.predict(X_test)

    # Avaliar o modelo
    accuracy_weighted = accuracy_score(y_test, y_pred_weighted)
    print(f"Acurácia do KNN Ponderado no conjunto de dados {dataset_name}: {accuracy_weighted * 100:.2f}%")
    print("Relatório de Classificação do KNN Ponderado:")
    print(classification_report(y_test, y_pred_weighted))

    return y_pred_weighted, accuracy_weighted

# Função principal para comparar ambos os modelos
def compare_knn_models(X, y, dataset_name, feature_names):
    # Dividir os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Normalizar os dados
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Avaliar KNN Clássico
    y_pred_classic, accuracy_classic = train_and_evaluate_knn_classic(X_train, X_test, y_train, y_test, dataset_name)

    # Avaliar KNN Ponderado
    y_pred_weighted, accuracy_weighted = train_and_evaluate_knn_weighted(X_train, X_test, y_train, y_test, dataset_name)

    # Plotar matriz de confusão para KNN Clássico
    cm_classic = confusion_matrix(y_test, y_pred_classic)
    sns.heatmap(cm_classic, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusão - KNN Clássico - {dataset_name}')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.show()

    # Plotar matriz de confusão para KNN Ponderado
    cm_weighted = confusion_matrix(y_test, y_pred_weighted)
    sns.heatmap(cm_weighted, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusão - KNN Ponderado - {dataset_name}')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.show()

    # Plotar dados de teste em 3D para KNN Clássico
    if X_test.shape[1] >= 3:
        plot_3d(X_test, y_test, y_pred_classic, f'{dataset_name} - KNN Clássico - Dados de Teste', feature_names)

    # Plotar dados de teste em 3D para KNN Ponderado
    if X_test.shape[1] >= 3:
        plot_3d(X_test, y_test, y_pred_weighted, f'{dataset_name} - KNN Ponderado - Dados de Teste', feature_names)

    # Comparar acurácias
    print(f"Acurácia do KNN Clássico no conjunto de dados {dataset_name}: {accuracy_classic * 100:.2f}%")
    print(f"Acurácia do KNN Ponderado no conjunto de dados {dataset_name}: {accuracy_weighted * 100:.2f}%")

    return accuracy_classic, accuracy_weighted

# Lista para armazenar as acurácias
results = []

# Função para adicionar resultados à lista
def add_result(dataset_name, accuracy_classic, accuracy_weighted):
    results.append({
        'Dataset': dataset_name,
        'KNN Clássico': f'{accuracy_classic * 100:.2f}%',
        'KNN Ponderado com Grid Search': f'{accuracy_weighted * 100:.2f}%'
    })

# Carregar e plotar dados do conjunto Iris
iris = load_iris()
X_iris, y_iris = iris.data, iris.target
accuracy_classic, accuracy_weighted = compare_knn_models(X_iris, y_iris, 'Iris', iris.feature_names)
add_result('Iris', accuracy_classic, accuracy_weighted)

# Carregar e plotar dados do conjunto Wine
wine = load_wine()
X_wine, y_wine = wine.data, wine.target
accuracy_classic, accuracy_weighted = compare_knn_models(X_wine, y_wine, 'Wine', wine.feature_names)
add_result('Wine', accuracy_classic, accuracy_weighted)

# Carregar e plotar dados do conjunto Breast Cancer
breast_cancer = load_breast_cancer()
X_cancer, y_cancer = breast_cancer.data, breast_cancer.target
accuracy_classic, accuracy_weighted = compare_knn_models(X_cancer, y_cancer, 'Breast Cancer', breast_cancer.feature_names)
add_result('Breast Cancer', accuracy_classic, accuracy_weighted)

# Carregar e plotar dados do conjunto Digits
digits = load_digits()
X_digits, y_digits = digits.data, digits.target
accuracy_classic, accuracy_weighted = compare_knn_models(X_digits, y_digits, 'Digits', [f'Pixel {i}' for i in range(X_digits.shape[1])])
add_result('Digits', accuracy_classic, accuracy_weighted)

# Carregar e plotar dados do conjunto Lung Cancer
#lung_cancer = fetch_openml(name="lung-cancer", version=1, as_frame=False)
#X_lung, y_lung = lung_cancer.data, lung_cancer.target.astype(int)  # Convert y to integer
#accuracy_classic, accuracy_weighted = compare_knn_models(X_lung, y_lung, 'Lung Cancer', lung_cancer.feature_names)
#add_result('Lung Cancer', accuracy_classic, accuracy_weighted)

# Mostrar tabela de acurácias
results_df = pd.DataFrame(results)
print("\nTabela de Acurácias:")
print(results_df)
