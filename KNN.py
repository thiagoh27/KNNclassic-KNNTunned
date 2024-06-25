import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_digits, load_breast_cancer, load_wine
from sklearn.preprocessing import StandardScaler

# Função para plotar KDE para cada dataset
def plot_kde(data, features, target, target_names, title):
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette('hsv', len(target_names))
    for target_value, target_name, color in zip(np.unique(data[target]), target_names, colors):
        subset = data[data[target] == target_value]
        sns.kdeplot(x=subset[features[0]], y=subset[features[1]], 
                    shade=True, alpha=0.5, label=target_name, color=color)

    plt.title(title)
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.legend(title='Classes')
    plt.grid(True)
    plt.show()

# Carregar e processar o dataset Iris
iris = load_iris()
iris_data = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
iris_data['species'] = iris_data['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
plot_kde(iris_data, ['sepal length (cm)', 'sepal width (cm)'], 'target', iris.target_names, 'Distribuição de Densidade KDE para o Dataset Iris')

# Carregar e processar o dataset Digits
digits = load_digits()
digits_data = pd.DataFrame(data=np.c_[digits['data'], digits['target']], columns=[f'pixel_{i}' for i in range(digits['data'].shape[1])] + ['target'])
# Para simplificação, usaremos as primeiras duas características dos dados
plot_kde(digits_data, ['pixel_0', 'pixel_1'], 'target', [str(i) for i in digits.target_names], 'Distribuição de Densidade KDE para o Dataset Digits')

# Carregar e processar o dataset Breast Cancer
cancer = load_breast_cancer()
cancer_data = pd.DataFrame(data=np.c_[cancer['data'], cancer['target']], columns=cancer['feature_names'].tolist() + ['target'])
plot_kde(cancer_data, ['mean radius', 'mean texture'], 'target', cancer.target_names, 'Distribuição de Densidade KDE para o Dataset Breast Cancer')

# Carregar e processar o dataset Wine
wine = load_wine()
wine_data = pd.DataFrame(data=np.c_[wine['data'], wine['target']], columns=wine['feature_names'].tolist() + ['target'])
plot_kde(wine_data, ['alcohol', 'malic_acid'], 'target', wine.target_names, 'Distribuição de Densidade KDE para o Dataset Wine')
