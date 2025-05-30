import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_curve, auc
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTEENN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

#---Codificação Variáveis Categóricas---
def encode_categorical_features(df, method='onehot', columns=None, drop_first=True):
    df_result = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if method == 'onehot':
        df_result = pd.get_dummies(df_result, columns=columns, drop_first=drop_first)
    return df_result

#---Técnicas de Balanceamento---
def apply_smote(X, y, random_state=42):
    return SMOTE(random_state=random_state).fit_resample(X, y)

def apply_adasyn(X, y, random_state=42):
    return ADASYN(random_state=random_state).fit_resample(X, y)

def apply_downsampling(X, y, random_state=42): 
    return RandomUnderSampler(random_state=random_state).fit_resample(X, y)

def apply_tomek_links(X, y, random_state=42):
    tl = TomekLinks()
    X_res, y_res = tl.fit_resample(X, y)
    return X_res, y_res

def apply_random_oversampling(X, y, random_state=42):
    """Aplica Random Over-Sampling."""
    ros = RandomOverSampler(random_state=random_state)
    X_res, y_res = ros.fit_resample(X, y)
    return X_res, y_res

def apply_smote_enn(X, y, random_state=42):
    smote_enn = SMOTEENN(random_state=random_state)
    X_res, y_res = smote_enn.fit_resample(X, y)
    return X_res, y_res

def plot_class_distribution(y, *args, labels=None, title='Distribuição de Classes'):
    n_distributions = 1 + len(args)
    fig, axes = plt.subplots(1, n_distributions, figsize=(5 * n_distributions, 5))
    
    if n_distributions == 1:
        axes = [axes] 
    
    if labels is None:
        labels = [f'Original'] + [f'Técnica {i+1}' for i in range(len(args))]
    
    distributions = [y] + list(args)
    
    for i, (dist, label) in enumerate(zip(distributions, labels)):
        unique_classes, counts = np.unique(dist, return_counts=True)
        axes[i].bar(unique_classes, counts, color=sns.color_palette("pastel"))
        axes[i].set_title(label)
        axes[i].set_xlabel('Classe')
        axes[i].set_ylabel('Contagem')
        axes[i].set_xticks(unique_classes) 
        
        for j, count_val in enumerate(counts):
            axes[i].text(unique_classes[j], count_val + (0.01 * max(counts) if counts.any() else 5), str(count_val), ha='center') 
    
    plt.tight_layout()
    return fig

#---Algoritmos de ML---
def create_models_dict(random_state=42, use_class_weights=False):
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=random_state),
        'Random Forest': RandomForestClassifier(random_state=random_state),
        'KNN': KNeighborsClassifier(),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state, solver='liblinear'),
        'Neural Network': MLPClassifier(max_iter=1000, random_state=random_state, early_stopping=True, n_iter_no_change=10),
        'Gradient Boosting': GradientBoostingClassifier(random_state=random_state)
    }

    if use_class_weights:
        if 'Decision Tree' in models:
            models['Decision Tree'] = DecisionTreeClassifier(random_state=random_state, class_weight='balanced')
        if 'Random Forest' in models:
            models['Random Forest'] = RandomForestClassifier(random_state=random_state, class_weight='balanced')
        if 'Logistic Regression' in models:
            models['Logistic Regression'] = LogisticRegression(max_iter=1000, random_state=random_state, class_weight='balanced', solver='liblinear')
    return models

def evaluate_model(y_test, y_pred):
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }

def train_all_models(X_train, y_train, random_state=42, use_class_weights=False):
    models = create_models_dict(random_state=random_state, use_class_weights=use_class_weights)
    for model_name, model_instance in models.items():
        model_instance.fit(X_train, y_train)
    return models

#---Validação Cruzada---
def perform_cross_validation(model, X, y, cv=5, random_state=42):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    metrics = ['accuracy', 'f1'] # Removed roc_auc
    cv_results = {}
    
    for metric in metrics:
        try:
            scores = cross_val_score(model, X, y, cv=skf, scoring=metric)
            cv_results[metric] = {
                'scores': scores,
                'mean': scores.mean(),
                'std': scores.std()
            }
        except Exception as e:
            cv_results[metric] = {
                'scores': np.array([]),
                'mean': np.nan,
                'std': np.nan
            }
    return cv_results

def plot_cv_results(cv_results_dict, figsize=(15, 5)):
    if not cv_results_dict:
        return None
    
    all_metrics = set()
    for results in cv_results_dict.values():
        all_metrics.update(results.keys())
    
    metrics_to_plot = []
    for m in sorted(list(all_metrics)):
        for model_results in cv_results_dict.values():
            if m in model_results and isinstance(model_results[m], dict):
                metrics_to_plot.append(m)
                break
    
    if not metrics_to_plot:
        return None
    
    data = []
    for model_name, results in cv_results_dict.items():
        for metric_name in metrics_to_plot:
            if (metric_name in results and 
                isinstance(results[metric_name], dict) and 
                'mean' in results[metric_name] and 
                not np.isnan(results[metric_name]['mean'])):
                
                data.append({
                    'Modelo': model_name,
                    'Métrica': metric_name,
                    'Pontuação Média': results[metric_name]['mean'],
                    'Desvio Padrão': results[metric_name].get('std', np.nan)
                })

    if not data:
        return None
    
    df_results = pd.DataFrame(data)
    actual_metrics_in_df = df_results['Métrica'].unique()
    num_actual_metrics = len(actual_metrics_in_df)
    
    if num_actual_metrics == 0:
        return None
    
    width = 6 * num_actual_metrics
    height = figsize[1] if isinstance(figsize, tuple) and len(figsize) > 1 else 5
       
    fig, axes = plt.subplots(1, num_actual_metrics, figsize=(width, height))
    
    if num_actual_metrics == 1:
        axes = [axes]
    elif not isinstance(axes, np.ndarray):
        axes = [axes]
    
    for i, metric_name_to_plot in enumerate(actual_metrics_in_df):
        if i < len(axes):  
            metric_data = df_results[df_results['Métrica'] == metric_name_to_plot].sort_values(by='Pontuação Média', ascending=False)
            if not metric_data.empty:
                bars = sns.barplot(x='Modelo', y='Pontuação Média', hue='Modelo', data=metric_data, ax=axes[i], palette="viridis", legend=False)
                axes[i].set_title(f'Validação Cruzada - {metric_name_to_plot.capitalize()}')
                
                max_value = metric_data['Pontuação Média'].max()
                axes[i].set_ylim(0, max(1.0, max_value * 1.1))
                
                plt.setp(axes[i].get_xticklabels(), rotation=45)
                
                for bar_patch in bars.patches:
                    value = bar_patch.get_height()
                    axes[i].text(
                        bar_patch.get_x() + bar_patch.get_width() / 2., 
                        value + 0.01, 
                        f'{value:.3f}', 
                        ha='center', 
                        va='bottom', 
                        fontsize=9
                    )
    
    fig.suptitle('Comparação de Modelos (Validação Cruzada)', fontsize=16, y=1.03)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig

#---Curvas de ROC---
def plot_roc_curves(models, X_test, y_test, figsize=(6, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    for name, model in models.items():
        if hasattr(model, 'predict_proba'):
            try:
                y_score = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_score)
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
            except Exception:
                pass 
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Taxa de Falsos Positivos (1 - Especificidade)')
    ax.set_ylabel('Taxa de Verdadeiros Positivos (Sensibilidade/Recall)')
    ax.set_title('Curvas ROC dos Modelos no Conjunto de Teste')
    ax.legend(loc="lower right")
    return fig

    
#---Matriz de Confusão---
def plot_confusion_matrix(models_dict, X_test, y_test):
    model_names = list(models_dict.keys())
    n_models = len(model_names)
    n_cols = 3 
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    for i, name in enumerate(model_names):
        model = models_dict[name]
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
        axes[i].set_title(f'{name}')
        axes[i].set_xlabel('Previsto')
        axes[i].set_ylabel('Real')

    for j in range(n_models, n_rows * n_cols):
        fig.delaxes(axes[j])
    
    fig.suptitle('Matrizes de Confusão por Modelo', fontsize=16, y=1.03)
    plt.tight_layout(rect=[0, 0, 1, 0.98]) 
    return fig

plot_confusion_matrices = plot_confusion_matrix

#---Característica Mais Importante---
def plot_feature_importance(model, feature_names, figsize=(10, 6)):
    """Plota a importância das características para modelos baseados em árvores."""
    if hasattr(model, 'feature_importances_'):
        # Obter importância das características
        importances = model.feature_importances_
        
        # Criar DataFrame para visualização
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Ordenar por importância
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        # Criar figura
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plotar gráfico de barras
        sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
        
        # Configurar título e labels
        ax.set_title('Importância das Características')
        ax.set_xlabel('Importância')
        ax.set_ylabel('Característica')
        
        plt.tight_layout()
        return fig
    else:
        print("O modelo não possui atributo 'feature_importances_'")
        return None

#---Clustering---
# Normalização dos dados
def normalize_data(X):
    """
    Normaliza os dados usando StandardScaler.
    
    Args:
        X (DataFrame ou array): Dados a serem normalizados
        
    Returns:
        array: Dados normalizados
        StandardScaler: Objeto scaler treinado
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

# Redução de dimensionalidade
def apply_pca(X, n_components=2):
    """
    Aplica PCA para redução de dimensionalidade.
    
    Args:
        X (array): Dados normalizados
        n_components (int): Número de componentes para reduzir
        
    Returns:
        array: Dados reduzidos
        PCA: Objeto PCA treinado
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, pca

def apply_tsne(X, n_components=2, perplexity=30, random_state=42):
    """
    Aplica t-SNE para redução de dimensionalidade.
    
    Args:
        X (array): Dados normalizados
        n_components (int): Número de componentes para reduzir
        perplexity (int): Parâmetro de perplexidade do t-SNE
        random_state (int): Semente aleatória
        
    Returns:
        array: Dados reduzidos
    """
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    X_tsne = tsne.fit_transform(X)
    return X_tsne

# Algoritmos de Clustering
def apply_kmeans(X, n_clusters=3, random_state=42):
    """
    Aplica o algoritmo K-Means.
    
    Args:
        X (array): Dados normalizados
        n_clusters (int): Número de clusters
        random_state (int): Semente aleatória
        
    Returns:
        array: Labels dos clusters
        KMeans: Objeto KMeans treinado
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X)
    return labels, kmeans

def apply_dbscan(X, eps=0.5, min_samples=5):
    """
    Aplica o algoritmo DBSCAN.
    
    Args:
        X (array): Dados normalizados
        eps (float): Distância máxima entre amostras
        min_samples (int): Número mínimo de amostras em uma vizinhança
        
    Returns:
        array: Labels dos clusters
        DBSCAN: Objeto DBSCAN treinado
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    return labels, dbscan

def apply_agglomerative(X, n_clusters=3, linkage='ward'):
    """
    Aplica o algoritmo Agglomerative Clustering.
    
    Args:
        X (array): Dados normalizados
        n_clusters (int): Número de clusters
        linkage (str): Critério de linkage
        
    Returns:
        array: Labels dos clusters
        AgglomerativeClustering: Objeto AgglomerativeClustering treinado
    """
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = agg.fit_predict(X)
    return labels, agg

# Avaliação de Clusters
def evaluate_silhouette(X, labels):
    """
    Calcula o score de silhueta para avaliar a qualidade dos clusters.
    
    Args:
        X (array): Dados normalizados
        labels (array): Labels dos clusters
        
    Returns:
        float: Score de silhueta
    """
    # Ignorar ruído (cluster -1) se existir
    if -1 in labels:
        mask = labels != -1
        if np.sum(mask) <= 1:  # Se houver apenas uma amostra válida ou nenhuma
            return 0
        return silhouette_score(X[mask], labels[mask])
    
    # Se houver apenas um cluster, retornar 0
    if len(np.unique(labels)) <= 1:
        return 0
    
    return silhouette_score(X, labels)

# Visualização de Clusters
def plot_clusters_2d(X_2d, labels, title="Visualização de Clusters", figsize=(10, 8)):
    """
    Visualiza clusters em 2D.
    
    Args:
        X_2d (array): Dados reduzidos para 2D
        labels (array): Labels dos clusters
        title (str): Título do gráfico
        figsize (tuple): Tamanho da figura
        
    Returns:
        matplotlib.figure.Figure: Objeto figura
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Definir uma paleta de cores que inclui preto para ruído (-1)
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    # Plotar cada cluster
    for i, label in enumerate(unique_labels):
        if label == -1:
            # Ruído em preto
            color = [0, 0, 0, 1]
            marker = 'x'
        else:
            color = colors[i]
            marker = 'o'
        
        mask = labels == label
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=[color], marker=marker,
            label=f'Cluster {label}' if label != -1 else 'Ruído',
            alpha=0.7
        )
    
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig

# Análise de Clusters
def analyze_clusters(df, labels, target_col='passed'):
    """
    Analisa a proporção de classes em cada cluster.
    
    Args:
        df (DataFrame): DataFrame original
        labels (array): Labels dos clusters
        target_col (str): Nome da coluna alvo
        
    Returns:
        DataFrame: DataFrame com análise de proporções
    """
    # Criar DataFrame com labels
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = labels
    
    # Calcular proporções por cluster
    cluster_analysis = []
    
    for cluster in np.unique(labels):
        cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster]
        total = len(cluster_data)
        
        if total == 0:
            continue
            
        # Contar valores na coluna alvo
        value_counts = cluster_data[target_col].value_counts()
        
        # Criar dicionário com proporções
        cluster_info = {'Cluster': cluster, 'Total': total}
        
        for value, count in value_counts.items():
            proportion = count / total * 100
            cluster_info[f'{value}'] = count
            cluster_info[f'{value} (%)'] = proportion
            
        cluster_analysis.append(cluster_info)
    
    # Converter para DataFrame
    return pd.DataFrame(cluster_analysis)

def describe_clusters(df, labels, features, n_top_features=5):
    """
    Descreve as principais características de cada cluster.
    
    Args:
        df (DataFrame): DataFrame original
        labels (array): Labels dos clusters
        features (list): Lista de características para analisar
        n_top_features (int): Número de características principais a mostrar
        
    Returns:
        dict: Dicionário com descrição dos clusters
    """
    # Criar DataFrame com labels
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = labels
    
    # Calcular médias por cluster
    cluster_means = df_with_clusters.groupby('cluster')[features].mean()
    
    # Calcular médias globais
    global_means = df[features].mean()
    
    # Calcular diferenças relativas
    relative_importance = cluster_means.copy()
    for feature in features:
        if global_means[feature] != 0:
            relative_importance[feature] = (cluster_means[feature] - global_means[feature]) / global_means[feature]
        else:
            relative_importance[feature] = cluster_means[feature]
    
    # Encontrar características mais importantes por cluster
    cluster_descriptions = {}
    
    for cluster in np.unique(labels):
        if cluster == -1:  # Ignorar ruído
            continue
            
        # Ordenar características por importância (absoluta)
        if cluster in relative_importance.index:
            sorted_features = relative_importance.loc[cluster].abs().sort_values(ascending=False)
            top_features = sorted_features.head(n_top_features)
            
            # Criar descrição
            description = []
            for feature in top_features.index:
                value = cluster_means.loc[cluster, feature]
                rel_imp = relative_importance.loc[cluster, feature]
                
                if rel_imp > 0:
                    direction = "acima da"
                else:
                    direction = "abaixo da"
                    
                description.append(f"{feature}: {value:.2f} ({abs(rel_imp)*100:.1f}% {direction} média)")
            
            cluster_descriptions[cluster] = description
    
    return cluster_descriptions

def plot_cluster_feature_importance(df, labels, features, figsize=(12, 8)):
    """
    Visualiza a importância das características para cada cluster.
    
    Args:
        df (DataFrame): DataFrame original
        labels (array): Labels dos clusters
        features (list): Lista de características para analisar
        figsize (tuple): Tamanho da figura
        
    Returns:
        matplotlib.figure.Figure: Objeto figura
    """
    # Criar DataFrame com labels
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = labels
    
    # Calcular médias por cluster
    cluster_means = df_with_clusters.groupby('cluster')[features].mean()
    
    # Calcular médias globais
    global_means = df[features].mean()
    
    # Calcular diferenças relativas
    for feature in features:
        if global_means[feature] != 0:
            cluster_means[f"{feature}_rel"] = (cluster_means[feature] - global_means[feature]) / global_means[feature]
        else:
            cluster_means[f"{feature}_rel"] = cluster_means[feature]
    
    # Selecionar apenas colunas relativas
    rel_columns = [f"{feature}_rel" for feature in features]
    rel_data = cluster_means[rel_columns]
    
    # Renomear colunas para nomes originais
    rel_data.columns = features
    
    # Criar heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(rel_data, cmap="coolwarm", center=0, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Importância Relativa das Características por Cluster")
    plt.tight_layout()
    
    return fig
