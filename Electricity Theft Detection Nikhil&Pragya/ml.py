import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, roc_curve, precision_recall_curve,
    auc, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.stats import mode

# --- Load and preprocess dataset (use only 200 rows) ---
df = pd.read_csv('C:/Users/LENOVO/OneDrive/Desktop/data.csv')
df = df.sample(n=min(200, len(df)), random_state=42)

df.set_index("CONS_NO", inplace=True)
flags = df["FLAG"]
df.drop(columns=["FLAG"], inplace=True)

df_transposed = df.T.reset_index().rename(columns={"index": "date"})
df_long = df_transposed.melt(id_vars=["date"], var_name="CONS_NO", value_name="reading")
df_long = df_long.merge(flags, on="CONS_NO")
df_long.dropna(inplace=True)

scaler = MinMaxScaler()
df_long['normalized'] = scaler.fit_transform(df_long[['reading']])
df_long['date'] = pd.to_datetime(df_long['date'], errors='coerce')
df_long.dropna(subset=['date'], inplace=True)
df_long.sort_values(by='date', inplace=True)

X = df_long[['normalized']].values
y = df_long['FLAG'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# --- Evaluation Function ---
def evaluate_model(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n{name} Performance:")
    print(f"{'Metric':<12}{'Value'}")
    print(f"{'-'*20}")
    print(f"{'Accuracy':<12}{acc:.4f}")
    print(f"{'Precision':<12}{prec:.4f}")
    print(f"{'Recall':<12}{rec:.4f}")
    print(f"{'F1-Score':<12}{f1:.4f}")
    print(f"{'ROC-AUC':<12}{roc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.title(f"{name} - Confusion Matrix")
    plt.show()

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f'{name} - ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    prec_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall_vals, prec_vals)
    plt.plot(recall_vals, prec_vals, label=f'{name} (AUC = {pr_auc:.2f})', color='green')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{name} - Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

# --- 1. KMeans (Unsupervised) ---
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X_train)
# Map each cluster to the most frequent class in y_train for that cluster
cluster_to_class = {}
for i in range(2):
    mask = (kmeans.labels_ == i)
    if np.any(mask):
        cluster_to_class[i] = mode(y_train[mask], keepdims=False).mode
    else:
        cluster_to_class[i] = 0  # fallback if cluster is empty
kmeans_preds = kmeans.predict(X_test)
# Map predicted clusters to class labels
mapped_kmeans_preds = np.array([cluster_to_class[cluster] for cluster in kmeans_preds])

# Print predicted and true label distributions for debugging
print("KMeans predicted label distribution:", np.unique(mapped_kmeans_preds, return_counts=True))
print("True label distribution:", np.unique(y_test, return_counts=True))

# Alternative mapping: try mapping clusters by mean target value (for robustness)
cluster_means = {}
for i in range(2):
    mask = (kmeans.labels_ == i)
    if np.any(mask):
        cluster_means[i] = np.mean(y_train[mask])
    else:
        cluster_means[i] = 0
# Assign cluster with higher mean to class 1, lower to class 0
if cluster_means[0] > cluster_means[1]:
    alt_cluster_to_class = {0: 1, 1: 0}
else:
    alt_cluster_to_class = {0: 0, 1: 1}
alt_mapped_kmeans_preds = np.array([alt_cluster_to_class[cluster] for cluster in kmeans_preds])
print("KMeans (mean-mapping) predicted label distribution:", np.unique(alt_mapped_kmeans_preds, return_counts=True))
evaluate_model("KMeans", y_test, alt_mapped_kmeans_preds)

# --- 2. CART (Decision Tree) ---
cart = DecisionTreeClassifier(random_state=0)
cart.fit(X_train, y_train)
cart_preds = cart.predict(X_test)
evaluate_model("CART", y_test, cart_preds)

# --- 3. KNN ---
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn_preds = knn.predict(X_test)
evaluate_model("KNN", y_test, knn_preds)
