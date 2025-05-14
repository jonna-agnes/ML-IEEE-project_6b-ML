import pandas as pd
df = pd.read_csv("/content/data.csv")
df.set_index("CONS_NO", inplace=True)
flags = df["FLAG"]
df.drop(columns=["FLAG"], inplace=True)
df_transposed = df.T
df_transposed = df_transposed.reset_index().rename(columns={"index": "date"})
df_long = df_transposed.melt(id_vars=["date"], var_name="CONS_NO", value_name="reading")
df_long = df_long.merge(flags, on="CONS_NO")
df_long.dropna(inplace=True)
from sklearn.preprocessing import MinMaxScaler
import numpy as np
scaler = MinMaxScaler()
df_long['normalized'] = scaler.fit_transform(df_long[['reading']])
df_long['state'] = pd.cut(df_long['normalized'], bins=5, labels=False)
df_long['date'] = pd.to_datetime(df_long['date'], errors='coerce')
df_long.dropna(subset=['date'], inplace=True)
df_long.sort_values(by='date', inplace=True)
split_index = int(len(df_long) * 0.8)
train_df = df_long.iloc[:split_index].reset_index(drop=True)
test_df = df_long.iloc[split_index:].reset_index(drop=True)
import random
num_states = 5
num_actions = 2
Q = np.zeros((num_states, num_actions))
alpha = 0.1
gamma = 0.9
epsilon = 0.1
episodes = 10
train_data = train_df[['state', 'FLAG']].to_numpy()
for ep in range(episodes):
    for i in range(len(train_data) - 1):
        s = int(train_data[i][0])
        true_label = int(train_data[i][1])
        if random.uniform(0, 1) < epsilon:
            a = random.choice([0, 1])
        else:
            a = np.argmax(Q[s])
        s_next = int(train_data[i + 1][0])
        true_next = int(train_data[i + 1][1])
        if random.uniform(0, 1) < epsilon:
            a_next = random.choice([0, 1])
        else:
            a_next = np.argmax(Q[s_next])
        reward = 1 if a == true_label else -1
        Q[s][a] += alpha * (reward + gamma * Q[s_next][a_next] - Q[s][a])
print("Trained Q-table:")
print(Q)
test_data = test_df[['state', 'FLAG']].to_numpy()
correct_predictions = 0
total_predictions = 0
for i in range(len(test_data)):
    s = int(test_data[i][0])
    true_label = int(test_data[i][1])
    a = np.argmax(Q[s])
    if a == true_label:
        correct_predictions += 1
    total_predictions += 1
accuracy = correct_predictions / total_predictions
print(f"Accuracy on test set: {accuracy * 100:.2f}%")
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score, roc_curve
import matplotlib.pyplot as plt
predictions = []
true_labels = test_df['FLAG'].to_numpy()
for i in range(len(test_data)):
    s = int(test_data[i][0])
    a = np.argmax(Q[s])
    predictions.append(a)
accuracy = accuracy_score(true_labels, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")
roc_auc = roc_auc_score(true_labels, predictions)
print(f"ROC-AUC: {roc_auc:.2f}")
precision, recall, _ = precision_recall_curve(true_labels, predictions)
pr_auc = auc(recall, precision)
print(f"Precision-Recall AUC: {pr_auc:.2f}")
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(true_labels, predictions)
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='green', lw=2, label=f'Precision-Recall curve (AUC = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()