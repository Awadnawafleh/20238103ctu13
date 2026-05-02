# Botnet Detection on CTU-13 dataset
# Unsupervised learning: K-Means, DBSCAN, Isolation Forest

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score,
                             silhouette_score)


# ----- 1. Load the data -----
df = pd.read_csv("capture20110815.binetflow")
print("Data shape:", df.shape)
print(df.head())

# make a binary label: 1 = Botnet, 0 = Normal/Background
df["is_botnet"] = df["Label"].str.lower().str.contains("botnet").astype(int)
print("\nBotnet flows:", df["is_botnet"].sum())
print("Normal flows:", (df["is_botnet"] == 0).sum())


# ----- 2. Quick exploration -----
print("\nMissing values per column:")
print(df.isnull().sum())

# label distribution plot
df["Label"].value_counts().head(10).plot(kind="bar", figsize=(8,4))
plt.title("Top 10 labels")
plt.tight_layout()
plt.savefig("fig_labels.png")
plt.close()


# ----- 3. Preprocessing -----
# pick only the useful columns (we drop IPs and times)
data = df[["Dur", "Proto", "Sport", "Dport", "State",
           "TotPkts", "TotBytes", "SrcBytes"]].copy()

# fix port columns (some values are hex like 0x303)
data["Sport"] = pd.to_numeric(data["Sport"], errors="coerce")
data["Dport"] = pd.to_numeric(data["Dport"], errors="coerce")

# fill missing with 0 (simple approach)
data = data.fillna(0)

# encode categorical columns
for col in ["Proto", "State"]:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))

# add 2 simple features
data["BytesPerPkt"] = data["TotBytes"] / (data["TotPkts"] + 1)
data["PktsPerSec"]  = data["TotPkts"]  / (data["Dur"] + 0.001)

# scale
scaler = StandardScaler()
X = scaler.fit_transform(data)
y = df["is_botnet"].values

print("\nFeatures used:", list(data.columns))
print("X shape:", X.shape)


# ----- 4. Find best k for K-Means -----
# we try k from 2 to 6 and look at inertia (elbow) and silhouette

# silhouette is slow so we sample 10000 points
np.random.seed(42)
sample_idx = np.random.choice(len(X), 10000, replace=False)

ks = [2, 3, 4, 5, 6]
inertias = []
sils = []

for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    s = silhouette_score(X[sample_idx], labels[sample_idx])
    sils.append(s)
    print(f"k={k}  inertia={km.inertia_:.0f}  silhouette={s:.3f}")

# plot elbow + silhouette
fig, axes = plt.subplots(1, 2, figsize=(11,4))
axes[0].plot(ks, inertias, marker="o")
axes[0].set_title("Elbow method")
axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia")
axes[1].plot(ks, sils, marker="o", color="green")
axes[1].set_title("Silhouette score")
axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette")
plt.tight_layout()
plt.savefig("fig_elbow.png")
plt.close()

# pick best k by silhouette
best_k = ks[int(np.argmax(sils))]
print("\nBest k =", best_k)


# ----- 5. K-Means with the best k -----
km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
km_labels = km.fit_predict(X)

# the cluster with the most botnet ratio is "the malicious one"
ratios = []
for c in range(best_k):
    mask = km_labels == c
    if mask.sum() > 0:
        ratios.append(y[mask].mean())
    else:
        ratios.append(0)
mal_cluster = int(np.argmax(ratios))
km_pred = (km_labels == mal_cluster).astype(int)
print("\nK-Means malicious cluster =", mal_cluster, "ratio=", ratios[mal_cluster])


# ----- 6. DBSCAN -----
# DBSCAN is slow on 130k points so we run it on a sample of 30k
np.random.seed(42)
sub = np.random.choice(len(X), 30000, replace=False)
X_sub = X[sub]
y_sub = y[sub]

db = DBSCAN(eps=0.5, min_samples=10)
db_labels = db.fit_predict(X_sub)
print("\nDBSCAN clusters found:", len(set(db_labels)) - (1 if -1 in db_labels else 0))
print("Noise points:", (db_labels == -1).sum())

# noise points are flagged as botnet
db_pred = (db_labels == -1).astype(int)


# ----- 7. Isolation Forest -----
iso = IsolationForest(contamination=0.01, random_state=42)
iso_raw = iso.fit_predict(X)
# in sklearn -1 means anomaly, 1 means normal
iso_pred = (iso_raw == -1).astype(int)


# ----- 8. Evaluation -----
print("\n========= RESULTS =========")

def show_results(name, y_true, y_pred):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)
    print(f"\n{name}")
    print(f"Accuracy  = {acc:.4f}")
    print(f"Precision = {prec:.4f}")
    print(f"Recall    = {rec:.4f}")
    print(f"F1-score  = {f1:.4f}")
    print("Confusion matrix:")
    print(cm)
    return [name, acc, prec, rec, f1]

results = []
results.append(show_results("K-Means", y, km_pred))
results.append(show_results("DBSCAN (sample)", y_sub, db_pred))
results.append(show_results("Isolation Forest", y, iso_pred))

# save table
res_df = pd.DataFrame(results, columns=["Model","Accuracy","Precision","Recall","F1"])
res_df.to_csv("results.csv", index=False)
print("\n", res_df)


# ----- 9. Visualize with PCA -----
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

# zoom around the dense area (most points are there)
x_low, x_high = np.percentile(X_pca[:,0], [1, 99])
y_low, y_high = np.percentile(X_pca[:,1], [1, 99])

plt.figure(figsize=(7,5))
plt.scatter(X_pca[y==0,0], X_pca[y==0,1], s=3, c="lightgray", label="Normal")
plt.scatter(X_pca[y==1,0], X_pca[y==1,1], s=8, c="red", label="Botnet")
plt.xlim(x_low, x_high)
plt.ylim(y_low, y_high)
plt.legend()
plt.title("PCA - true labels")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("fig_pca_true.png")
plt.close()

plt.figure(figsize=(7,5))
plt.scatter(X_pca[:,0], X_pca[:,1], s=2, c=km_labels, cmap="tab10")
plt.title(f"PCA - K-Means clusters (k={best_k})")
plt.tight_layout()
plt.savefig("fig_pca_kmeans.png")
plt.close()

# bar chart for comparison
plt.figure(figsize=(8,4))
x = np.arange(len(res_df))
w = 0.2
plt.bar(x-w, res_df["Precision"], width=w, label="Precision")
plt.bar(x,   res_df["Recall"],    width=w, label="Recall")
plt.bar(x+w, res_df["F1"],        width=w, label="F1")
plt.xticks(x, res_df["Model"])
plt.legend()
plt.title("Models comparison")
plt.tight_layout()
plt.savefig("fig_comparison.png")
plt.close()

print("\nDone. All figures and results saved.")
