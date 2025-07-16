# === DEPENDENCIES ===
import pandas as pd
import numpy as np
from gseapy import gsva
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import shap

# === STEP 1: Load Expression Data ===
expr = pd.read_csv("data/processed/expression_matrix.csv", index_col=0).T  # samples x genes

# === STEP 2: Load Gene Sets ===
with open("data/ferroptosis_driver_genes.txt") as f:
    driver_genes = [line.strip() for line in f]
with open("data/ferroptosis_suppressor_genes.txt") as f:
    suppressor_genes = [line.strip() for line in f]

gene_sets = {
    "ferroptosis_driver": driver_genes,
    "ferroptosis_suppressor": suppressor_genes
}

# === STEP 3: Calculate GSVA Scores ===
gsva_result = gsva(expr, gene_sets=gene_sets, method='ssgsea', verbose=False).T

# === STEP 4: Calculate Ferroptosis Score ===
gsva_result["ferroptosis_score"] = gsva_result["ferroptosis_driver"] - gsva_result["ferroptosis_suppressor"]

# === STEP 5: Create Binary Labels ===
threshold = gsva_result["ferroptosis_score"].median()
labels = (gsva_result["ferroptosis_score"] > threshold).astype(int)

# === STEP 6: Logistic Regression on APOE ===
# Logistic regression is used here because we are evaluating the effect of a single gene.
# It offers a simple, interpretable model where the direction and strength of association
# between gene expression (e.g., APOE) and ferroptosis sensitivity can be easily understood.
X = expr[["Gene"]]
y = labels
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

clf_lr = LogisticRegression(max_iter=1000, random_state=42)
clf_lr.fit(X_train, y_train)
y_pred = clf_lr.predict(X_test)

print("Logistic Regression - Classification Report:\n")
print(classification_report(y_test, y_pred))

# === Confusion Matrix Plot ===
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Resistant", "Sensitive"],
            yticklabels=["Resistant", "Sensitive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression Confusion Matrix")
plt.tight_layout()
plt.show()

# === SHAP for Logistic Regression ===
explainer = shap.Explainer(clf_lr, X_train)
shap_values = explainer(X_test)
shap.plots.beeswarm(shap_values)

# === Correlation Analysis (APOE vs FPS) ===
ferroptosis_score = gsva_result["ferroptosis_score"]
apoe_expr = expr["Gene"]
rho, pval = spearmanr(apoe_expr, ferroptosis_score)

plt.figure(figsize=(7, 5))
sns.regplot(x=apoe_expr, y=ferroptosis_score, scatter_kws={"alpha": 0.6})
plt.xlabel("Gene Expression")
plt.ylabel("Ferroptosis Score")
plt.title(f"Gene vs Ferroptosis Score\nSpearman ρ = {rho:.2f}, p = {pval:.2g}")
plt.grid(True)
plt.tight_layout()
plt.savefig("apoe_vs_ferroptosis_score.png", dpi=300)
plt.show()

# === Correlation if expression and FPS are separate ===
# FPS should be a DataFrame with index matching expr and a column 'FPS'
# common_index = expr.index.intersection(FPS.index)
# x = expr.loc[common_index, "APOE"]
# y = FPS.loc[common_index, "FPS"]
# rho, p = spearmanr(x, y)
# print(f"Spearman ρ = {rho:.3f}, p = {p:.4g}")

# === Random Forest on Gene1 + Gene2 ===
# Random forest is used here to analyze multiple genes simultaneously.
# Unlike logistic regression, it can capture non-linear relationships and interactions between genes,
# such as APOE and TFRC, to improve classification of ferroptosis sensitivity.
selected_genes = ["Gene1", "Gene2"]
if not all(g in expr.columns for g in selected_genes):
    raise ValueError("One or both of Gene1 and Gene2 are not in the expression matrix.")

X_subset = expr[selected_genes]
X_scaled = scaler.fit_transform(X_subset)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, labels.values.ravel(), test_size=0.3, random_state=42)

clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf.fit(X_train, y_train)
y_pred = clf_rf.predict(X_test)

print("\nRandom Forest (Gene1 + Gene2) - Classification Report:\n")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Resistant", "Sensitive"],
            yticklabels=["Resistant", "Sensitive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest (Gene1 + Gene2)")
plt.tight_layout()
plt.show()

# === Feature Importance ===
importances = clf_rf.feature_importances_
for gene, importance in zip(selected_genes, importances):
    print(f"Importance of {gene}: {importance:.4f}")

# === SHAP for Random Forest ===
rf_explainer = shap.TreeExplainer(clf_rf)
shap_rf_values = rf_explainer.shap_values(X_test)

# Summary plot (class 1: sensitive)
shap.summary_plot(shap_rf_values[1], X_test, feature_names=selected_genes)
