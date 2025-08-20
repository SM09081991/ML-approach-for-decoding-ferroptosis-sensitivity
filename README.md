# ML-approach-for-decoding-ferroptosis-sensitivity
This is a proof-of-concept pipeline designed to predict whether a gene or set of genes can modulate ferroptosis. It leverages GSVA and machine learning to assess cell state vulnerabilities and can be easily adapted for other datasets or gene sets.

📖 **Background**
This analysis is inspired by the study “Secreted APOE rewires melanoma cell state vulnerability to ferroptosis” published in Science Advances. In that work, a ferroptosis score (FPS) was computed using curated gene sets classified as drivers or suppressors of ferroptosis from the FerrDb database. Only genes with a protein-coding product and a validated confidence level were selected. For each sample, GSVA enrichment scores were calculated separately for the suppressor and driver gene sets. The ferroptosis score was defined as the difference between these two scores: FPS = GSVA(driver genes) − GSVA(suppressor genes). In this script, samples with FPS above the median were labeled ferroptosis-sensitive, while those below were considered ferroptosis-resistant. Here, machine learning classifiers were applied to identify individual genes or gene combinations that can predict ferroptosis sensitivity across patient samples or cell lines.

✅ **Understanding the ML approach**
**Why Use Logistic Regression for One Gene?**
Logistic regression is:

A simple, interpretable model that assumes a linear relationship between input (gene expression) and outcome (ferroptosis sensitivity).

Well-suited for analyzing a single gene, because it shows the direction and strength of association (positive or negative).

Ideal for evaluating:
“Does high expression of "Gene1" increase or decrease ferroptosis sensitivity?”

**Why Use Random Forest for Two or More Genes?**
Random forest:

Is a non-linear, ensemble model that builds many decision trees and averages their results.

Handles interactions and non-additive effects between genes (e.g., Gene1 impact might depend on Gene2 level).

Automatically learns complex decision boundaries — perfect when you're combining genes.


---

## 🧪 Analysis Pipeline

### **Step 1: Load Expression Matrix**
The input expression matrix is transposed so that each row represents a sample and columns represent gene expression levels.

### **Step 2: Load Ferroptosis Gene Sets**
Gene lists for ferroptosis drivers and suppressors are loaded from `.txt` files.

### **Step 3: GSVA Scoring**
We use [GSEApy's `ssGSEA`](https://gseapy.readthedocs.io/en/latest/) method to compute enrichment scores for driver and suppressor genes in each sample.

### **Step 4: Compute Ferroptosis Score**
Ferroptosis Score = `GSVA(driver genes)` − `GSVA(suppressor genes)`  
This gives a relative measure of ferroptotic activity per sample.

### **Step 5: Binarize Samples**
Samples are split into:
- **Ferroptosis-sensitive** (score > median)
- **Ferroptosis-resistant** (score ≤ median)

### **Step 6: Logistic Regression**
We use only one gene (e.g., `Gene1`) to predict binary ferroptosis sensitivity using a logistic regression classifier.

### **Step 7: SHAP for Logistic Regression**
We apply SHAP (`shap.Explainer`) to interpret how APOE expression influences the logistic regression model's predictions.

### **Step 8: Correlation Analysis**
A scatter plot is generated showing Spearman correlation between APOE expression and ferroptosis score, visualizing linear or rank-based relationships.

### **Step 9: Random Forest (Gene Combinations)**
We train a `RandomForestClassifier` using combinations of two genes (e.g., `Gene1` and `Gene2`) to model interactions that contribute to ferroptosis sensitivity.

### **Step 10: SHAP for Random Forest**
Using `shap.TreeExplainer`, we decompose each prediction to visualize how Gene1 and Gene2 jointly influence model outcomes.

---

## 📈 Outputs

- `gene1_vs_ferroptosis_score.png`: Scatterplot of Gene1 vs ferroptosis score.
- Console output of classification reports, confusion matrices, and feature importances.
- SHAP visualizations for both linear and nonlinear models.

---




