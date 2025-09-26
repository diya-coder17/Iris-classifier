# 🌸 Iris Classifier – EDA, KNN, Decision Trees & K-Means Clustering

This project explores the **Iris dataset** through **exploratory data analysis (Task 1)** and applies multiple machine learning techniques:

* **K-Nearest Neighbors (KNN)** (Task 2)
* **Decision Trees with pruning & hyperparameter tuning** (Intermediate Task 2)
* **K-Means Clustering (Unsupervised learning)** (Task 3)

The goal is to classify iris flowers into their species (**Setosa, Versicolor, Virginica**) and analyze clustering patterns without labels.

---

## 🔹 Tasks Completed

### ✅ Task 1: Exploratory Data Analysis (EDA)

* Loaded and cleaned the dataset
* Checked for missing & duplicate values
* Visualized feature distributions and pairplots
* Explored correlations (Petal dimensions show stronger correlation with species)

---

### ✅ Task 2: KNN Classifier

* Preprocessed data with train/test split + standard scaling
* Trained KNN models with different `k` values (3, 5, 7, 9)
* Evaluated performance using Accuracy, Confusion Matrix, and Classification Report
* Achieved **100% accuracy** across all tested `k` values

---

### ✅ Task 3: Decision Tree Classifier

* Built a baseline **Decision Tree** classifier
* Visualized the decision tree using `pydotplus`
* Evaluated accuracy, precision, recall, and F1-score
* Performed **hyperparameter tuning (GridSearchCV)** to prune the tree and reduce overfitting
* Final model achieved:

  * **Train Accuracy:** 99.1%
  * **Test Accuracy:** 100%
* Balanced generalization → no misclassifications on test set 🎉

---

### ✅ Task 4: K-Means Clustering (Unsupervised Learning)

* Standardized all 4 features (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`)
* Applied **K-Means clustering** with `k=1…10`
* Used the **Elbow Method** to determine optimal clusters (**k=3**)
* Visualized clusters in 2D scatter plots and **pairplots** with centroids

#### 🔎 Interpretation of Results

* **Optimal clusters:** 3, which aligns with the actual species classes.
* **Cluster separation:** Petal length & width provide the strongest separation.
* **Overlap:** Versicolor & Virginica overlap slightly (sepal-based features are less discriminative).
* **Centroids:** Represent the “typical” flower in each cluster.
* **Insight:** Even without labels, K-Means discovered meaningful groupings close to the true species distribution.

---

## 📊 Results & Observations

* **EDA:** Petal dimensions are the most informative features.
* **KNN:** Scaling is essential; perfect classification was achieved.
* **Decision Tree:** Needed pruning for generalization; interpretable boundaries.
* **K-Means:** Unsupervised learning confirmed **natural groupings** exist in the dataset (Setosa clearly separated, overlap between Versicolor & Virginica).

---

## 🛠️ Tech Stack

* **Language:** Python 🐍
* **Libraries:**

  * Data Handling → pandas, numpy
  * Visualization → matplotlib, seaborn
  * ML Models → scikit-learn
  * Tree Visualization → pydotplus, graphviz

---

## 🚀 Run the Project

Clone the repository:

```bash
git clone https://github.com/yourusername/Iris-Classifier.git
cd Iris-Classifier
pip install -r requirements.txt
jupyter notebook iris_classifier.ipynb
```

---

## 📌 Learnings

* **Scaling** improves distance-based algorithms like KNN & clustering.
* **Choice of k** (for KNN and K-Means) directly impacts performance and grouping.
* **Decision Trees** → interpretable but need pruning for generalization.
* **Unsupervised Learning** (K-Means) can reveal natural groupings even without labels.
* **Visualization** (pairplots, heatmaps, cluster plots) is crucial to understanding feature importance & separability.

---

✨ Author: **Diya Agarwal**
