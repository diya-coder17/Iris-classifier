# 🌸 Iris Classifier – EDA, KNN & Decision Trees

This project explores the **Iris dataset** through **exploratory data analysis (Task 1)** and applies two machine learning techniques:

* **K-Nearest Neighbors (KNN)** (Task 2)
* **Decision Trees with pruning & hyperparameter tuning** (Intermediate Task 2)

The goal is to classify iris flowers into their species (**Setosa, Versicolor, Virginica**) and analyze model performance.

---

## 🔹 Tasks Completed

### ✅ Task 1: Exploratory Data Analysis (EDA)

* Loaded and cleaned the dataset
* Checked for missing & duplicate values
* Visualized feature distributions and pairplots
* Explored correlations (Petal dimensions show stronger correlation with species)

### ✅ Task 2: KNN Classifier

* Preprocessed data with train/test split + standard scaling
* Trained KNN models with different `k` values (3, 5, 7, 9)
* Evaluated performance using Accuracy, Confusion Matrix, and Classification Report
* Achieved **100% accuracy** across all tested `k` values

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

## 📊 Results & Observations

* **EDA:** Petal length & petal width are the strongest predictors of species.
* **KNN:** Scaling features is essential. The model classified perfectly with different values of `k`.
* **Decision Tree:**

  * Initial model showed slight overfitting.
  * After pruning & tuning, the model generalized well, achieving perfect classification on test data.
  * Decision boundaries are interpretable, making trees a great choice for explainability.

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
jupyter notebook iris_decision_tree.ipynb
```

---

## 📌 Learnings

* Scaling improves performance for distance-based algorithms like **KNN**.
* **Choice of k** directly impacts performance and generalization.
* **Decision Trees** are highly interpretable but can overfit → pruning/hyperparameter tuning is necessary.
* Visualization (pairplots, correlation heatmaps, decision tree graphs) is crucial to understanding feature importance and separability.

---

✨ Author: **Diya Agarwal**
