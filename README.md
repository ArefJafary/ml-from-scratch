# ml-from-scratch
🧱 Implementing classical ML algorithms from scratch , no frameworks just NumPy

This repository contains five foundational machine learning algorithms implemented entirely from scratch using only NumPy. The goal of this project is to deeply understand how classic ML algorithms work internally — without relying on external ML libraries such as scikit-learn.


---

## 🛠️ Implemented Algorithms

All models are implemented as Python classes with .fit() and .predict() methods, following the familiar scikit-learn API design:
```

| Task           | Algorithm              | File                        |
|----------------|------------------------|-----------------------------|
| Classification | Support Vector Machine | models/svm.py              |
| Classification | Decision Tree          | models/decision_tree.py    |
| Regression     | Linear Regression      | models/linear_regression.py|
| Regression     | K-Nearest Neighbors    | models/knn.py              |
| Clustering     | K-Means                | models/k_means.py          |

```

---

## 📁 Repository Structure
```
ML-from-scratch/
├── models/           # Python classes implementing ML algorithms
│   ├── svm.py
│   ├── decision_tree.py
│   ├── knn.py
│   ├── linear_regression.py
│   └── k_means.py
│   └──__init__.py
│
├── notebooks/        # Jupyter notebooks demonstrating each model
│   ├── SVM_Demo.ipynb
│   ├── DecisionTree_Demo.ipynb
│   ├── KNN_Demo.ipynb
│   ├── Linear_Regression_Demo.ipynb
│   └── K_Means_Demo.ipynb
│
└── README.md
``` 


---



## 📓 Notebooks

Each model comes with a corresponding demo notebook using small datasets from scikit-learn (e.g., Iris, Wine, etc.) to:

Load and prepare data

Train the model

Evaluate performance using standard metrics (e.g., accuracy or R² score)


---

## 🔮 Future Work

Add data visualization to notebooks

Extend the repository with more algorithms (e.g., Naive Bayes, Logistic Regression)

---

## ⚙️ Requirements

Python 3.7+

NumPy

scikit-learn (for datasets and evaluation only)

Jupyter (optional, for notebooks)

