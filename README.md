# Artificial Intelligence Lab – 5

This repository contains implementations of three fundamental **Machine Learning algorithms** using Python and Scikit-learn:

1. **Linear Regression**
2. **Logistic Regression**
3. **Decision Tree Classification**

The programs use datasets provided by Scikit-learn and demonstrate model training, prediction, evaluation, and visualization.

---

## 📁 Project Structure

```text
Artificial_Intelligence_Lab_5/
│
├── Linear Regression.py
├── Logistic Regression.py
├── Decision Tree.py
└── README.md
```

---

## 🛠️ Technologies Used

* **Python 3**
* **Pandas** – Data manipulation and analysis
* **Matplotlib** – Data visualization
* **Scikit-learn** – Machine Learning algorithms and datasets

---

## 📌 1. Linear Regression

### Dataset

The program uses the **Diabetes dataset** available in Scikit-learn.

```python
from sklearn.datasets import load_diabetes
```

### Algorithm

**Linear Regression** is a supervised learning algorithm used to predict a continuous numerical value based on one or more input features.

### Implementation Steps

1. Load the Diabetes dataset.
2. Convert the dataset into Pandas DataFrames/Series.
3. Split the data into training and testing sets.
4. Create a `LinearRegression` model.
5. Train the model using the training data.
6. Predict values for the test dataset.
7. Evaluate the model using:

   * Mean Absolute Error (MAE)
   * Mean Squared Error (MSE)
   * R² Score
8. Plot actual values against predicted values.

### Evaluation Metrics

**Mean Absolute Error (MAE)**

Measures the average absolute difference between actual and predicted values.

**Mean Squared Error (MSE)**

Measures the average squared difference between actual and predicted values.

**R² Score**

Measures how well the model explains the variation in the target variable.

---

## 📌 2. Logistic Regression

### Dataset

The program uses the **Breast Cancer Wisconsin dataset** provided by Scikit-learn.

```python
from sklearn.datasets import load_breast_cancer
```

### Algorithm

**Logistic Regression** is a supervised classification algorithm commonly used for binary classification problems.

In this program, the model predicts whether a breast cancer sample belongs to one of the two classes.

### Implementation Steps

1. Load the Breast Cancer dataset.
2. Convert the features and target into Pandas structures.
3. Split the dataset into training and testing sets.
4. Create a `LogisticRegression` model.
5. Train the model.
6. Predict the test data.
7. Calculate the classification accuracy.
8. Generate a classification report.
9. Generate and display a confusion matrix.

### Evaluation Metrics

The program uses:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**
* **Confusion Matrix**

The confusion matrix visually represents correct and incorrect predictions for each class.

---

## 📌 3. Decision Tree Classification

### Dataset

The program uses the famous **Iris dataset** provided by Scikit-learn.

```python
from sklearn.datasets import load_iris
```

### Algorithm

A **Decision Tree** is a supervised learning algorithm that can be used for both classification and regression.

For this experiment, a `DecisionTreeClassifier` is used to classify Iris flowers into their respective species.

### Classes

The Iris dataset contains three classes:

* Iris Setosa
* Iris Versicolor
* Iris Virginica

### Implementation Steps

1. Load the Iris dataset.
2. Convert the features and target into Pandas structures.
3. Split the dataset into training and testing sets.
4. Create a Decision Tree Classifier.
5. Use the **Gini Index** as the splitting criterion.
6. Limit the maximum tree depth to 3.
7. Train the model.
8. Predict the test data.
9. Calculate the accuracy.
10. Generate a classification report.
11. Visualize the trained Decision Tree.

### Evaluation Metrics

The model is evaluated using:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**
* **Classification Report**

The final tree is also visualized using Matplotlib.

---

## ⚙️ Installation

Make sure Python 3 is installed on your system.

Install the required Python libraries using:

```bash
pip install pandas matplotlib scikit-learn
```

---

## ▶️ How to Run

Clone or download this repository and navigate to the project directory.

### Run Linear Regression

```bash
python "Linear Regression.py"
```

### Run Logistic Regression

```bash
python "Logistic Regression.py"
```

### Run Decision Tree

```bash
python "Decision Tree.py"
```

---

## 📊 Expected Output

### Linear Regression

The program displays:

* First few rows of the dataset
* MAE
* MSE
* R² Score

It also displays an **Actual vs Predicted Values** scatter plot.

### Logistic Regression

The program displays:

* Dataset dimensions
* Target class names
* Accuracy
* Classification report

It also displays a **Confusion Matrix**.

### Decision Tree

The program displays:

* First few rows of the dataset
* Target class names
* Accuracy
* Classification report

It also displays a graphical representation of the **Decision Tree**.

---

## 🧠 Algorithms at a Glance

| Algorithm           | Type           | Dataset       | Main Purpose               |
| ------------------- | -------------- | ------------- | -------------------------- |
| Linear Regression   | Regression     | Diabetes      | Predict continuous values  |
| Logistic Regression | Classification | Breast Cancer | Binary classification      |
| Decision Tree       | Classification | Iris          | Multi-class classification |

---

## 📚 Learning Objectives

Through these programs, the following concepts are demonstrated:

* Loading built-in Machine Learning datasets
* Data preparation using Pandas
* Splitting datasets into training and testing sets
* Supervised Machine Learning
* Model training
* Making predictions
* Model evaluation
* Classification metrics
* Regression metrics
* Confusion matrix
* Data visualization
* Decision Tree visualization

---

## 🔑 Key Concepts

### Training Data

The portion of the dataset used by the Machine Learning model to learn patterns.

### Testing Data

The unseen portion of the dataset used to evaluate how well the trained model performs.

### Supervised Learning

A Machine Learning approach where the model learns from input data along with known target/output values.

### Classification

Predicting a category or class, such as:

```text
Setosa / Versicolor / Virginica
```

### Regression

Predicting a continuous numerical value, such as a disease progression measurement.

---

## 👨‍💻 Author

**Soutik**

Artificial Intelligence / Machine Learning Lab Assignment

---

## 📄 License

This project is created for **educational and academic purposes**.
