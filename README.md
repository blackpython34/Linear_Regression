# Linear Regression Learning System 📈

An interactive, educational web application designed to help students visually and mathematically understand the foundations of **Simple and Multiple Linear Regression**. Built entirely in Python using Streamlit, this tool bridges the gap between raw code and theoretical mathematics by providing a hands-on, step-by-step learning environment.

---

## 🎯 Educational Objectives

This system was developed with a strict focus on educational outcomes. It allows users to track the entire machine learning lifecycle without treating algorithms like a "black box."

### 1. Dataset Upload & Preview
Students can upload custom datasets (`.csv`) to experiment with data they care about. The system dynamically parses columns, differentiating between categorical and numerical features.

### 2. Preprocessing & Data Cleaning
A guided interface teaches students the importance of data hygiene. 
- **Missing Value Handling:** Choose between mean imputation or dropping NaN rows.
- **Categorical Encoding:** Automatically encodes textual categories to numerical labels.
- **Feature Scaling:** Demonstrates Standardization (Z-Score) and Normalization (Min-Max) to explain why scaling is necessary for gradient descent convergence.

### 3. Exploratory Data Analysis (EDA)
Before training, students must analyze relationships in their data to ensure linear regression is an appropriate algorithm:
- **Feature Distributions:** Interactive Histograms and Box Plots.
- **Correlation Analysis:** A full Correlation Heatmap to identify collinearity and strong predictors.
- **Relationship Visualization:** Feature vs. Target scatter plots with OLS trendlines.

### 4. Interactive Learning Module (Theoretical Foundations)
A dedicated module that walks students through the math using proper $\LaTeX$ notation:
- Differentiates the formulas for **Simple** vs. **Multiple** Linear Regression.
- Defines **Hypothesis Formulation** ($h_\theta(x)$) and the **Cost Function** ($J(\theta)$).
- **Step-by-Step Computation:** Explicitly breaks down the math for the *first sample* of the dataset, showing the exact formula used, intermediate weight values, prediction calculation, and error calculation.
- Defines the **Gradient Descent** update rule mathematically.

### 5. Training Configuration (Gradient Descent)
Students can visually see the training process rather than just calling `.fit()`:
- Manually configure Hyperparameters: **Learning Rate ($\alpha$)** and **Epochs**.
- Split data into Training and Testing sets.
- A live **Cost Convergence Graph** plots the Cost Function $J(\theta)$ over time, allowing students to visually understand convergence, underfitting, and over-shooting (exploding gradients).

### 6. Prediction & Evaluation
Evaluates the custom-trained model:
- Displays learned parameters (Weights and Bias).
- Reports **Mean Squared Error (MSE)**, **Mean Absolute Error (MAE)**, and **R² Score**.
- Allows users to input custom values into their trained model for live predictions.

---

## 🛠 Technical Architecture

- **Frontend / Framework:** [Streamlit](https://streamlit.io/) with heavy custom CSS injection to create a modern, dark-themed, React-like UI.
- **Data Manipulation:** `pandas` and `numpy`.
- **Machine Learning:** Custom Gradient Descent implementation built from scratch using `numpy` (No Scikit-Learn `.fit()` shortcuts were used for training to ensure educational transparency).
- **Visualizations:** `plotly.express` for interactive charts.

---

## 🚀 Installation & Execution

### Prerequisites
- Python 3.8+ installed on your system.

### Setup Instructions

1. **Clone or Extract the Repository:**
   Navigate to the project directory:
   ```bash
   cd Linear_Regression2
   ```

2. **Create a Virtual Environment:**
   It is recommended to use a virtual environment to manage dependencies.
   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment:**
   - On Windows:
     ```powershell
     .\venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install Dependencies:**
   Install the required packages from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Application:**
   Launch the Streamlit server:
   ```bash
   streamlit run app.py
   ```

6. **View the App:**
   Your default web browser should automatically open the app. If not, navigate to `http://localhost:8501`.

---

*Developed for interactive learning and academic demonstration.*
