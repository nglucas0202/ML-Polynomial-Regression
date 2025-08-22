<<<<<<< HEAD
# ML-Portfolio
Showcasing Machine Learning projects - Polynomial Regression
=======

## Overview
This repository contains the implementation of **regularized polynomial regression**. Polynomial regression extends linear regression by fitting a polynomial function to the input variable:

$h_\theta(x)$ = $\theta_0$ + $\theta_1$ x + $\theta_2$ $x^2$ + $\dots$ + $\theta_d$ $x^d$

Using a **basis expansion**, this can be written as a linear model:

$h_\theta(x) = \theta_0 + \theta_1 \phi_1(x) + \dots + \theta_d \phi_d(x), \quad \phi_j(x) = x^j$

where $d$ is the degree of the polynomial.

This project demonstrates:
- Polynomial feature expansion of univariate data.
- Standardization of polynomial features for numerical stability.
- Implementation of **regularized polynomial regression**.
- Examine the bias-variance tradeoff through learning curves, by using different values for $d$ and $\lambda$

## Part 1 - Regularization
### Data
A small dataset with $n=11$ points:
- **Input variable ($X$)**: A single real-valued feature ranging from $0$ to $10$
- **Output Variable ($y$)**: A real-valued response vary nonlinearly with $X$

If we plot $(X, y)$, it suggests a nonlinear relationship with bumps and curvature.

### Features
- $PolynomialRegression(degree, regLambda)$: Constructor to create a polynomial regression model with degree $d$ and regularization parameter $\lambda$.
- $PolynomialRegression:polyfeatures(X, degree)$: Expands univariate input $X$ into its polynomial features up to the specified degree $d$. Does **not** include the zero-th power feature ($x^0$) to allow generalization to multivariate data.
- $PolynomialRegression:fit(X, Y)$: Trains the model using the closed-form solution. Polynomial features are standardized during training.
- $PolynomialRegression:predict(X)$: Applies the trained model to new input data, with the same standardization applied.

### Plotting Prediction
By fixing the degree of the polynomial $d=8$, the following two plots show the prediction curve with regularization ($\lambda=0.02$) and without regularization ($\lambda=0$). From the plot without regularization, we see that
the function fits the data well, but will not generalize well to new data points. By increasing $\lambda$ to 0.02, the curve smoothed out quite a lot while still following the treand of the data, showing the effect of shrinking weights and reducing in variance.

## Part 2 - Bias vs Variance
We also analyze the **bias-variance tradeoff** using **learning curves**, which show how the training error and testing error evolve as the number of training examples increases.

### Data
We use the same dataset as Part 1 but this time, we use **Leave-One-Out Cross Validation** by repeatedly train the model on $n-1$ points and test it on the **1 point left out**, each time leaving out a different data point as the test set.

### Feature
- $PolynomialRegression:learningCurve(Xtrain, Ytrain, Xtest, Ytest, reg_lambda, degree)$: Progressively iterate through the training set ($n$ times), train/predict based on the polynomial regression model with the given degree $d$ and regularization $\lambda$, and predict the value for the lone test data point. It also stores the training and testing errors for each iteration for plotting.

### Learning Curves
We plot the learning curves for various values of $\lambda$ and $d$ as shown below (y-axis is using a log-scale) and notice the following:
- In General, with very small training size, training error is low (model memorizes points) but test error is high (overfitting). As training size increases, training error increases (harder to perfectly fit more points), but test error decreases (better generalization). Eventually, both curves flatten out → that’s the bias–variance balance point.
- The plot of the unregularized model with $d=1$ shows poor training error, indicating a high bias (i.e., it is a standard univariate linear regression fit).
- The plot of the (almost) unregularized model ($\lambda = 10^{−6}$) with $d=8$ shows that the training error is low, but that the testing error is high. There is a huge gap between the training and testing errors caused by the model overfitting the training data, indicating a high variance problem.
- As the regularization parameter increases (e.g., $\lambda = 1$) with $d=8$, we see that the gap between the training and testing error narrows, with both the training and testing errors converging to a low value. We can see that the model fits the data well and generalizes well, and therefore does not have either a high bias or a high variance problem. Effectively, it has a good tradeoff between bias and variance.
- Once the regularization parameter is too high ($\lambda = 100$), we see that the training and testing errors are once again high, indicating a poor fit. Effectively, there is too much regularization, resulting in high bias.

## Installation / Setup
### 1. Conda Environment Setup
This project requires **Python 3.9+** and was developed in a Conda environment.  
I would recommend using [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage dependencies.

### 2. Clone this repository
```bash
git clone https://github.com/nglucas0202/ML-Polynomial-Regression.git
```
### 3. Clone this repositoryCreate new Conda environment (one time only)
```Anaconda Prompt
cd ML-Polynomial-Regression
conda env create -f environment.yaml
```
*Note*: The command may take long time, especially if your connection is slow.

### 4. Aconccctive the environment
```Anaconda Prompt
conda activate lucas-ml-projects
```
*Note*: All my ML repositories use the same Conda environment

### 5. Select VSCode Python Interpreter
In VSCode, click into a python file and in bottom left you should see *Python* with some version.
Click into that and choose one that says "lucas-ng-projects".

### 6. Part 1 Regularization
To run Part 1, open code/poly_regression/plot_polyreg_univariate.py and click *Run* button.

### 7. Part 2 Bias vs Variance
To run Part 2, open code/poly_regression/plot_polyreg_learningCurve.py and click *Run* button.