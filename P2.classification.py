import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# =============================================================================
# 1. Binary Logistic Regression Implementation
# =============================================================================

class BinaryLogisticRegression:
    def __init__(self, lr=0.1, num_iter=500, reg_type=None, reg_lambda=0.0, verbose=False):
        """
        reg_type: None, 'l1', or 'l2'
        reg_lambda: regularization strength
        """
        self.lr = lr
        self.num_iter = num_iter
        self.reg_type = reg_type
        self.reg_lambda = reg_lambda
        self.verbose = verbose

    def sigmoid(self, z):
        return 1.0 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        m, n = X.shape
        # Initialize weights and bias
        self.w = np.zeros(n)
        self.b = 0.0

        for i in range(self.num_iter):
            # Compute model predictions
            linear_model = np.dot(X, self.w) + self.b
            y_pred = self.sigmoid(linear_model)
            # Compute error
            error = y_pred - y
            # Gradient without regularization
            dw = (1/m) * np.dot(X.T, error)
            db = (1/m) * np.sum(error)
            # Add regularization gradients if specified
            if self.reg_type == 'l2':
                dw += (self.reg_lambda/m) * self.w
            elif self.reg_type == 'l1':
                dw += (self.reg_lambda/m) * np.sign(self.w)
            # Gradient descent update
            self.w -= self.lr * dw
            self.b -= self.lr * db
            # Optionally print progress
            if self.verbose and i % 100 == 0:
                loss = self._loss(y, y_pred, m)
                print(f"[Binary] Iteration {i}, Loss: {loss:.4f}")

    def _loss(self, y, y_pred, m):
        # Negative log-likelihood (binary cross entropy) plus regularization penalty
        loss = - (1/m) * np.sum( y * np.log(y_pred + 1e-15) + (1-y) * np.log(1-y_pred + 1e-15) )
        if self.reg_type == 'l2':
            loss += (self.reg_lambda/(2*m)) * np.sum(self.w**2)
        elif self.reg_type == 'l1':
            loss += (self.reg_lambda/m) * np.sum(np.abs(self.w))
        return loss

    def predict_prob(self, X):
        return self.sigmoid(np.dot(X, self.w) + self.b)
    
    def predict(self, X):
        return (self.predict_prob(X) >= 0.5).astype(int)

# =============================================================================
# 2. Multiclass Logistic Regression Implementation (Softmax)
# =============================================================================

class MulticlassLogisticRegression:
    def __init__(self, lr=0.1, num_iter=500, reg_type=None, reg_lambda=0.0, verbose=False):
        """
        reg_type: None, 'l1', or 'l2'
        reg_lambda: regularization strength
        """
        self.lr = lr
        self.num_iter = num_iter
        self.reg_type = reg_type
        self.reg_lambda = reg_lambda
        self.verbose = verbose

    def softmax(self, z):
        # subtract max for numerical stability
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        """
        X: shape (m, n_features)
        y: labels (0, 1, ..., K-1)
        """
        m, n = X.shape
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        # Initialize weight matrix and bias
        self.W = np.zeros((n, K))
        self.b = np.zeros(K)
        # One-hot encode y
        Y_onehot = np.zeros((m, K))
        for i, label in enumerate(y):
            Y_onehot[i, label] = 1

        for i in range(self.num_iter):
            logits = np.dot(X, self.W) + self.b   # (m x K)
            probs = self.softmax(logits)            # (m x K)
            # Compute gradient
            error = probs - Y_onehot                # (m x K)
            dW = (1/m) * np.dot(X.T, error)           # (n x K)
            db = (1/m) * np.sum(error, axis=0)        # (K,)

            # Add regularization gradients if needed
            if self.reg_type == 'l2':
                dW += (self.reg_lambda/m) * self.W
            elif self.reg_type == 'l1':
                dW += (self.reg_lambda/m) * np.sign(self.W)

            self.W -= self.lr * dW
            self.b -= self.lr * db

            if self.verbose and i % 100 == 0:
                loss = self._loss(Y_onehot, probs, m)
                print(f"[Multiclass] Iteration {i}, Loss: {loss:.4f}")

    def _loss(self, Y_onehot, probs, m):
        loss = - (1/m) * np.sum(Y_onehot * np.log(probs + 1e-15))
        if self.reg_type == 'l2':
            loss += (self.reg_lambda/(2*m)) * np.sum(self.W**2)
        elif self.reg_type == 'l1':
            loss += (self.reg_lambda/m) * np.sum(np.abs(self.W))
        return loss

    def predict_prob(self, X):
        logits = np.dot(X, self.W) + self.b
        return self.softmax(logits)

    def predict(self, X):
        probs = self.predict_prob(X)
        return np.argmax(probs, axis=1)

# =============================================================================
# 3. Utility Functions for Experimentation and Evaluation
# =============================================================================

def evaluate_binary(y_true, y_pred_prob):
    """
    Evaluate binary logistic regression predictions using
    squared loss and absolute loss.
    """
    sq_loss = np.mean((y_true - y_pred_prob) ** 2)
    abs_loss = np.mean(np.abs(y_true - y_pred_prob))
    return sq_loss, abs_loss

def evaluate_multiclass(y_true, y_pred_prob, encoder):
    """
    For multiclass, we first one-hot encode y_true and then compute
    the average squared error (akin to the Brier score) and average absolute error.
    """
    y_true_onehot = encoder.transform(y_true.reshape(-1, 1))
    sq_loss = np.mean((y_true_onehot - y_pred_prob) ** 2)
    abs_loss = np.mean(np.abs(y_true_onehot - y_pred_prob))
    return sq_loss, abs_loss

def run_binary_experiments(X, y, train_sizes, reg_options):
    """
    For each training set size and regularization option (none, l1, l2),
    train a binary logistic regression model and evaluate test losses.
    Returns a dictionary with losses.
    """
    from collections import defaultdict
    results = defaultdict(lambda: {"train_size": [], "sq_loss": [], "abs_loss": []})
    
    for train_size in train_sizes:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)
        for reg in reg_options:
            if reg == 'none':
                reg_type = None
                reg_lambda = 0.0
            else:
                reg_type = reg
                # For demonstration, we fix lambda=0.1.
                # In practice, use k-fold cross-validation to select this hyperparameter.
                reg_lambda = 0.1

            model = BinaryLogisticRegression(lr=0.1, num_iter=500, reg_type=reg_type, reg_lambda=reg_lambda)
            model.fit(X_train, y_train)
            y_pred_prob = model.predict_prob(X_test)
            sq_loss, abs_loss = evaluate_binary(y_test, y_pred_prob)
            
            results[reg]["train_size"].append(train_size)
            results[reg]["sq_loss"].append(sq_loss)
            results[reg]["abs_loss"].append(abs_loss)
    
    return results

def run_multiclass_experiments(X, y, train_sizes, reg_options):
    """
    Similar to run_binary_experiments but for multiclass logistic regression.
    Returns a dictionary with losses.
    """
    from collections import defaultdict
    results = defaultdict(lambda: {"train_size": [], "sq_loss": [], "abs_loss": []})
    encoder = OneHotEncoder(sparse=False)
    encoder.fit(y.reshape(-1, 1))
    
    for train_size in train_sizes:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)
        for reg in reg_options:
            if reg == 'none':
                reg_type = None
                reg_lambda = 0.0
            else:
                reg_type = reg
                reg_lambda = 0.1

            model = MulticlassLogisticRegression(lr=0.1, num_iter=500, reg_type=reg_type, reg_lambda=reg_lambda)
            model.fit(X_train, y_train)
            y_pred_prob = model.predict_prob(X_test)
            sq_loss, abs_loss = evaluate_multiclass(y_test, y_pred_prob, encoder)
            
            results[reg]["train_size"].append(train_size)
            results[reg]["sq_loss"].append(sq_loss)
            results[reg]["abs_loss"].append(abs_loss)
    
    return results

def plot_results(results, title, metric_name):
    """
    Plot the test loss (metric_name) versus training set size for each regularization option.
    """
    plt.figure(figsize=(8, 6))
    for reg, res in results.items():
        plt.plot([int(100*ts) for ts in res["train_size"]], res[metric_name],
                 marker='o', label=f"{reg}")
    plt.xlabel("Training Set Size (%)")
    plt.ylabel(metric_name.replace("_", " ").title())
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# =============================================================================
# 4. Main Experimentation
# =============================================================================

if __name__ == "__main__":
    # -------------------------
    # 4.1 Binary Logistic Regression Experiments
    # -------------------------
    # For binary classification, assume we choose a UCI dataset (e.g., spam detection).
    # Here, for demonstration, we generate synthetic data with 5,000 samples.
    X_binary, y_binary = make_classification(n_samples=5000, n_features=20,
                                             n_informative=15, n_redundant=5,
                                             random_state=42)
    # Standardize features.
    scaler = StandardScaler()
    X_binary = scaler.fit_transform(X_binary)
    
    # Define training set proportions (20%, 40%, 60%, 80%).
    train_sizes = [0.2, 0.4, 0.6, 0.8]
    # Regularization options: no regularization, L1 and L2.
    reg_options = ['none', 'l1', 'l2']
    
    print("Running binary logistic regression experiments on 5K samples...")
    binary_results = run_binary_experiments(X_binary, y_binary, train_sizes, reg_options)
    
    # Plot squared loss and absolute loss for binary classification.
    plot_results(binary_results, "Binary Logistic Regression: Squared Loss vs. Training Size", "sq_loss")
    plot_results(binary_results, "Binary Logistic Regression: Absolute Loss vs. Training Size", "abs_loss")
    
    # -------------------------
    # 4.2 Increasing the Dataset Size for Binary Classification
    # -------------------------
    for n_samples in [10000, 20000]:
        X_bin_large, y_bin_large = make_classification(n_samples=n_samples, n_features=20,
                                                       n_informative=15, n_redundant=5,
                                                       random_state=42)
        X_bin_large = scaler.fit_transform(X_bin_large)
        print(f"\nRunning binary logistic regression experiments on {n_samples} samples...")
        results_large = run_binary_experiments(X_bin_large, y_bin_large, train_sizes, reg_options)
        plot_results(results_large, f"Binary Logistic Regression (n={n_samples}): Squared Loss vs. Training Size", "sq_loss")
        plot_results(results_large, f"Binary Logistic Regression (n={n_samples}): Absolute Loss vs. Training Size", "abs_loss")
    
    # -------------------------
    # 4.3 Multiclass Logistic Regression Experiments
    # -------------------------
    # For multiclass classification, assume we choose a UCI dataset (e.g., Iris, digit recognition, species classification).
    # For demonstration, we generate a synthetic dataset with 3 classes.
    X_multi, y_multi = make_classification(n_samples=5000, n_features=20,
                                           n_informative=15, n_redundant=5,
                                           n_classes=3, n_clusters_per_class=1,
                                           random_state=42)
    X_multi = scaler.fit_transform(X_multi)
    
    print("\nRunning multiclass logistic regression experiments on 5K samples...")
    multiclass_results = run_multiclass_experiments(X_multi, y_multi, train_sizes, reg_options)
    plot_results(multiclass_results, "Multiclass Logistic Regression: Squared Loss vs. Training Size", "sq_loss")
    plot_results(multiclass_results, "Multiclass Logistic Regression: Absolute Loss vs. Training Size", "abs_loss")
    
    # Increase dataset size for multiclass experiments.
    for n_samples in [10000, 20000]:
        X_multi_large, y_multi_large = make_classification(n_samples=n_samples, n_features=20,
                                                           n_informative=15, n_redundant=5,
                                                           n_classes=3, n_clusters_per_class=1,
                                                           random_state=42)
        X_multi_large = scaler.fit_transform(X_multi_large)
        print(f"\nRunning multiclass logistic regression experiments on {n_samples} samples...")
        results_multi_large = run_multiclass_experiments(X_multi_large, y_multi_large, train_sizes, reg_options)
        plot_results(results_multi_large, f"Multiclass Logistic Regression (n={n_samples}): Squared Loss vs. Training Size", "sq_loss")
        plot_results(results_multi_large, f"Multiclass Logistic Regression (n={n_samples}): Absolute Loss vs. Training Size", "abs_loss")
    
    # =============================================================================
    # 5. Additional Questions
    # =============================================================================
    print("\nAdditional Questions Answers:")
    print("\nQ1: How do you select the appropriate value of k in k-fold cross-validation?")
    print("A1: The choice of k is a trade-off between bias and variance. A larger k (e.g., 10 or even leave-one-out)")
    print("    results in lower bias because more data is used for training on each fold, but increases variance")
    print("    and computational cost. A smaller k (e.g., 5) reduces variance and computational burden but may have")
    print("    slightly higher bias. Common practice is to use 5- or 10-fold cross-validation.")
    
    print("\nQ2: How did you handle missing / NaN entries in the dataset? Why did you use that method?")
    print("A2: In this implementation, we assume the data was preprocessed. In practice, missing values can be handled")
    print("    by imputation methods (e.g., using the mean or median for numerical features, or the most frequent value")
    print("    for categorical features) or by using models that can handle missing data. Imputation preserves the number")
    print("    of samples and avoids discarding potentially valuable information.")
