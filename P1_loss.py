import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

# ---------------------------
# Loss Function Implementations
# ---------------------------

def indicator_loss(y_true, y_pred):
    """
    Indicator loss function:
    Returns 1 if prediction is incorrect, 0 otherwise.
    (This is provided as an example.)
    """
    y_pred_rounded = np.round(y_pred)  # Convert probabilities to binary predictions
    return np.mean(y_true != y_pred_rounded)  # Fraction of incorrect predictions

def squared_loss(y_true, y_pred):
    """
    Squared Loss (L2 loss):
    L = (y_true - y_pred)^2
    Penalizes errors quadratically and is very sensitive to outliers.
    Suitable for regression tasks where errors are assumed to be normally distributed.
    """
    return (y_true - y_pred) ** 2

def absolute_loss(y_true, y_pred):
    """
    Absolute Loss (L1 loss):
    L = |y_true - y_pred|
    Penalizes errors linearly. More robust to outliers than squared loss,
    but its gradient is not smooth everywhere.
    """
    return np.abs(y_true - y_pred)

def huber_loss(y_true, y_pred, delta=0.5):
    """
    Huber Loss:
    A combination of squared loss (for small errors) and absolute loss (for large errors).
    
    For an error 'e = y_true - y_pred':
      - If |e| <= delta: loss = 0.5 * e^2
      - If |e| >  delta: loss = delta * (|e| - 0.5 * delta)
    
    Delta determines the threshold between quadratic and linear behavior.
    Huber loss is robust to outliers while remaining differentiable.
    """
    error = y_true - y_pred
    is_small_error = np.abs(error) <= delta
    loss = np.where(is_small_error,
                    0.5 * error**2,
                    delta * (np.abs(error) - 0.5 * delta))
    return loss

def logistic_loss(y_true, y_pred, eps=1e-15):
    """
    Logistic Loss (Binary Cross-Entropy Loss):
    L = - [y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]
    
    y_pred is clipped to avoid log(0).
    This loss is most suitable for binary classification tasks.
    """
    # Ensure numerical stability by clipping predicted probabilities
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def hinge_loss(y_true, y_pred):
    """
    Hinge Loss:
    Typically used in Support Vector Machines (SVMs).
    
    For binary classification with labels in {0,1}, we first transform:
      y_true_mapped = 2*y_true - 1   --> now in {-1, +1}
    
    Also, we transform the predicted probability to a score:
      score = 2*y_pred - 1  (so that higher probability gives a higher score)
    
    Then the hinge loss is:
      L = max(0, 1 - y_true_mapped * score)
    
    It penalizes predictions that are not confidently correct.
    """
    # Convert y_true from {0,1} to {-1, +1}
    y_true_mapped = 2 * y_true - 1
    # Convert predicted probability to a decision score
    score = 2 * y_pred - 1
    return np.maximum(0, 1 - y_true_mapped * score)

# ---------------------------
# Dataset Generation
# ---------------------------
np.random.seed(42)  # For reproducibility

# Generate 100 samples:
# y_true: binary labels (0 or 1)
y_true_dataset = np.random.randint(0, 2, size=100)
# y_pred: predicted probabilities between 0 and 1
y_pred_dataset = np.random.rand(100)

# Compute and print average loss for each loss function on the dataset:
print("Average losses on the dataset:")
print(f"Indicator Loss: {indicator_loss(y_true_dataset, y_pred_dataset):.4f}")
print(f"Squared Loss:   {np.mean(squared_loss(y_true_dataset, y_pred_dataset)):.4f}")
print(f"Absolute Loss:  {np.mean(absolute_loss(y_true_dataset, y_pred_dataset)):.4f}")
print(f"Huber Loss:     {np.mean(huber_loss(y_true_dataset, y_pred_dataset)):.4f}")
print(f"Logistic Loss:  {np.mean(logistic_loss(y_true_dataset, y_pred_dataset)):.4f}")
print(f"Hinge Loss:     {np.mean(hinge_loss(y_true_dataset, y_pred_dataset)):.4f}")

# ---------------------------
# 3D Visualization of Loss Functions
# ---------------------------

def plot_loss_surface(loss_fn, title, delta=None, eps=None):
    """
    Plots a 3D surface of the given loss function.
    
    Parameters:
      - loss_fn: the loss function to visualize
      - title: title for the plot
      - delta: (optional) parameter for Huber loss
      - eps: (optional) parameter for logistic loss
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a grid of values for y_true and y_pred.
    # Although y_true in our dataset is binary, we plot over [0,1] for illustration.
    y_true_vals = np.linspace(0, 1, 100)
    y_pred_vals = np.linspace(0, 1, 100)
    Y_true, Y_pred = np.meshgrid(y_true_vals, y_pred_vals)
    
    # Compute the loss over the grid. Pass additional parameters if needed.
    if delta is not None:
        Z = loss_fn(Y_true, Y_pred, delta)
    elif eps is not None:
        Z = loss_fn(Y_true, Y_pred, eps)
    else:
        Z = loss_fn(Y_true, Y_pred)
    
    # Plot the surface
    surf = ax.plot_surface(Y_true, Y_pred, Z, cmap='viridis', edgecolor='none', alpha=0.8)
    ax.set_xlabel('y_true')
    ax.set_ylabel('y_pred')
    ax.set_zlabel('Loss')
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

# Plot each loss function
plot_loss_surface(squared_loss, "Squared Loss Surface")
plot_loss_surface(absolute_loss, "Absolute Loss Surface")
plot_loss_surface(huber_loss, "Huber Loss Surface (delta = 0.5)", delta=0.5)
plot_loss_surface(logistic_loss, "Logistic Loss Surface", eps=1e-15)
plot_loss_surface(hinge_loss, "Hinge Loss Surface")

# ---------------------------
# Short Explanation of Each Loss Function
# ---------------------------
print("\nExplanation of Loss Functions:")
print("1. Squared Loss: Penalizes errors quadratically. Sensitive to outliers. "
      "Commonly used in regression problems with Gaussian noise.")
print("2. Absolute Loss: Penalizes errors linearly. More robust to outliers compared to squared loss, "
      "but may lead to less smooth gradients. Often used in robust regression.")
print("3. Huber Loss: Combines squared and absolute losses by being quadratic for small errors "
      "and linear for large errors. It provides a balance between sensitivity and robustness, making it "
      "suitable when outliers are present but you still desire differentiability.")
print("4. Logistic Loss: Measures the performance of a classifier by heavily penalizing misclassified "
      "instances. It is the basis of logistic regression and is best for binary classification tasks.")
print("5. Hinge Loss: Commonly used with support vector machines (SVMs). It penalizes not only misclassifications "
      "but also predictions that are correct yet lack confidence. It is ideal for margin-based classification.")

