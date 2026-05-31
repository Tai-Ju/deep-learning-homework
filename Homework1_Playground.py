# ============================================================
# Homework (1) - Neural Network Playground Exercise
# Simulates Google ML Playground using sklearn + matplotlib
# Dataset: Circle (make_circles), 2 hidden layers
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# ============================================================
# 1. Generate Dataset (Circle - same as Playground)
# ============================================================
np.random.seed(42)
X, y = make_circles(n_samples=500, noise=0.3, factor=0.5, random_state=42)

# Train/Test split: 70% / 30%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ============================================================
# 2. Add Non-linear Features (X1^2, X2^2, X1*X2)
#    Same as enabling extra features in Playground
# ============================================================
def add_features(X):
    x1, x2 = X[:, 0], X[:, 1]
    return np.column_stack([x1, x2, x1**2, x2**2, x1*x2,
                            np.sin(x1), np.sin(x2)])

X_train_feat = add_features(X_train)
X_test_feat  = add_features(X_test)

# ============================================================
# 3. Build Model: 2 Hidden Layers (4 neurons + 2 neurons)
#    Matches Playground default configuration
# ============================================================
model = MLPClassifier(
    hidden_layer_sizes=(4, 2),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    batch_size=10,
    max_iter=1000,
    random_state=42,
    verbose=False
)

model.fit(X_train_feat, y_train)

train_acc = accuracy_score(y_train, model.predict(X_train_feat))
test_acc  = accuracy_score(y_test,  model.predict(X_test_feat))
print(f"Train Accuracy : {train_acc:.4f}")
print(f"Test  Accuracy : {test_acc:.4f}")
print(f"Train Loss     : {model.loss_:.4f}")

# ============================================================
# 4. Decision Boundary Plot
# ============================================================
def plot_decision_boundary(model, X_raw, y, title="Decision Boundary"):
    h = 0.05
    x_min, x_max = X_raw[:, 0].min() - 0.5, X_raw[:, 0].max() + 0.5
    y_min, y_max = X_raw[:, 1].min() - 0.5, X_raw[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_feat = add_features(grid)
    Z = model.predict(grid_feat).reshape(xx.shape)

    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, Z, alpha=0.45, cmap=plt.cm.RdBu)
    scatter = plt.scatter(X_raw[:, 0], X_raw[:, 1], c=y,
                          cmap=plt.cm.RdBu, edgecolors='k',
                          s=35, linewidths=0.6)
    plt.colorbar(scatter, label="Class")
    plt.title(title, fontsize=13)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.tight_layout()
    plt.savefig("hw1_decision_boundary.png", dpi=150)
    plt.show()
    print("Saved: hw1_decision_boundary.png")

plot_decision_boundary(
    model, X_test, y_test,
    title=f"Homework1 Decision Boundary  (Test Acc={test_acc:.2%})"
)

# ============================================================
# 5. Loss Curve
# ============================================================
plt.figure(figsize=(7, 4))
plt.plot(model.loss_curve_, color='steelblue', linewidth=2)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("hw1_loss_curve.png", dpi=150)
plt.show()
print("Saved: hw1_loss_curve.png")

# ============================================================
# 6. Experiment: Compare different hidden layer configs
# ============================================================
configs = {
    "4+2  (default)": (4, 2),
    "8+4":            (8, 4),
    "16+8":           (16, 8),
    "32+16":          (32, 16),
}

print("\n--- Hidden Layer Experiment ---")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for label, hidden in configs.items():
    clf = MLPClassifier(hidden_layer_sizes=hidden, activation='relu',
                        solver='adam', learning_rate_init=0.001,
                        batch_size=10, max_iter=1000, random_state=42)
    clf.fit(X_train_feat, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test_feat))
    print(f"  Hidden layers {label:15s}  Test Acc: {acc:.4f}")
    axes[0].plot(clf.loss_curve_, label=label)
    axes[1].bar(label, acc, color='steelblue', alpha=0.7)

axes[0].set_title("Loss Curve by Hidden Layer Config")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

axes[1].set_title("Test Accuracy by Hidden Layer Config")
axes[1].set_ylabel("Accuracy")
axes[1].set_ylim(0, 1)
axes[1].tick_params(axis='x', labelsize=8)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("hw1_hidden_layer_experiment.png", dpi=150)
plt.show()
print("Saved: hw1_hidden_layer_experiment.png")
