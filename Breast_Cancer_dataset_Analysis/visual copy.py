import matplotlib.pyplot as plt
import numpy as np

def plot_2d_data(X, y, title="Linearly Separable Data"):
    plt.figure(figsize=(7, 6))

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap="bwr",
        edgecolors="k",
        alpha=0.8,
        s=60
    )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("x1", fontsize=12)
    plt.ylabel("x2", fontsize=12)

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_svm_decision_boundary(svm, X, y, w ,b ,title):
    plt.figure(figsize=(8,6))

    plt.scatter(X[:,0], X[:,1], c=y,cmap="bwr", edgecolors='k')

    plt.scatter(
        svm.support_vectors_[:, 0],
        svm.support_vectors_[:, 1],
        s=120,facecolors='none',
        edgecolors='k',linewidths=2,
        label='Support Vectors'
    )

    x_vals = np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200)

    # Decision boundary:w-x+b = 0
    y_decision = -(w[0] * x_vals + b) / w[1]

    # Decision boundary:w-x+b = +-1
    y_decision_pos = -(w[0] * x_vals + b - 1) / w[1]
    y_decision_neg = -(w[0] * x_vals + b + 1) / w[1]

    plt.plot(x_vals, y_decision, 'k-', label='Decision Boundary')
    plt.plot(x_vals, y_decision_pos, 'k--', label='Margin +1')
    plt.plot(x_vals, y_decision_neg, 'k--', label='Margin -1')

    plt.fill_between(
        x_vals, y_decision_pos, y_decision_neg, alpha=0.2, color='gray'
    )

    plt.title(title)
    plt.xlabel("Feature 1 (x1)")
    plt.ylabel("Feature 2 (x2)")
    plt.legend()