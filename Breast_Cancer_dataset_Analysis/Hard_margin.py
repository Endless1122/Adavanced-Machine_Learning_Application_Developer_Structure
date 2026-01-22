from sklearn.svm import SVC
from visual import plot_2d_data, plot_svm_decision_boundary
from data import generate_Overlpping_data, generate_linear_data

def main():
    X, y = generate_linear_data()
    svm_hard = SVC(C=1e6, kernel='linear')  # Very high C for hard margin
    svm_hard.fit(X, y)
    # print("Support Vectors (Hard Margin):")
    # print(svm_hard.support_vectors_)

    w = svm_hard.coef_[0]
    b = svm_hard.intercept_[0]
    print("Weight vector (w):", w)
    print("Bias (b):", b)

    plot_svm_decision_boundary(svm_hard,X, y, w, b, title="Hard Margin SVM Data")

    X_overlap, y_overlap = generate_Overlpping_data()
    plot_2d_data(X_overlap, y_overlap, title="Overlapping Data for Hard Margin SVM")
    svm_hard = SVC(C=1e6, kernel='linear')  # Very high C for hard margin
    svm_hard.fit(X_overlap, y_overlap)
    # print("Support Vectors (Hard Margin):")
    # print(svm_hard.support_vectors_)

    w = svm_hard.coef_[0]
    b = svm_hard.intercept_[0]
    print("Weight vector (w):", w)
    print("Bias (b):", b)

    plot_svm_decision_boundary(svm_hard,X_overlap, y_overlap, w, b, title="Hard Margin SVM OverlappedData")

if __name__ == "__main__":
    main()