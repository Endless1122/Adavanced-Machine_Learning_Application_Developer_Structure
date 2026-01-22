from sklearn.svm import SVC
from visual import plot_2d_data, plot_svm_decision_boundary
from data import generate_Overlpping_data, generate_linear_data
from matplotlib import pyplot as plt

def main():
    X, y = generate_linear_data()
    c_list = [0.1, 1, 10, 100]
    for c in c_list:
        svm_Soft = SVC(C=c, kernel='linear')  # Very high C for Soft margin
        svm_Soft.fit(X, y)
        print("Support Vectors (Soft Margin):")
        # print(svm_Soft.support_vectors_)

        w = svm_Soft.coef_[0]
        b = svm_Soft.intercept_[0]
        print("Weight vector (w):", w)
        print("Bias (b):", b)

        plot_svm_decision_boundary(svm_Soft,X, y, w, b, title="Soft Margin SVM Data for C={}".format(c))

        X_overlap, y_overlap = generate_Overlpping_data()
        # plot_2d_data(X_overlap, y_overlap, title="Overlapping Data for Soft Margin SVM")
        svm_Soft = SVC(C=c, kernel='linear')  # Very high C for Soft margin
        svm_Soft.fit(X_overlap, y_overlap)
        print("Support Vectors (Soft Margin):")
        # print(svm_Soft.support_vectors_)

        w = svm_Soft.coef_[0]
        b = svm_Soft.intercept_[0]
        print("Weight vector (w):", w)
        print("Bias (b):", b)

        plot_svm_decision_boundary(svm_Soft,X_overlap, y_overlap, w, b, title="Soft Margin SVM Overlapped Data for C={}".format(c))

    plt.show()


if __name__ == "__main__":
    main()