from data import generate_xor_data
from visual import plot_2d_data
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
 
def main():
    X,y = generate_xor_data(n=200)
    plot_2d_data(X, y, title="XOR Data (original Space)")
 
    #Linear SVM
    linear_svm = SVC(kernel='linear')
    linear_svm.fit(X,y)
    y_pred_linear = linear_svm.predict(X)
    plot_2d_data(X, y_pred_linear, title = "Linear SVM Prediction (Fails)")
 
    # Polynomial Kernel SVM
    poly_svm = SVC(kernel='poly', degree=2)
    poly_svm.fit(X,y)
    y_pred_poly = poly_svm.predict(X)
    plot_2d_data(X, y_pred_poly, title = "Polynomial kernel SVM Predictionr (Succeds)")
 
    # RBF Kernel SVM
    rbf_svm = SVC(kernel='rbf', gamma='scale')
    rbf_svm.fit(X,y)
    y_pred_rbf = rbf_svm.predict(X)
    plot_2d_data(X, y_pred_rbf, title = "RBF kernel SVM Prediction (Succeds)")

    # Sigmoid Kernel SVM
    sigmoid_svm = SVC(kernel='sigmoid', gamma='scale', coef0=1)
    sigmoid_svm.fit(X, y)
    y_pred_sigmoid = sigmoid_svm.predict(X)
    plot_2d_data(X, y_pred_sigmoid, title="Sigmoid Kernel SVM Prediction")

    # Polynomial Kernel SVM (degree = 3)
    poly3_svm = SVC(kernel='poly', degree=3)
    poly3_svm.fit(X, y)
    y_pred_poly3 = poly3_svm.predict(X)
    plot_2d_data(X, y_pred_poly3, title="Polynomial Kernel (degree=3) SVM Prediction")
 
    # Print Accuracies
    print("Linear SVM Accuracy: ", accuracy_score(y, y_pred_linear))
    print("Polynomial Kernel SVM Accuracy: ", accuracy_score(y, y_pred_poly))
    print("RBF Kernel SVM Accuracy: ", accuracy_score(y, y_pred_rbf))
    print("Sigmoid Kernel SVM Accuracy: ", accuracy_score(y, y_pred_sigmoid))
    print("Polynomial Kernel (degree=3) SVM Accuracy: ", accuracy_score(y, y_pred_poly3))

if __name__ == "__main__":
    main()