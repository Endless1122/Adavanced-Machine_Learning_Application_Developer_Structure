from sklearn.datasets import load_breast_cancer

def generate_data():
    dataset = load_breast_cancer()
    X = dataset.data
    y = dataset.target
    return X,y