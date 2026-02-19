import numpy as np

def generate_data_ht():
    np.random.seed(1)

    true_p = 0.7 #(we are taking p)
    data = np.random.binomial(1,true_p,10)
    return data,true_p