from data import generate_linear_data
from visual import plot_2d_data

X,y = generate_linear_data(n=100)
plot_2d_data(X, y, title="Linear Data")