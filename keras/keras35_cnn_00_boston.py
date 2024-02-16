from sklearn.datasets import load_breast_cancer
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (569, 30) (569,)  -> (569, 15, 2, 1) -> (569, 10, 3, 1) -> (569, 3, 5, 2)