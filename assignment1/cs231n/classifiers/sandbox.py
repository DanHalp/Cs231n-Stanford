import numpy as np

num_of_points = 10
x = np.arange(num_of_points)
x = np.array_split(x, 5, axis=0)
num_folds = 5

x = np.array(np.array_split(np.array([[0, x] for x in range(10)]), num_folds))
y = np.array(np.array_split(sorted(np.random.choice(np.arange(3), num_of_points)), num_folds))

for i in range(num_folds - 1, -1, -1):
    indices = np.array(np.delete(np.arange(num_folds), i))
    train_x, y_train = x[indices], y[indices]
    train_x = train_x.reshape(-1, train_x.shape[-1])
    train_y = y_train.reshape(-1)
    x=3