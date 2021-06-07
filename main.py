import numpy as np

def sigmoid(z):
    """
    Сигмоидная функция активации
    :param z: сумма весов
    :return: сигмоида
    """
    return 1 / 1 + np.exp(-z)


def forward_propagation(X, Y, W1, b1, W2, b2):
    """
    Вычисляет операцию прямого распространения перцептрона и
    возвращает результат после применения пошаговой фукции фктивации
    """
    net_h = np.dot(W1, X) + b1
    out_h = sigmoid(net_h)
    net_y = np.dot(W2, out_h) + b2
    out_y = sigmoid(net_y)
    return out_y, out_y

## Инициализация параметров
np.random.seed(42) # инициализируем с некоторыми случайными числами

X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]]) # входной массив
Y = np.array([[0, 1, 1, 0]]) # целевые метки
n_h = 2 # количество нейронов в скрытом слое
n_x = X.shape[0] # количество нейронов во входном слое
n_y = Y.shape[0] # количество нейронов в выходном слое
W1 = np.random.randn(n_h, n_x) # веса из входного слоя
b1 = np.zeros((n_h, 1)) # смещение в скрытом слое
W2 = np.random.randn(n_y, n_h) # веса из скрытого слоя
b2 = np.zeros((n_y, 1)) # смещение в выходном слое

# вычисляем проход прямого распределения
A1, A2 = forward_propagation(X, Y, W1, b1, W2, b2)

pred = (A2 > 0.5) * 1
print("Predicted label:", pred) # прогноз