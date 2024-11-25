import numpy as np

np.random.seed(42)

# Функция подсчета градиента
def gradient(y_true: int, y_pred: float, x: np.array) -> np.array:
    """
    y_true - истинное значение ответа для объекта x
    y_pred - значение степени принадлежности объекта x классу 1, предсказанное нашей моделью
    x - вектор признакового описания данного объекта

    На выходе ожидается получить вектор частных производных H по параметрам модели, предсказавшей значение y_pred.
    Размерность этого градиента будет на единицу больше, чем размерность x за счет свободного коэффициента a0.
    """
    # Добавляем единицу для учета свободного коэффициента
    x = np.insert(x, 0, 1)  # добавляем 1 в начало x для учета a0
    grad = (y_pred - y_true) * x
    return grad

# Функция обновления весов
def update(alpha: np.array, gradient: np.array, lr: float) -> np.array:
    """
    alpha: текущее приближения вектора параметров модели
    gradient: посчитанный градиент по параметрам модели
    lr: learning rate, множитель перед градиентом в формуле обновления параметров
    """
    alpha_new = alpha - lr * gradient
    return alpha_new

# Функция для подсчета сигмоиды
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Функция тренировки модели
def train(
    alpha0: np.array, x_train: np.array, y_train: np.array, lr: float, num_epoch: int
):
    """
    alpha0 - начальное приближение параметров модели
    x_train - матрица объект-признак обучающей выборки
    y_train - верные ответы для обучающей выборки
    lr - learning rate, множитель перед градиентом в формуле обновления параметров
    num_epoch - количество эпох обучения, то есть полных 'проходов' через весь датасет
    """
    alpha = alpha0.copy()
    for epo in range(num_epoch):
        for i, x in enumerate(x_train):
            # Добавляем единичный столбец к x для свободного коэффициента
            x_with_bias = np.insert(x, 0, 1)  # добавляем 1 для учета свободного коэффициента
            y_pred = sigmoid(np.dot(alpha, x_with_bias))  # предсказание с сигмоидой

            # Считаем градиент
            grad = gradient(y_train[i], y_pred, x)

            # Обновляем веса
            alpha = update(alpha, grad, lr)
    
    return alpha
