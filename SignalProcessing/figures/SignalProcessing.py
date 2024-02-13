import numpy as np
mean = 0  # середній розподіл
std_deviation = 10  # стандартне відхилення розподілу
num_elements = 100  # кількість згенерованих елементів

# Генеруємо випадковий сигнал
random_signal = np.random.normal(mean, std_deviation, num_elements)

# Результат
print("Генерування випадкового сигналу:", random_signal)
