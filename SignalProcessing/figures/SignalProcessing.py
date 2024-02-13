import numpy as np
# Генерація випадкового сигналу
mean = 0
std_deviation = 10
n = 500  # Довжина сигналу у відліках
Fs = 1000  # Частота дискретизації

# Генерація випадкового сигналу
random_signal = np.random.normal(mean, std_deviation, n)
# Визначення відліків часу
time_values = np.arange(n) / Fs

# Результат
print("Відліки часу:", time_values)
