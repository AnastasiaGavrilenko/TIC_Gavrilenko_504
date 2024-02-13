import numpy as np
from scipy import signal

mean = 0
std_deviation = 10
n = 500  # Довжина сигналу у відліках
Fs = 1000  # Частота дискретизації

# Генерація випадкового сигналу
random_signal = np.random.normal(mean, std_deviation, n)

# Визначення параметрів фільтру
order = 3  # порядок фільтру
F_max = 5  # максимальна частота сигналу в Гц
w = F_max / (Fs / 2)
b, a = signal.butter(order, w, 'low', output='ba')

# Фільтрація сигналу
filtered_signal = signal.sosfiltfilt(signal.butter(order, w, 'low', output='sos'), random_signal)

# Результат
print("Фільтрований сигнал:", filtered_signal)
