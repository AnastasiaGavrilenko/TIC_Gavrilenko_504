import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os
def plot_signal(x, y, title, xlabel, ylabel, width=21, height=14, line_width=1, font_size=14, save_path=None):
    fig, ax = plt.subplots(figsize=(width/2.54, height/2.54))
    ax.plot(x, y, linewidth=1)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=14)

    if save_path:
        plt.savefig(save_path, dpi=600)
        print(f"Графік '{title}' збережено за шляхом {save_path}")
    else:
        plt.show()

# Генерація випадкового сигналу
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

# Визначення відліків часу
time_values = np.arange(n) / Fs

# Побудова та збереження графіків
figure_dir = "./figures"
os.makedirs(figure_dir, exist_ok=True)

plot_signal(time_values, filtered_signal, "Сигнал з максимальною частотою F_макс = 5 Гц", "Час (с)", "Амплітуда сигналу", save_path=os.path.join(figure_dir, "filtered_signal.png"))
