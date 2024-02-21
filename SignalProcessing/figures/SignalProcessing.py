from typing import Any

import numpy as np
from numpy import ndarray, dtype
from scipy import signal, fft
import matplotlib.pyplot as plt
import os
def plot_signal(x, y, title, xlabel, ylabel, width=21, height=14, line_width=1, font_size=14, save_path=None):
    fig, ax = plt.subplots(figsize=(width/2.54, height/2.54))
    ax.plot(x, y, linewidth=line_width)
    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    plt.title(title, fontsize=font_size)

    if save_path:
        plt.savefig(save_path, dpi=600)
        print(f"Графік '{title}' збережено за шляхом {save_path}")
    else:
        plt.show()

# Генерація випадкового сигналу
mean = 0
std_deviation = 10
n = 500  # Довжина сигналу у відліках
Fs = 1000  # частота дискретизації

# Генерація випадкового сигналу
random_signal = np.random.normal(mean, std_deviation, n)

# Визначення параметрів фільтру
order = 3  # порядок фільтру
F_max = 5  # максимальна частота сигналу в Гц
w = F_max / (Fs / 2)
b, a = signal.butter(order, w, 'low', output='ba')

# Фільтрація сигналу за допомогою sosfiltfilt
filtered_signal = signal.sosfiltfilt(signal.butter(order, w, 'low', output='sos'), random_signal)

spectrum = fft.fft(filtered_signal)
shifted_spectrum = fft.fftshift(spectrum)
freq = fft.fftfreq(len(filtered_signal), 1/Fs)
shifted_freq = fft.fftshift(freq)

# Визначення параметрів для графіку спектру
title_spectrum = "Спектр сигналу з максимальною частотою F_макс = 5 Гц"
xlabel_spectrum = "Частота (Гц)"
ylabel_spectrum = "Амплітуда спектру"
width_spectrum = 21
height_spectrum = 14
line_width_spectrum = 1
font_size_spectrum = 14

# Графік спектру
figure_dir = "./figures"
os.makedirs(figure_dir, exist_ok=True)
spectrum_save_path = os.path.join(figure_dir, "spectrum.png")

plot_signal(
    shifted_freq,
    np.abs(shifted_spectrum),
    title_spectrum,
    xlabel_spectrum,
    ylabel_spectrum,
    width_spectrum,
    height_spectrum,
    line_width_spectrum,
    font_size_spectrum,
    save_path=spectrum_save_path
)

# Дискретизація сигналу з кроками Dt = 2, 4, 8, 16
Dt_values = [2, 4, 8, 16]
discrete_signals = []

for Dt in Dt_values:
    discrete_signal = np.zeros(n)

    for i in range(0, round(n / Dt)):
        discrete_signal[i * Dt] = filtered_signal[i * Dt]

    discrete_signals.append(list(discrete_signal))

# Відображення результатів
fig, ax = plt.subplots(2, 2, figsize=(21 / 2.54, 14 / 2.54))
fig.suptitle("Сигнал з кроком дискретизації Dt = (2, 4, 8, 16)", fontsize=14)

s = 0
for i in range(0, 2):
    for j in range(0, 2):
        ax[i][j].plot(range(n), discrete_signals[s], linewidth=1)
        s += 1

fig.supxlabel("Час (секунди)", fontsize=14)
fig.supylabel("Амплітуда сигналу", fontsize=14)

# Збереження зображення
figure_dir = "./figures"
os.makedirs(figure_dir, exist_ok=True)
discrete_signals_save_path = os.path.join(figure_dir, "discrete_signals.png")
fig.savefig(discrete_signals_save_path, dpi=600)
print(f"Графіки дискретизованих сигналів збережено за шляхом {discrete_signals_save_path}")
