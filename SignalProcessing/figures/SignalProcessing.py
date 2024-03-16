import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os

# Параметри сигналу
Fs = 1000  # Частота дискретизації
F_filter = 12  # Полоса пропускання фільтру
n = 500  # Довжина сигналу у відліках

# Генерація випадкового сигналу
mean = 0
std_deviation = 10
random_signal = np.random.normal(mean, std_deviation, n)

# Визначення параметрів фільтру
order = 3  # Порядок фільтру
w = F_filter / (Fs / 2)  # Нормалізована частота зрізу

# Фільтрація сигналу
filtered_signal = signal.sosfiltfilt(signal.butter(order, w, 'low', output='sos'), random_signal)

# Часова вісь для сигналу
t = np.arange(n) / Fs

# Дискретизація сигналу з різними кроками
Dt_values = [2, 4, 8, 16]
discrete_signals = []

for Dt in Dt_values:
    idx = np.arange(0, n, Dt)
    discrete_signal = np.zeros(n)
    discrete_signal[idx] = filtered_signal[idx]
    discrete_signals.append(discrete_signal)


def plot_signals_like_image(t, signals, Dt_values, title, xlabel, ylabel, width, height, line_width, font_size):
    fig, axs = plt.subplots(2, 2, figsize=(width / 2.54, height / 2.54), constrained_layout=True)
    fig.suptitle(title, fontsize=font_size)

    for i, Dt in enumerate(Dt_values):
        discrete_signal = np.zeros(n)
        idx = np.arange(0, n, Dt)
        discrete_signal[idx] = signals[i][idx]

        ax = axs[i // 2, i % 2]
        ax.fill_between(t, discrete_signal, 0, where=(discrete_signal != 0), color='blue', step='pre', alpha=0.5)
        markerline, stemlines, baseline = ax.stem(t, discrete_signal, linefmt='blue', markerfmt='bo', basefmt=' ')
        plt.setp(markerline, 'markersize', 2)  # Задаємо розмір маркерів
        plt.setp(stemlines, 'linewidth', line_width)  # Задаємо товщину ліній

        ax.set_xlabel(xlabel, fontsize=font_size)
        ax.set_ylabel(ylabel, fontsize=font_size)
        ax.set_title(f'Dt = {Dt}', fontsize=font_size)
        ax.grid(True)

    # Підписи для осей
    fig.supxlabel(xlabel, fontsize=font_size)
    fig.supylabel(ylabel, fontsize=font_size)

    # Зберігання графіків
    save_path_like_image = f'./figures/discrete_signal'
    os.makedirs(os.path.dirname(save_path_like_image), exist_ok=True)
    plt.savefig(save_path_like_image, dpi=600)
    plt.close(fig)
    return save_path_like_image


plot_title = 'Сигнал з кроком дискретизації Dt = (2,4,8,16)'
plot_xlabel = 'Час (секунди)'
plot_ylabel = 'Амплітуда сигналу'

save_path_like_image = plot_signals_like_image(t, discrete_signals, Dt_values, plot_title, plot_xlabel, plot_ylabel, 21,
                                               14, 1, 14)

# Функція для побудови графіків спектрів
def plot_spectrums(freqs, spectrums, title, xlabel, ylabel, width, height, line_width, font_size):
    fig, axs = plt.subplots(2, 2, figsize=(width/2.54, height/2.54), constrained_layout=True)
    fig.suptitle(title, fontsize=font_size)

    for i, (freq, spectrum) in enumerate(zip(freqs, spectrums)):
        ax = axs[i // 2, i % 2]
        ax.plot(freq, np.abs(spectrum), linewidth=line_width)
        ax.set_title(f'Dt = {Dt_values[i]}', fontsize=font_size)
        ax.set_xlabel(xlabel, fontsize=font_size)
        ax.set_ylabel(ylabel, fontsize=font_size)
        ax.grid(True)

    # Підписи для осей
    fig.supxlabel(xlabel, fontsize=font_size)
    fig.supylabel(ylabel, fontsize=font_size)

    # Зберігання графіків
    save_path_spectrums = f'./figures/spectres_signals'
    os.makedirs(os.path.dirname(save_path_spectrums), exist_ok=True)
    plt.savefig(save_path_spectrums, dpi=600)
    plt.close(fig)
    return save_path_spectrums

# Функція для відображення відновлених аналогових сигналів
def plot_reconstructed_signals(t, reconstructed_signals, Dt_values, title, xlabel, ylabel, width, height, line_width, font_size):
    fig, axs = plt.subplots(2, 2, figsize=(width / 2.54, height / 2.54), constrained_layout=True)
    fig.suptitle(title, fontsize=font_size)

    for i, signal in enumerate(reconstructed_signals):
        ax = axs[i // 2, i % 2]
        ax.plot(t, signal, linewidth=line_width)
        ax.set_title(f'Dt = {Dt_values[i]}', fontsize=font_size)
        ax.set_xlabel(xlabel, fontsize=font_size)
        ax.set_ylabel(ylabel, fontsize=font_size)
        ax.grid(True)

    # Підписи для осей
    fig.supxlabel(xlabel, fontsize=font_size)
    fig.supylabel(ylabel, fontsize=font_size)

    # Зберігання графіків
    save_path_reconstructed = f'./figures/analog_signal'
    os.makedirs(os.path.dirname(save_path_reconstructed), exist_ok=True)
    plt.savefig(save_path_reconstructed, dpi=600)
    plt.close(fig)
    return save_path_reconstructed

# Відновлення аналогових сигналів з дискретних
reconstructed_signals = []
for discrete_signal in discrete_signals:
    # Відновлення сигналу через фільтрацію
    reconstructed_signal = signal.sosfiltfilt(signal.butter(order, w, 'low', output='sos'), discrete_signal)
    reconstructed_signals.append(reconstructed_signal)

# Відображення графіків відновлених аналогових сигналів
reconstructed_title = 'Відновлені аналогові сигнали з кроком дискретизації Dt = (2, 4, 8, 16)'
reconstructed_xlabel = 'Час (секунди)'
reconstructed_ylabel = 'Амплітуда сигналу'

# Виклик функції для відображення відновлених аналогових сигналів
save_path_reconstructed = plot_reconstructed_signals(t, reconstructed_signals, Dt_values, reconstructed_title, reconstructed_xlabel, reconstructed_ylabel, 21, 14, 1, 14)

# Розрахунок дисперсії для відновлених сигналів та співвідношення сигнал-шум
variances = []
snr_values = []

for discrete_signal in reconstructed_signals:
    E1 = discrete_signal - filtered_signal
    variance_E1 = np.var(E1)
    variances.append(variance_E1)
    snr = np.var(filtered_signal) / variance_E1
    snr_values.append(snr)

# Побудова графіків залежності дисперсії від кроку дискретизації
plt.figure(figsize=(10, 6))
plt.plot(Dt_values, variances, marker='o')
plt.title('Залежність дисперсії від кроку дискретизації')
plt.xlabel('Крок дискретизації')
plt.ylabel('Дисперсія')
plt.grid(True)
plt.savefig(os.path.join(f'./figures/dispersiya_vid_discretyzacii.png'), dpi=600)
plt.show()

# Побудова графіків залежності співвідношення сигнал-шум від кроку дискретизації
plt.figure(figsize=(10, 6))
plt.plot(Dt_values, snr_values, marker='o')
plt.title('Залежність співвідношення сигнал-шум від кроку дискретизації')
plt.xlabel('Крок дискретизації')
plt.ylabel('Співвідношення сигнал-шум')
plt.grid(True)
plt.savefig(os.path.join(f'./figures/signal-shum_vid_discretyzacii.png'), dpi=600)
plt.show()