import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os

# Параметри сигналу
Fs = 1000  # Частота дискретизації
F_filter = 12  # Полоса пропускання фільтру
n = 500  # Довжина сигналу у відліках
F_max = 5  # Максимальна частота сигналу

# Генерація випадкового сигналу
mean = 0
std_deviation = 10
random_signal = np.random.normal(mean, std_deviation, n)

# Визначення параметрів фільтру
order = 3  # Порядок фільтру
w = F_filter / (Fs / 2)

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

# Розрахунок рівнів квантування
M = 16
delta = (np.max(filtered_signal) - np.min(filtered_signal)) / (M - 1)
quantized_signal = delta * np.floor(filtered_signal / delta)
quantize_levels = np.arange(np.min(quantized_signal), np.max(quantized_signal)+delta, delta)
bits_required = int(np.ceil(np.log2(M)))
quantize_bits = [format(i, '0' + str(bits_required) + 'b') for i in range(M)]

# Створення таблиці квантування
quantize_table = np.c_[quantize_levels, quantize_bits]

# Відображення таблиці квантування у вигляді рисунку
fig, ax = plt.subplots(figsize=(8, 6))  # Змініть розмір за потребою
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=quantize_table, colLabels=['Значення сигналу', 'Кодова послідовність'], loc='center')
plt.show()

fig.savefig(os.path.join(f'./figures/Таблиця квантування для 4 рівня.png'), dpi=600)

# Розрахунок квантованих сигналів та бітових послідовностей
M = 4  # Кількість рівнів квантування
delta = (np.max(filtered_signal) - np.min(filtered_signal)) / (M - 1)
quantized_signal = delta * np.floor(filtered_signal / delta)
quantize_levels = np.linspace(np.min(filtered_signal), np.max(filtered_signal), M)
quantize_bits = [format(i, '0' + str(int(np.log2(M))) + 'b') for i in range(M)]

# Визначення бітових послідовностей для квантованого сигналу
bits_sequence = []
for value in quantized_signal:
    index = np.argmin(np.abs(quantize_levels - value))
    bits_sequence.append(quantize_bits[index])

# Об'єднання бітових послідовностей у єдиний масив
bits = ''.join(bits_sequence)
bits_array = np.array([int(bit) for bit in bits])

# Створення графіку бітових послідовностей
fig, ax = plt.subplots(figsize=(21/2.54, 14/2.54))
ax.step(range(len(bits_array)), bits_array, where='post', linewidth=0.5)
ax.set_xlabel('Біти')
ax.set_ylabel('Амплітуда')
ax.set_title(f'Кодова послідовність сигналу при кількості рівнів квантування {M}')
plt.show()

figures_directory = './figures'
os.makedirs(figures_directory, exist_ok=True)
fig.savefig(os.path.join(figures_directory, f'bit_sequence_quantization_{M}_levels.png'), dpi=300)
plt.close(fig)

# Визначаємо функцію для квантування сигналу
def quantize_signal(signal, num_levels):
    min_val, max_val = np.min(signal), np.max(signal)
    quantized_levels = np.linspace(min_val, max_val, num_levels)
    quantized_signal = np.digitize(signal, quantized_levels, right=True) - 1
    quantized_signal = quantized_levels[quantized_signal]
    return quantized_signal

# Функція для розрахунку дисперсії та співвідношення сигнал-шум
def calculate_variance_and_snr(signal, quantized_signal):
    noise = signal - quantized_signal
    variance = np.var(noise)
    snr = 10 * np.log10(np.var(signal) / variance)
    return variance, snr

# Цикл для квантування сигналу та розрахунку дисперсії та співвідношення сигнал-шум
variances = []
snr_values = []
M_values = [4, 16, 64, 256]
quantized_signals = []

for M in M_values:
    quantized_signal = quantize_signal(filtered_signal, M)
    quantized_signals.append(quantized_signal)
    variance, snr = calculate_variance_and_snr(filtered_signal, quantized_signal)
    variances.append(variance)
    snr_values.append(snr)

# Побудова графіків цифрових сигналів
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()
for i, M in enumerate(M_values):
    axs[i].step(t, quantized_signals[i], where='post')
    axs[i].set_title(f'Цифровий сигнал з рівнями квантування {M}')
    axs[i].set_xlabel('Час (секунди)')
    axs[i].set_ylabel('Амплітуда сигналу')
plt.tight_layout()
plt.savefig(os.path.join(figures_directory, 'Цифрові сигнали з рівнями квантування.png'), dpi=300)
plt.close(fig)

# Побудова і збереження графіка залежності дисперсії
plt.figure(figsize=(10, 6))
plt.plot(M_values, variances, marker='o')
plt.title('Залежність дисперсії від кількості рівнів квантування')
plt.xlabel('Кількість рівнів квантування')
plt.ylabel('Дисперсія')
plt.grid(True)
plt.savefig(os.path.join(figures_directory, 'Залежність дисперсії від рівнів квантування.png'), dpi=300)
plt.close()

# Побудова і збереження графіка залежності співвідношення сигнал-шум
plt.figure(figsize=(10, 6))
plt.plot(M_values, snr_values, marker='o')
plt.title('Залежність співвідношення сигнал-шум від кількості рівнів квантування')
plt.xlabel('Кількість рівнів квантування')
plt.ylabel('Співвідношення сигнал-шум (дБ)')
plt.grid(True)
plt.savefig(os.path.join(figures_directory, 'Залежність співвід. сигнал-шум від квантування.png'), dpi=300)
plt.close()


# Функція для створення і збереження таблиці квантування
def create_and_save_quantization_table(M, signal, figures_directory):
    # Розрахунок рівнів квантування
    quantization_step = (np.max(signal) - np.min(signal)) / (M - 1)
    quantized_levels = np.linspace(np.min(signal), np.max(signal), M)

    # Розрахунок бітових послідовностей
    bits_required = int(np.ceil(np.log2(M)))
    quantize_bits = [format(i, '0' + str(bits_required) + 'b') for i in range(M)]

    # Створення таблиці квантування
    quantization_table = np.c_[quantized_levels, quantize_bits]

    # Візуалізація та збереження таблиці
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    table = ax.table(cellText=quantization_table, colLabels=['Рівень', 'Біти'], loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    plt.savefig(os.path.join(figures_directory, f'Таблиці квантування з рівнями {M}.png'), dpi=300)
    plt.close(fig)


# Створення та збереження таблиць квантування для кожного M
for M in [16, 64, 256]:
    create_and_save_quantization_table(M, filtered_signal, figures_directory)