import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.graphics.tsaplots import plot_acf
import yfinance as yf

# 1. Daten holen
df = yf.download("AAPL", start="2024-01-01", end="2025-01-01")
df = df[["Close"]].dropna()

# 2. Zeitreihe vorbereiten
series = df["Close"]

# 3. Additive Zerlegung mit fester Periode (7 Tage)
decompose_result = seasonal_decompose(series, model="additive", period=7)

# 4. STL-Zerlegung (robust, lokal)
stl = STL(series, period=7)
stl_result = stl.fit()

# 5. Autokorrelationsanalyse (ACF)
plt.figure(figsize=(10, 4))
plot_acf(series, lags=60)
plt.title("Autokorrelation (ACF)")
plt.tight_layout()
plt.show()

# 6. Fourier-Transformation zur Frequenzanalyse
x = series.values - np.mean(series.values)
fft = np.fft.fft(x)
frequencies = np.fft.fftfreq(len(x))
amplitudes = np.abs(fft)

plt.figure(figsize=(10, 4))
plt.plot(frequencies[1:len(x)//2], amplitudes[1:len(x)//2])
plt.title("Frequenzanalyse (Fourier-Transformation)")
plt.xlabel("Frequenz")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

# 7. Vergleich der Zerlegungen
fig, axes = plt.subplots(4, 2, figsize=(15, 12))

# Original
axes[0, 0].plot(series)
axes[0, 0].set_title("Original-Zeitreihe")

# Additive Decomposition
axes[1, 0].plot(decompose_result.trend)
axes[1, 0].set_title("Trend (Seasonal Decompose)")
axes[2, 0].plot(decompose_result.seasonal)
axes[2, 0].set_title("Saisonalität (Seasonal Decompose)")
axes[3, 0].plot(decompose_result.resid)
axes[3, 0].set_title("Residuum (Seasonal Decompose)")

# STL Decomposition
axes[1, 1].plot(stl_result.trend)
axes[1, 1].set_title("Trend (STL)")
axes[2, 1].plot(stl_result.seasonal)
axes[2, 1].set_title("Saisonalität (STL)")
axes[3, 1].plot(stl_result.resid)
axes[3, 1].set_title("Residuum (STL)")

plt.tight_layout()
plt.show()
