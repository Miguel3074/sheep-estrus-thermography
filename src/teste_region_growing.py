import numpy as np
import matplotlib.pyplot as plt
import cv2 # Adicionado para ler a foto visual

def region_growing(matriz, semente_y, semente_x, tolerancia=0.5):
    altura, largura = matriz.shape
    mascara = np.zeros((altura, largura), dtype=bool)

    fila = [(semente_y, semente_x)]
    mascara[semente_y, semente_x] = True

    temp_semente = matriz[semente_y, semente_x]
    limite_inferior = temp_semente - tolerancia

    direcoes = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while fila:
        y, x = fila.pop(0)

        for dy, dx in direcoes:
            ny, nx = y + dy, x + dx

            if 0 <= ny < altura and 0 <= nx < largura and not mascara[ny, nx]:
                if matriz[ny, nx] >= limite_inferior:
                    mascara[ny, nx] = True
                    fila.append((ny, nx))

    return mascara

# --- Carregamento dos Dados ---
caminho_npy = r'..\data\Lote Vermelho\Ovelha 10\FLIR0420.npy'
# Pega o mesmo caminho, mas troca o final para carregar a foto visual
caminho_jpg = caminho_npy.replace('.npy', '.jpg')

# 1. Carrega a matriz térmica (80x60)
matriz_termica = np.load(caminho_npy)

# 2. Carrega a imagem visual RGB (320x240)
img_bgr = cv2.imread(caminho_jpg)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# --- Processamento Térmico ---
indice_maximo = np.argmax(matriz_termica)
y_semente, x_semente = np.unravel_index(indice_maximo, matriz_termica.shape)
temp_max = matriz_termica[y_semente, x_semente]

print(f"Pixel mais quente (Semente) -> X: {x_semente}, Y: {y_semente} | Temp: {temp_max:.2f}°C")

mascara_vulva = region_growing(matriz_termica, y_semente, x_semente, tolerancia=0.8)

imagem_segmentada = np.copy(matriz_termica)
imagem_segmentada[~mascara_vulva] = np.nan

# --- Plotagem ---
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))

# Plot 1: Imagem Visual (A foto real da ovelha)
ax1.imshow(img_rgb)
ax1.set_title('Imagem Visual (Original)')
ax1.axis('off') # Tira os eixos numéricos para a foto ficar limpa

# Plot 2: Termograma com a Semente
im2 = ax2.imshow(matriz_termica, cmap='magma')
ax2.plot(x_semente, y_semente, 'w*', markersize=10, markeredgecolor='black', label='Hotspot (Semente)')
ax2.set_title('Termograma (Semente)')
ax2.legend()

# Plot 3: Máscara Binária
ax3.imshow(mascara_vulva, cmap='gray')
ax3.set_title('Máscara do Region Growing')

# Plot 4: Região Segmentada
im4 = ax4.imshow(imagem_segmentada, cmap='magma', vmin=np.nanmin(matriz_termica), vmax=np.nanmax(matriz_termica))
ax4.set_title('Região Segmentada')

# Barra de cor conectada apenas aos gráficos térmicos
plt.colorbar(im2, ax=[ax2, ax3, ax4], orientation='horizontal', fraction=0.05, pad=0.1, label='Temperatura (°C)')

plt.tight_layout()
plt.show()