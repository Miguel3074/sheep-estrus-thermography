import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def anotar_com_backup():
    # 1. Caminhos para os dados
    pasta_data = Path('../data')
    arquivo_original = Path('../planilha/planilha_final.CSV')
    arquivo_saida = Path('../planilha/planilha_anotada.CSV')

    # 2. Lógica Inteligente de Carregamento
    if arquivo_saida.exists():
        print(f"Retomando progresso do arquivo: {arquivo_saida.name}")
        try:
            df = pd.read_csv(arquivo_saida, sep=';')
        except Exception as e:
            print(f"Erro ao ler a planilha anotada: {e}")
            return
    else:
        print(f"Iniciando nova anotação a partir do arquivo original: {arquivo_original.name}")
        try:
            df = pd.read_csv(arquivo_original, sep=';')
        except Exception as e:
            print(f"Erro ao ler a planilha original: {e}")
            return

    # Cria as colunas de coordenadas se elas ainda não existirem
    if 'Coord_X' not in df.columns:
        df['Coord_X'] = pd.NA
    if 'Coord_Y' not in df.columns:
        df['Coord_Y'] = pd.NA

    # Conta o progresso
    anotadas = df['Coord_X'].notna().sum()
    total_validas = df['Foto'].notna().sum()
    print(f"Progresso atual: {anotadas} de {total_validas} imagens válidas anotadas.")

    # 3. Itera sobre cada linha da planilha
    for idx, row in df.iterrows():
        foto_val = row['Foto']

        # Se a linha não tem foto anotada ou você já marcou a coordenada, pula
        if pd.isna(foto_val) or pd.notna(row['Coord_X']):
            continue

        # Formata o número da foto
        try:
            foto_num = int(float(foto_val))
            nome_npy = f"FLIR{foto_num:04d}.npy"
            nome_jpg = f"FLIR{foto_num:04d}.jpg"
        except ValueError:
            print(f"Formato de foto não reconhecido na linha {idx}: {foto_val}")
            continue

        # Formata o nome da ovelha e do lote com base no CSV
        lote_val = row['Lote']
        id_val = row['id']
        id_str = f"Ovelha {int(id_val)}" if isinstance(id_val, float) else f"Ovelha {id_val}"
        lote_str = f"Lote {lote_val}"

        # Monta os caminhos exatos no HD
        caminho_npy = pasta_data / lote_str / id_str / nome_npy
        caminho_jpg = pasta_data / lote_str / id_str / nome_jpg

        if not caminho_npy.exists() or not caminho_jpg.exists():
            print(f"Aviso: Imagem {nome_jpg} não encontrada na pasta {id_str}. Pulando...")
            continue

        # 4. Carrega as matrizes para exibição
        matriz_termica = np.load(caminho_npy)
        img_bgr = cv2.imread(str(caminho_jpg))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 5. Interface Gráfica
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.canvas.manager.set_window_title(f"Anotador - {lote_str} / {id_str} / {nome_npy} (Progresso: {anotadas}/{total_validas})")

        ax1.imshow(img_rgb)
        ax1.set_title('Referência Visual (NÃO CLIQUE AQUI)')
        ax1.axis('off')

        ax2.imshow(matriz_termica, cmap='magma')
        ax2.set_title('Termograma: CLIQUE NO CENTRO DA VULVA')

        plt.tight_layout()

        print(f"\n[{id_str}] Aguardando clique para a foto {nome_npy}...")

        coordenadas = plt.ginput(1, timeout=0)
        plt.close(fig)

        # 6. Salva as coordenadas NO NOVO ARQUIVO
        if coordenadas:
            x_click, y_click = coordenadas[0]
            x_int, y_int = int(round(x_click)), int(round(y_click))

            df.at[idx, 'Coord_X'] = x_int
            df.at[idx, 'Coord_Y'] = y_int

            # SOBRESCREVE APENAS O ARQUIVO 'planilha_anotada.CSV'
            df.to_csv(arquivo_saida, sep=';', index=False)

            anotadas += 1
            print(f"Salvo com segurança em {arquivo_saida.name}! -> X: {x_int}, Y: {y_int}. Faltam: {total_validas - anotadas}")
        else:
            print("Janela fechada sem marcação. Encerrando o anotador para continuar depois.")
            break

    print("\n--- Processo finalizado ou interrompido ---")

if __name__ == "__main__":
    anotar_com_backup()