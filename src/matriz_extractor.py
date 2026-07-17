import flirimageextractor
import numpy as np
from pathlib import Path
from tqdm import tqdm # Biblioteca para mostrar barra de progresso (instale com: pip install tqdm)

def extrair_matrizes_termicas():
    # Define o diretório raiz onde estão os dados
    pasta_data = Path('../data')
    lotes = ['Lote Rosa', 'Lote Vermelho']

    # Inicializa o extrator da FLIR apenas uma vez para economizar memória
    flir = flirimageextractor.FlirImageExtractor()

    imagens_processadas = 0
    erros = []

    todas_imagens = []
    for lote in lotes:
        pasta_lote = pasta_data / lote
        if pasta_lote.exists():
            todas_imagens.extend(list(pasta_lote.rglob('*.jpg')))
        else:
            print(f"Aviso: Pasta {pasta_lote} não encontrada.")

    print(f"Total de imagens encontradas: {len(todas_imagens)}")

    for caminho_img in tqdm(todas_imagens, desc="Extraindo matrizes térmicas"):
        try:
            flir.process_image(str(caminho_img))

            matriz_celsius = flir.get_thermal_np()

            caminho_saida = caminho_img.with_suffix('.npy')

            np.save(caminho_saida, matriz_celsius)

            imagens_processadas += 1

        except Exception as e:
            erros.append((caminho_img.name, str(e)))

    print("\n--- Concluído ---")
    print(f"Matrizes salvas com sucesso: {imagens_processadas}")
    if erros:
        print(f"Erros encontrados: {len(erros)}")
        for arquivo, erro in erros[:5]:
            print(f" - {arquivo}: {erro}")

if __name__ == "__main__":
    extrair_matrizes_termicas()