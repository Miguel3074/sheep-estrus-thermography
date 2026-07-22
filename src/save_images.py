import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

def gerar_imagens_termicas():
    pasta_origem = Path('../data')
    pasta_destino = Path('../data/Termogramas_Gerados')

    arquivos_npy = list(pasta_origem.rglob('*.npy'))
    print(f"Encontrados {len(arquivos_npy)} arquivos .npy para conversão.")

    for caminho_npy in tqdm(arquivos_npy, desc="Salvando imagens PNG"):
        try:
            matriz_termica = np.load(caminho_npy)

            caminho_relativo = caminho_npy.relative_to(pasta_origem)

            caminho_saida = pasta_destino / caminho_relativo.with_suffix('.png')

            caminho_saida.parent.mkdir(parents=True, exist_ok=True)

            plt.imsave(str(caminho_saida), matriz_termica, cmap='magma')

        except Exception as e:
            print(f"\nErro ao processar {caminho_npy.name}: {e}")

    print("\n--- Concluído ---")
    print(f"As imagens coloridas foram salvas em: {pasta_destino.resolve()}")

if __name__ == "__main__":
    gerar_imagens_termicas()