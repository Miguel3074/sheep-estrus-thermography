# Decisão metodológica para a ROI vulvar

Data da decisão: 30 de julho de 2026.

## Decisão

Para a base atual, a análise principal usará uma janela quadrada fixa de
`11 x 11` pixels centrada no ponto de interesse (POI) anotado manualmente.
Janelas concêntricas de `7 x 7` e `15 x 15` serão usadas como análise de
sensibilidade. A temperatura do POI e um crescimento de região limitado serão
mantidos como métodos de comparação.

Essa escolha é a alternativa mais reproduzível com os dados disponíveis; ela
não equivale a afirmar que o quadrado representa perfeitamente os limites
anatômicos da vulva. Uma segmentação anatômica só poderá ser validada quando
existirem máscaras manuais de referência.

## Evidência considerada

### Evidência externa

- Em ovelhas, a temperatura superficial da vulva varia durante o ciclo estral,
  sustentando a escolha dessa estrutura como região de interesse
  [de Freitas et al., 2018](https://doi.org/10.1016/j.theriogenology.2018.07.015).
- Em estudo de estro com leitoas, foram analisados mínimo, máximo, média e
  desvio-padrão da temperatura vulvar; a temperatura ambiente se correlacionou
  com as medidas térmicas, justificando a extração de estatísticas da ROI e de
  gradientes em relação ao ambiente
  [Sykes et al., 2012](https://doi.org/10.1016/j.theriogenology.2012.01.030).
- A repetibilidade da termografia depende da região anatômica e do parâmetro
  térmico, o que exige um procedimento de aquisição e medição padronizado
  [Byrne et al., 2017](https://doi.org/10.2527/jas.2016.1005).
- Um estudo experimental de confiabilidade em caprinos destaca que tamanho,
  forma e posição consistentes da ROI afetam especialmente a repetibilidade
  das médias; também recomenda avaliar média e máximo em vez de depender de
  uma única medida
  [Bhattacharjee et al., 2026](https://doi.org/10.7717/peerj.20861).
- O crescimento de região semeado depende da semente e de um critério de
  homogeneidade. Esse critério não cria, por si só, uma fronteira anatômica
  [Adams e Bischof, 1994](https://doi.org/10.1109/34.295913).

### Evidência da própria base

A matriz radiométrica possui `80 x 60 = 4.800` pixels. Foram auditados 829
registros com fotografia:

| Situação | Registros |
|---|---:|
| Fotografias referenciadas | 829 |
| Pares de coordenadas preenchidos | 781 |
| POIs válidos para a ROI principal `11 x 11` | 767 |
| Coordenadas ausentes | 48 |
| Coordenadas `(0,0)` sentinela | 12 |
| POIs inválidos por proximidade da borda | 2 |
| Total ainda sem ROI principal | 62 |

No crescimento de região com raio máximo de 5 pixels, referência igual à
mediana local `3 x 3`, tolerância de `±0,3 °C` e conectividade de oito
vizinhos, 720 de 767 máscaras (93,9%) tocaram o limite da área de busca. Esse
resultado mostra que, na maior parte dos termogramas, o método só parou por
causa do limite espacial e não por encontrar uma borda térmica estável.

Por isso, o crescimento de região não será chamado de segmentação anatômica.
Ele permanece no experimento para verificar se a seleção de pixels
termicamente homogêneos acrescenta sinal preditivo.

## Especificação da extração

As coordenadas são interpretadas como `(x, y)`, enquanto a matriz NumPy é
indexada como `[y, x]`. Para uma janela ímpar de lado `s`, o raio é
`r = (s - 1) / 2`, e a ROI contém:

```text
y - r ... y + r
x - r ... x + r
```

As três escalas correspondem a:

| Janela | Pixels | Fração da matriz | Papel |
|---|---:|---:|---|
| `7 x 7` | 49 | 1,02% | sensibilidade, região mais central |
| `11 x 11` | 121 | 2,52% | análise principal |
| `15 x 15` | 225 | 4,69% | sensibilidade à inclusão periférica |

A janela `11 x 11` foi pré-especificada como compromisso entre reduzir a
instabilidade de um único pixel e evitar a inclusão excessiva de tecido
periférico observada nas sobreposições de controle. A escala `15 x 15` não
cabe em dois dos 767 POIs válidos para a análise principal; nesses dois
registros, seus atributos ficam ausentes, sem excluir as ROIs `7 x 7` e
`11 x 11`.

Para cada janela são extraídos:

- número de pixels;
- mínimo, média, mediana, percentil 90 e máximo;
- desvio-padrão;
- média menos temperatura ambiente;
- p90 menos temperatura ambiente;
- máximo menos temperatura ambiente.

O mínimo é preservado para auditoria, mas deve ser interpretado com cautela,
pois pixels frios isolados podem representar umidade, sujeira ou erro de
posicionamento. Média, mediana e p90 são as estatísticas principais; o máximo
é mantido para comparação com a prática de usar o ponto mais quente.

## Métodos comparados

| Identificador | Descrição | Uso |
|---|---|---|
| `poi_temperature` | pixel exatamente no POI | baseline radiométrico |
| leitura do termograma na planilha | valor registrado durante a coleta | baseline de campo |
| `fixed_square_7` | janela `7 x 7` | sensibilidade |
| `fixed_square_11` | janela `11 x 11` | principal |
| `fixed_square_15` | janela `15 x 15` | sensibilidade |
| `region_growing_r5_t0.3` | região conectada, raio 5 e `±0,3 °C` | comparação exploratória |

Não se deve escolher retrospectivamente a melhor ROI usando o mesmo conjunto
de teste empregado para estimar o desempenho final. A análise principal será
`11 x 11`; as demais configurações serão relatadas como sensibilidade. Se
houver seleção de método ou hiperparâmetro, ela deve ocorrer apenas nos dados
de treino, preferencialmente em validação aninhada.

## Controle de qualidade e exclusões

Um registro é recusado para a análise principal quando:

1. a fotografia ou as coordenadas estão ausentes;
2. o POI é `(0,0)`, usado como sentinela de anotação ausente;
3. o POI está fora da matriz;
4. a janela principal `11 x 11` ultrapassa a borda;
5. a matriz não possui formato `60 x 80`;
6. o arquivo `.npy` correspondente não existe.

O script produz imagens de revisão distribuídas ao longo da base. As cores
usadas são verde para `7 x 7`, ciano para `11 x 11`, branco para `15 x 15`,
magenta para crescimento de região e cruz branca para o POI.

## Reprodução

```powershell
python src/extrair_roi_multiescala.py
python -m unittest discover -s tests -v
```

Arquivos de saída:

- `outputs/roi_multiescala/roi_features_comparative.csv`;
- `outputs/roi_multiescala/roi_method_summary.csv`;
- `outputs/roi_multiescala/roi_errors.csv`;
- `outputs/roi_multiescala/revisao/`.

## Resultado preliminar e próxima decisão científica

A primeira validação agrupada por animal foi executada na coorte comum de 765
registros. A ROI `11 x 11` com regressão logística obteve PR-AUC média de
0,192 e ROC-AUC média de 0,620. A escala `15 x 15` apresentou resultado
exploratório um pouco maior na regressão logística, mas a diferença não foi
consistente entre modelos. Por isso, `11 x 11` permanece como análise principal
pré-especificada e `15 x 15` como sensibilidade.

Os resultados completos e as ressalvas estão em
[`resultados_preliminares_modelagem.md`](resultados_preliminares_modelagem.md).

Depois da revisão dos 61 POIs disponíveis e da recuperação de `FLIR1107`, o
protocolo deve ser repetido sem alterar a especificação com base no desempenho
observado. Se forem produzidas máscaras anatômicas manuais, uma futura
segmentação automática poderá ser avaliada com Dice/IoU antes de ser usada na
classificação.
