# Resultados preliminares da modelagem agrupada

Data da execução: 30 de julho de 2026.

## Escopo

Esta análise verifica se as representações térmicas atuais contêm sinal
generalizável para animais não vistos. Ela não constitui ainda o resultado
final do TCC porque:

1. 62 fotografias ainda não possuem ROI principal revisada;
2. a interpretação de célula vazia em `Monta` como classe negativa precisa ser
   confirmada pelos responsáveis pela coleta;
3. não existe um conjunto externo independente;
4. a temperatura ambiente e a data de coleta podem refletir o protocolo de
   sincronização, criando confundimento temporal.

## Recuperação dos dados

O conjunto recebido possuía 850 matrizes `.npy`. Foram localizados JPEGs
radiométricos para 25 dos 26 registros referenciados que estavam sem matriz.
As 25 matrizes foram reconstruídas, verificadas como `60 x 80` e acompanhadas
de visualizações PNG, sem sobrescrever arquivos existentes.

Situação atual:

| Item | Quantidade |
|---|---:|
| Matrizes `.npy` locais | 875 |
| Visualizações PNG | 875 |
| Fotografias referenciadas na planilha | 829 |
| Referências com matriz disponível | 828 |
| Arquivo ainda indisponível | 1 (`FLIR1107`) |
| POIs pendentes, mas disponíveis para marcação | 61 |

## Coorte modelada

A modelagem foi mantida na coorte congelada anterior à nova marcação manual:

| Item | Quantidade |
|---|---:|
| Registros com ROI principal | 767 |
| Coorte comum a todas as escalas | 765 |
| Registros positivos para `Monta` | 65 |
| Registros negativos provisórios | 700 |
| Prevalência positiva | 8,5% |
| Animais | 36 |
| Animais com ao menos um positivo | 24 |

Os dois registros excluídos da coorte comum possuem ROI `11 x 11`, mas não
comportam a janela `15 x 15`. Usar a coorte comum mantém idênticas as amostras
comparadas.

## Validação

Foram realizadas cinco repetições de validação cruzada com cinco folds,
totalizando 25 divisões. Utilizou-se `StratifiedGroupKFold`, que tenta preservar
a proporção das classes sem permitir que o mesmo grupo apareça no treino e no
teste ([documentação do scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)).

O grupo foi o identificador da ovelha. As 25 divisões foram verificadas:

- nenhuma contém o mesmo animal no treino e no teste;
- todos os conjuntos de teste possuem exemplos de estro;
- cada teste contém entre 10 e 16 registros positivos.

Foram avaliados regressão logística balanceada, SVM RBF balanceada e
`Random Forest` balanceada. Imputação pela mediana e padronização foram
ajustadas somente com os dados de treino de cada fold.

## Representações

- somente temperatura ambiente;
- leitura de ponto registrada na planilha + ambiente;
- pixel radiométrico no POI + ambiente;
- ROI fixa `7 x 7` + ambiente;
- ROI fixa `11 x 11` + ambiente, análise principal;
- ROI fixa `15 x 15` + ambiente;
- crescimento de região + ambiente.

Cada ROI usa média, mediana, p90, máximo e desvio-padrão. O mínimo não entrou
no modelo preliminar por ser mais sensível a pixels frios isolados.

## Resultados

Principais resultados médios nos 25 folds:

| Representação | Modelo | PR-AUC | ROC-AUC | F1 | Sens. | Espec. |
|---|---|---:|---:|---:|---:|---:|
| Ambiente | Logística | 0,144 | 0,630 | 0,226 | 0,593 | 0,664 |
| Ponto de campo | Logística | 0,170 | 0,633 | 0,216 | 0,591 | 0,646 |
| POI radiométrico | Logística | 0,196 | 0,630 | 0,192 | 0,525 | 0,641 |
| ROI `7 x 7` | Logística | 0,193 | 0,619 | 0,207 | 0,569 | 0,643 |
| ROI `11 x 11` | Logística | 0,192 | 0,620 | 0,200 | 0,559 | 0,636 |
| ROI `15 x 15` | Logística | 0,220 | 0,642 | 0,210 | 0,574 | 0,645 |
| Crescimento de região | Logística | 0,184 | 0,629 | 0,186 | 0,517 | 0,635 |
| POI radiométrico | SVM RBF | 0,218 | 0,688 | 0,241 | 0,561 | 0,720 |
| ROI `11 x 11` | SVM RBF | 0,198 | 0,665 | 0,238 | 0,521 | 0,731 |

A análise principal `11 x 11 + regressão logística` apresentou grande
variação entre os folds:

| Métrica | Média | Desvio-padrão | Quantis 2,5%–97,5% dos folds |
|---|---:|---:|---:|
| PR-AUC | 0,192 | 0,076 | 0,077–0,329 |
| ROC-AUC | 0,620 | 0,113 | 0,396–0,775 |
| F1 | 0,200 | 0,058 | 0,097–0,289 |
| Sensibilidade | 0,559 | 0,194 | 0,230–0,818 |
| Especificidade | 0,636 | 0,044 | 0,550–0,709 |

Os quantis acima descrevem a dispersão dos folds e não devem ser chamados de
intervalos de confiança independentes, pois os folds repetidos compartilham
observações.

## Bootstrap pareado por animal

As predições fora da amostra foram primeiro promediadas por registro nas cinco
repetições. Em seguida, 500 amostras bootstrap foram formadas reamostrando
animais inteiros. A diferença é sempre `ROI 11 x 11 - referência`, usando
regressão logística:

| Referência | Métrica | Diferença observada | Faixa bootstrap 2,5%–97,5% |
|---|---|---:|---:|
| Ambiente | PR-AUC | +0,032 | +0,001 a +0,074 |
| Ambiente | ROC-AUC | +0,006 | -0,031 a +0,043 |
| Ponto de campo | PR-AUC | +0,022 | -0,007 a +0,059 |
| POI radiométrico | PR-AUC | +0,010 | -0,021 a +0,040 |
| ROI `7 x 7` | PR-AUC | +0,004 | -0,013 a +0,018 |
| ROI `15 x 15` | PR-AUC | -0,022 | -0,051 a +0,001 |
| Crescimento de região | PR-AUC | +0,012 | -0,017 a +0,041 |

Somente a diferença de PR-AUC contra o modelo ambiental ficou ligeiramente
acima de zero nessa análise. Isso ainda não prova superioridade porque o
bootstrap condiciona-se às predições já produzidas e não incorpora toda a
incerteza de refazer o treinamento.

## Interpretação

Há um sinal acima da prevalência para identificar estro, mas ele é fraco e
instável. A ROI `11 x 11` não superou de forma consistente o POI radiométrico,
a leitura de campo ou todas as escalas. A janela `15 x 15` foi favorável na
regressão logística, porém a vantagem variou conforme o modelo e foi observada
no mesmo experimento usado para comparação; portanto, ela permanece uma
hipótese para validação futura, não uma nova análise principal.

O desempenho do modelo somente ambiental é um alerta importante. Como os
animais foram acompanhados em datas comuns e o ciclo foi sincronizado, a
temperatura ambiente pode funcionar como marcador indireto de data ou etapa do
protocolo. O resultado final deve analisar explicitamente data, lote e
protocolo e, se possível, usar uma coleta independente.

## Reprodução

```powershell
.\venv\Scripts\python.exe src/modelar_estro.py `
  --folds 5 `
  --repeats 5 `
  --rf-estimators 100 `
  --bootstrap-iterations 500
```

Os parâmetros, folds, métricas, predições e comparações bootstrap ficam em
`outputs/modeling_grouped/`.

## Próximos passos

1. confirmar a semântica dos vazios em `Monta`;
2. marcar os 61 POIs disponíveis e obter `FLIR1107`;
3. regenerar todas as ROIs e repetir exatamente o protocolo;
4. avaliar um bloqueio temporal ou uma coleta externa;
5. manter `11 x 11` como análise principal e `15 x 15` como hipótese de
   sensibilidade até existir validação independente.
