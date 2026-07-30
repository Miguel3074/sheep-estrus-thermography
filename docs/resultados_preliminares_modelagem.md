# Resultados preliminares da modelagem agrupada

Data da execução: 30 de julho de 2026.

## Escopo

Esta análise verifica se as representações térmicas atuais contêm sinal
generalizável para animais não vistos. Ela não constitui ainda o resultado
final do TCC porque:

1. 62 fotografias ainda não possuem ROI principal revisada;
2. não existe um conjunto externo independente;
3. a temperatura ambiente e a data de coleta podem refletir o protocolo de
   sincronização, criando confundimento temporal.

Em 30 de julho de 2026, o responsável confirmou que célula vazia em `Monta`
significa ausência de monta observada. Portanto, a codificação negativa deixou
de ser provisória, embora a qualidade da observação de campo continue sendo
uma limitação normal do rótulo.

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
| Registros negativos confirmados pela regra de coleta | 700 |
| Prevalência positiva | 8,5% |
| Animais | 36 |
| Animais com ao menos um positivo | 24 |

Os dois registros excluídos da coorte comum possuem ROI `11 x 11`, mas não
comportam a janela `15 x 15`. Usar a coorte comum mantém idênticas as amostras
comparadas.

## Auditoria das datas

Foram encontrados 137 registros com anos entre 2026 e 2033. Eles aparecem em
sequência nos dias 17 e 21–28 de fevereiro, dentro do mesmo lote e entre
fotografias consecutivas da coleta de 2025. O padrão é compatível com
incremento automático ou erro de preenchimento do ano.

O CSV original não foi modificado. Para a análise temporal, o script cria:

- `collection_date_original`, com o texto recebido;
- `collection_date`, preservando dia e mês e usando provisoriamente o ano
  informado por `--collection-year 2025`;
- `date_year_corrected`, indicador explícito da correção;
- `date_audit.csv`, com contagens por data original e analítica.

Depois da normalização existem 24 datas de coleta. A atribuição de 2025 deve
ser confirmada com os responsáveis pela coleta antes da análise final.

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

Uma segunda análise repetiu o protocolo usando a data normalizada como grupo.
Nesse caso, nenhuma data aparece simultaneamente no treino e no teste. Os
animais podem aparecer nos dois conjuntos, pois quase todos foram acompanhados
em várias datas. Assim, as duas análises respondem perguntas diferentes:

- agrupamento por animal: desempenho em animais ainda não vistos;
- agrupamento por data: desempenho em um novo dia para animais potencialmente
  já conhecidos.

Com esta coleta não é possível bloquear simultaneamente todos os animais e
todas as datas sem perder praticamente toda a amostra conectada. Uma coleta
externa continua sendo necessária para testar as duas generalizações ao mesmo
tempo.

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

## Validação agrupada por data

Resultados médios nos 25 folds temporais:

| Representação | Modelo | PR-AUC | ROC-AUC | F1 | Sens. | Espec. |
|---|---|---:|---:|---:|---:|---:|
| Ambiente | Logística | 0,138 | 0,621 | 0,231 | 0,577 | 0,663 |
| POI radiométrico | Logística | 0,212 | 0,618 | 0,191 | 0,541 | 0,630 |
| ROI `11 x 11` | Logística | 0,189 | 0,603 | 0,203 | 0,570 | 0,634 |
| ROI `11 x 11` | Random Forest | 0,148 | 0,580 | 0,076 | 0,064 | 0,973 |
| ROI `11 x 11` | SVM RBF | 0,179 | 0,625 | 0,203 | 0,479 | 0,694 |
| ROI `15 x 15` | Logística | 0,187 | 0,612 | 0,204 | 0,536 | 0,653 |
| ROI `15 x 15` | SVM RBF | 0,179 | 0,635 | 0,217 | 0,504 | 0,700 |

A regressão logística com POI radiométrico apresentou a maior PR-AUC média
temporal, enquanto a SVM com ROI `15 x 15` apresentou a maior ROC-AUC. Como
essas combinações foram identificadas no mesmo conjunto usado para comparação,
elas são hipóteses exploratórias, não modelos finais selecionados.

Na ROI principal `11 x 11`, a SVM foi melhor que a regressão logística em
ROC-AUC na validação por animal (`0,665` contra `0,620`), mas a diferença caiu
na validação por data (`0,625` contra `0,603`). A Random Forest apresentou
sensibilidade muito baixa e não é recomendada como candidata principal nesta
configuração.

No bootstrap de 500 reamostragens por data, a diferença da ROI `11 x 11`
contra o modelo ambiental foi:

| Métrica | Diferença observada | Faixa bootstrap 2,5%–97,5% |
|---|---:|---:|
| PR-AUC | +0,008 | -0,023 a +0,048 |
| ROC-AUC | -0,027 | -0,099 a +0,027 |

As duas faixas incluem zero. Portanto, quando datas inteiras são bloqueadas, a
ROI principal ainda não demonstra ganho robusto sobre a temperatura ambiente.

## Experimento temporal exploratório

Estudos com medições repetidas indicam que a temperatura superficial da vulva
varia entre as fases do ciclo e tende a aumentar entre o estro, a ovulação e o
pós-ovulatório
([de Freitas et al., 2018](https://doi.org/10.1016/j.theriogenology.2018.07.015);
[de Freitas et al., 2018b](https://doi.org/10.1016/j.livsci.2018.07.014)).
Por isso foi testada uma representação longitudinal, sem substituir a análise
principal pré-especificada.

Para cada animal e variável térmica da ROI `11 x 11`, o script calcula:

- diferença para a medição anterior;
- diferença dividida pelo intervalo em dias;
- diferença para a mediana das três medições anteriores;
- intervalo em dias e indicador de disponibilidade do histórico.

Também são calculadas as mudanças da temperatura ambiente. O processamento é
feito em ordem cronológica, nunca utiliza `Monta` e não consulta medições
futuras. Dos 765 registros, 729 possuem uma medição anterior. Os 36 primeiros
registros, um por animal, ficam com atributos temporais ausentes; a imputação
pela mediana é ajustada somente no conjunto de treino de cada fold.

Resultados médios nos 25 folds:

| Grupo de teste | Representação/modelo | PR-AUC | ROC-AUC | F1 | Acurácia balanceada |
|---|---|---:|---:|---:|---:|
| Animal | ROI `11 x 11` + Logística | 0,192 | 0,620 | 0,200 | 0,598 |
| Animal | ROI + histórico + Logística | 0,174 | 0,667 | 0,238 | 0,643 |
| Animal | ROI `11 x 11` + SVM | 0,198 | 0,665 | 0,238 | 0,626 |
| Animal | ROI + histórico + SVM | 0,171 | 0,690 | 0,219 | 0,625 |
| Data | ROI `11 x 11` + Logística | 0,189 | 0,603 | 0,203 | 0,602 |
| Data | ROI + histórico + Logística | 0,172 | 0,661 | 0,247 | 0,650 |
| Data | ROI `11 x 11` + SVM | 0,179 | 0,625 | 0,203 | 0,587 |
| Data | ROI + histórico + SVM | 0,183 | 0,630 | 0,203 | 0,609 |

A regressão logística temporal melhorou ROC-AUC, F1 e acurácia balanceada nos
dois bloqueios, mas reduziu a PR-AUC média dos folds. A SVM teve ganho menor de
ROC e não melhorou simultaneamente F1 e PR-AUC. A representação contendo
somente mudanças, sem as temperaturas atuais, ficou próxima ou abaixo do
acaso.

Para a regressão logística, o ganho observado de ROC-AUC da representação
temporal foi `+0,056` por animal, com faixa bootstrap de `-0,010` a `+0,124`,
e `+0,077` por data, com faixa de `-0,018` a `+0,216`. Como ambas incluem
zero, o ganho ainda é exploratório.

A `Random Forest` não melhorou com o histórico. Na validação por data, sua
sensibilidade temporal foi zero, apesar da alta acurácia bruta causada pelo
predomínio da classe negativa. Ela não deve ser selecionada como modelo
principal.

## Reprodução

Validação por animal:

```powershell
.\venv\Scripts\python.exe src/modelar_estro.py `
  --group-by animal `
  --folds 5 `
  --repeats 5 `
  --rf-estimators 100 `
  --bootstrap-iterations 500
```

Validação por data:

```powershell
.\venv\Scripts\python.exe src/modelar_estro.py `
  --group-by date `
  --collection-year 2025 `
  --folds 5 `
  --repeats 5 `
  --rf-estimators 100 `
  --bootstrap-iterations 500 `
  --output-dir outputs/modeling_grouped_date
```

Os parâmetros, folds, métricas, predições, auditoria das datas e comparações
bootstrap ficam nos respectivos diretórios de saída.

## Próximos passos

1. marcar os 61 POIs disponíveis e obter `FLIR1107`;
2. confirmar o ano verdadeiro dos 137 registros sinalizados;
3. regenerar todas as ROIs e repetir exatamente os dois bloqueios;
4. obter uma coleta externa;
5. manter `11 x 11` como análise principal e `15 x 15` como hipótese de
   sensibilidade até existir validação independente.
