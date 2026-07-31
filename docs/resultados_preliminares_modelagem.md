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

## Avaliação aprofundada da janela de alerta

Foi investigado se o problema fica mais informativo ao considerar positiva
uma fotografia obtida no dia da monta **ou no dia anterior**. Como a planilha
registra apenas a data e não o horário, esse alvo deve ser descrito
corretamente como `dia atual + dia seguinte`, uma aproximação de alerta em
24 horas. Ele não representa uma janela cronometrada de exatamente 24 horas.

Essa análise é fisiologicamente plausível porque a temperatura vulvar pode
permanecer elevada entre o estro, a ovulação e o período pós-ovulatório
([de Freitas et al., 2018](https://doi.org/10.1016/j.theriogenology.2018.07.015)).
Janelas de alerta de 12, 24, 48 e 72 horas também já foram avaliadas em
sistemas automatizados de detecção de estro em bovinos
([Perez Marquez et al., 2022](https://doi.org/10.1016/j.animal.2022.100585)).
Isso fundamenta o experimento, mas não substitui sua validação específica em
ovelhas.

### Protocolo contra otimismo

Foram comparados os alvos `monta no dia` e `monta no dia ou no dia seguinte`
na mesma coorte de 765 registros. O segundo alvo aumentou o número de
positivos de 65 (8,5%) para 101 (13,2%).

O protocolo utilizou três repetições de cinco folds externos. Em cada fold, o
limiar que maximiza F1 foi escolhido somente por validação agrupada interna
com três folds. Foram executados dois bloqueios:

- por animal, sem repetir o mesmo animal entre treino e teste;
- por data, sem repetir a data e removendo também do treino as datas a até um
  dia do teste. Essa purga evita que alvos sobrepostos de dias consecutivos
  produzam vazamento.

Em todos os 15 testes de cada alvo e bloqueio havia exemplos positivos e
negativos. Depois da purga temporal, cada treino conservou de 282 a 413
registros e cada teste, de 141 a 169 registros para o alvo de alerta.

Como as classes são desbalanceadas, acurácia comum não foi usada para escolher
o modelo. F1 é a média harmônica entre precisão e sensibilidade, não entre
acurácia e sensibilidade, e depende do limiar adotado. Por isso ela foi
reportada junto com precisão, sensibilidade e especificidade. A PR-AUC também
foi mantida como métrica principal independente de um único ponto de corte,
pois curvas precisão-revocação são mais informativas que ROC em conjuntos
fortemente desbalanceados
([Saito e Rehmsmeier, 2015](https://doi.org/10.1371/journal.pone.0118432)).

### O que mudou ao incluir o dia seguinte

Comparação justa usando a mesma representação `15 x 15` e a mesma regressão
logística:

| Grupo de teste | Alvo | PR-AUC | ROC-AUC | F1 |
|---|---|---:|---:|---:|
| Animal | Monta no dia | 0,205 | 0,644 | 0,227 |
| Animal | Dia atual + seguinte | 0,279 | 0,653 | 0,310 |
| Data com purga | Monta no dia | 0,182 | 0,583 | 0,144 |
| Data com purga | Dia atual + seguinte | 0,264 | 0,651 | 0,168 |

A PR-AUC de referência de um classificador aleatório sobe de 0,085 para 0,132
quando a prevalência aumenta. Mesmo descontando esse efeito, o excesso de
PR-AUC da ROI `15 x 15` sobre a prevalência passou de `0,120` para `0,147` no
bloqueio por animal e de `0,097` para `0,132` no bloqueio temporal. Portanto,
a melhora não se explica apenas pelo maior número de positivos.

### Comparação de representações e algoritmos

Além do POI e da ROI fixa, foram avaliados:

- elipse de raios 4 e 7 pixels;
- 10% dos pixels mais quentes do quadrado `15 x 15`, combinados com contraste
  entre núcleo e anel local;
- região conectada a `±0,1 °C` da semente, limitada a sete pixels;
- `Extra Trees`, além de regressão logística e SVM.

Resultados selecionados para `dia atual + seguinte`, como média dos folds:

| Grupo | Representação/modelo | PR-AUC | ROC-AUC | F1 | Precisão | Sens. | Espec. |
|---|---|---:|---:|---:|---:|---:|---:|
| Animal | `15 x 15` + Logística | 0,279 | 0,653 | 0,310 | 0,233 | 0,490 | 0,758 |
| Animal | Contraste quente + Logística | 0,282 | 0,663 | 0,310 | 0,241 | 0,476 | 0,772 |
| Animal | POI + SVM | 0,268 | 0,660 | 0,328 | 0,275 | 0,422 | 0,835 |
| Animal | POI + Extra Trees | 0,282 | 0,699 | 0,279 | 0,248 | 0,396 | 0,823 |
| Data | `15 x 15` + Logística | 0,264 | 0,651 | 0,168 | 0,124 | 0,498 | 0,546 |
| Data | Contraste quente + Logística | 0,254 | 0,653 | 0,199 | 0,164 | 0,507 | 0,565 |
| Data | POI + SVM | 0,229 | 0,616 | 0,253 | 0,157 | 0,778 | 0,338 |
| Data | POI + Extra Trees | 0,256 | 0,656 | 0,249 | 0,184 | 0,737 | 0,348 |

O POI com SVM apresentou o maior F1 por animal, mas sua configuração temporal
atingiu esse F1 com 77,8% de sensibilidade e apenas 33,8% de especificidade:
ela geraria muitos falsos alertas. O desvio-padrão do limiar interno da SVM por
data foi `0,635`, evidenciando que o ponto de corte não transfere de forma
estável entre dias. A `Extra Trees` aumentou ROC-AUC em alguns cenários, mas
não melhorou simultaneamente PR-AUC, F1 e estabilidade. Não há evidência de
que trocar apenas o algoritmo resolva o problema.

### Bootstrap pareado das ROIs

As predições fora da amostra foram promediadas por registro nas três
repetições. Em seguida, foram feitas 1.000 reamostragens de animais ou datas
inteiras. As comparações abaixo são `candidata - ROI 15 x 15` na regressão
logística:

| Grupo | Candidata | Métrica | Diferença | Faixa 2,5%–97,5% |
|---|---|---|---:|---:|
| Animal | Contraste quente | PR-AUC | +0,002 | -0,031 a +0,032 |
| Animal | Contraste quente | ROC-AUC | +0,016 | -0,022 a +0,048 |
| Data | Contraste quente | PR-AUC | -0,009 | -0,022 a +0,025 |
| Data | Contraste quente | ROC-AUC | +0,009 | -0,016 a +0,040 |
| Animal | Região conectada | PR-AUC | -0,046 | -0,095 a -0,010 |
| Data | Região conectada | ROC-AUC | -0,033 | -0,069 a -0,005 |

As faixas do contraste térmico incluem zero: ele empatou estatisticamente com
a ROI fixa nesta amostra. Já o crescimento conectado foi inferior em duas
comparações. Isso reforça que sem máscaras anatômicas manuais não é possível
afirmar que um limiar de temperatura segmenta melhor a vulva. Estudos que
medem a vulva por termografia precisam controlar a qualidade de aquisição. Um
estudo recente manteve tamanho e orientação da ROI constantes e excluiu
imagens com borrão, umidade, sujeira ou reflexão
([Pérez-García et al., 2026](https://doi.org/10.1111/avj.70112)).

### Decisão resultante

O alvo de `dia atual + dia seguinte` deve continuar como experimento
exploratório de **alerta**, enquanto `Monta` no próprio dia permanece o alvo
original. A ROI fixa ainda é a referência mais defensável; o contraste quente
pode permanecer como comparação, mas o crescimento por limiar não deve
substituí-la.

O próximo ganho provável não está em testar dezenas de classificadores, e sim
em melhorar o desenho dos dados:

1. obter horário da fotografia e da monta para testar janelas reais de 12, 24
   e 48 horas;
2. confirmar os anos das datas e concluir a revisão dos POIs;
3. desenhar máscaras anatômicas em uma amostra estratificada e então medir
   Dice/IoU de qualquer segmentação automática;
4. registrar ou controlar sujeira, umidade, distância, ângulo e temperatura
   ambiente;
5. congelar modelo e limiar antes de avaliar um novo período ou lote.

## Experimento personalizado e avaliação por evento

Foi investigado se a temperatura deveria ser interpretada em relação ao
próprio animal e ao restante do rebanho, em vez de usar somente o valor
absoluto da ROI. A auditoria mostrou que os 765 registros são 765 combinações
distintas de animal e data; portanto, não existem fotografias repetidas no
mesmo dia que possam ser agregadas para reduzir ruído.

A base possui 36 animais e 24 datas. Os 65 dias com monta formam 44 sequências
quando apenas dias positivos consecutivos são unidos. Se positivos separados
por um único dia negativo forem considerados o mesmo episódio, existem 36
eventos. Doze animais não possuem nenhuma monta positiva. Todos os positivos
ocorrem entre 12 e 28 de fevereiro; de 1º a 6 de março não há positivos. Essa
concentração reforça o risco de confundimento com data, ambiente e etapa do
protocolo.

### Atributos e validação

Foram comparados:

- temperatura ambiente isolada;
- ROI fixa `15 x 15`;
- diferença e percentil da ovelha contra as demais ovelhas fotografadas no
  mesmo dia, usando mediana leave-one-out;
- diferença para a medição anterior e para a mediana dos cinco dias anteriores
  do próprio animal;
- combinação de valores absolutos, relativos ao rebanho e históricos;
- temperatura retal como controle comparativo, não como proposta de método
  não invasivo.

Os atributos históricos usam somente o passado. Os atributos do rebanho usam
apenas temperaturas, nunca os rótulos, e pressupõem que o lote inteiro seja
fotografado antes dos alertas. Regressão logística, SVM RBF e
`HistGradientBoosting` foram avaliados com três repetições de cinco folds. O
limiar de F1 foi escolhido em três folds internos e a validação por data
manteve a purga de um dia.

Resultados selecionados:

| Grupo | Representação/modelo | PR-AUC | ROC-AUC | F1 | Precisão | Sens. | Espec. |
|---|---|---:|---:|---:|---:|---:|---:|
| Animal | `15 x 15` + Logística | 0,279 | 0,653 | 0,310 | 0,233 | 0,490 | 0,758 |
| Animal | Combinado + Logística | 0,284 | 0,686 | 0,313 | 0,227 | 0,556 | 0,722 |
| Animal | Combinado + SVM | 0,280 | 0,679 | 0,323 | 0,240 | 0,545 | 0,737 |
| Data | `15 x 15` + Logística | 0,264 | 0,651 | 0,168 | 0,124 | 0,498 | 0,546 |
| Data | Combinado + Logística | 0,220 | 0,627 | 0,216 | 0,168 | 0,779 | 0,271 |
| Data | Personalizado + Logística | 0,163 | 0,494 | 0,195 | 0,117 | 0,710 | 0,260 |
| Data | Rebanho relativo + Logística | 0,156 | 0,466 | 0,215 | 0,125 | 0,850 | 0,112 |
| Data | Retal + Logística | 0,231 | 0,635 | 0,128 | 0,077 | 0,451 | 0,565 |

O conjunto combinado apresentou um pequeno ganho em animais novos, mas perdeu
PR-AUC e especificidade entre datas. Seu F1 temporal maior foi obtido emitindo
muito mais positivos: a sensibilidade chegou a 77,9%, enquanto a
especificidade caiu para 27,1%.

No bootstrap de 500 reamostragens, a diferença `combinado - 15 x 15` foi:

| Grupo | Métrica | Diferença | Faixa 2,5%–97,5% |
|---|---|---:|---:|
| Animal | PR-AUC | +0,028 | -0,050 a +0,100 |
| Animal | ROC-AUC | +0,041 | -0,027 a +0,092 |
| Data | PR-AUC | -0,034 | -0,079 a +0,036 |
| Data | ROC-AUC | -0,002 | -0,075 a +0,094 |

Todas as faixas incluem zero. Em contraste, a normalização somente pelo
rebanho reduziu a PR-AUC temporal em `-0,104`, com faixa de `-0,160` a
`-0,016`, e os atributos somente personalizados reduziram em `-0,108`, com
faixa de `-0,170` a `-0,007`. Portanto, essas representações isoladas foram
inferiores à ROI fixa nesta base.

O `HistGradientBoosting` não melhorou a ordenação dos positivos entre datas.
Seu F1 mais alto por animal ocorreu usando somente ambiente, evidência de
confundimento e não de sinal anatômico. A temperatura retal também não
apresentou ganho estável sobre a termografia.

### Alertas por evento e ranking diário

Considerando 44 eventos e o limiar escolhido para F1, a ROI `15 x 15` com
regressão logística, validada por data, detectou em média 46,2% dos eventos.
Porém, somente 10,2% dos episódios de alerta correspondiam a um evento e
ocorreram 23,3 falsos alertas por 100 animais-dia. Assim, o limiar atual não é
operacionalmente aceitável.

Também foi avaliada a ordenação das ovelhas dentro de cada dia, que é imune ao
ganho artificial do ambiente comum ao lote. A ROI `15 x 15` com regressão
logística obteve a melhor precisão média diária (`0,290`). Se fossem
inspecionadas as três ovelhas de maior escore por dia, a precisão seria 18,6%
e a cobertura dos registros positivos, 12,9%. Com cinco ovelhas, os valores
seriam 15,8% e 18,2%. O ranking é mais defensável que o alerta binário, mas
ainda recupera poucos positivos.

### Decisão após a personalização

A ROI `15 x 15` com regressão logística continua sendo a melhor referência
exploratória para o alvo `dia atual + dia seguinte`. Os atributos combinados
podem permanecer registrados como análise de sensibilidade, mas não devem
substituí-la: o pequeno ganho por animal não foi confirmado entre datas nem no
bootstrap.

O gargalo agora é a definição e a qualidade do evento. Antes de outra rodada
de algoritmos, é necessário confirmar:

1. se montas em dias próximos pertencem ao mesmo episódio;
2. qual é o primeiro ciclo de interesse após a sincronização;
3. se existem horários de fotografia e monta;
4. se os dias após um evento deveriam continuar na coorte ou ser censurados;
5. qual número de falsos alertas por animal-dia seria aceitável.

## Auditoria EXIF e alvos prospectivos

Uma auditoria posterior recuperou a data e a hora dos JPEGs FLIR. Dos 829
registros com fotografia, 828 JPEGs foram localizados e todos possuem horário
EXIF. Depois de normalizar provisoriamente o ano para 2025, 826 datas
coincidiram com o EXIF. Foram encontradas duas divergências:

- `FLIR1106`: planilha `01.02.2025`, EXIF `01.03.2025 18:25:42`;
- `FLIR0689`: planilha `18.02.2025`, EXIF `19.02.2025 16:49:52`.

A primeira é compatível com erro de mês na planilha. A segunda precisa ser
confirmada, pois o mesmo animal já possui outra fotografia em 19 de fevereiro.
`FLIR1107` continua ausente.

Todos os 765 registros da coorte modelada têm EXIF. Nela, apenas a data de
`FLIR1106` é alterada. O conjunto passa a possuir 23 datas reais de captura.

Como fotografia e teste de monta parecem ter ocorrido na mesma sessão, dois
alvos mais rigorosos foram avaliados:

1. **próximo dia:** a fotografia atual é positiva somente se houver monta no
   dia seguinte;
2. **até 24 h pelo EXIF:** a monta positiva recebe provisoriamente o horário da
   própria fotografia, e somente eventos estritamente futuros em até 24 horas
   são considerados.

O segundo alvo depende da confirmação de que a observação de monta e a
fotografia realmente ocorreram juntas e em ordem comparável.

### Monta somente no dia seguinte

Existem 57 positivos em 765 registros, prevalência de 7,45%. Nas predições
fora da amostra da regressão logística, promediadas entre repetições:

| Grupo | Representação | PR-AUC | Faixa bootstrap | ROC-AUC | Faixa bootstrap |
|---|---|---:|---:|---:|---:|
| Animal | `15 x 15` | 0,125 | 0,081–0,197 | 0,653 | 0,577–0,738 |
| Animal | Contraste quente | 0,129 | 0,078–0,211 | 0,669 | 0,592–0,747 |
| Data | `15 x 15` | 0,133 | 0,075–0,197 | 0,661 | 0,558–0,753 |
| Data | Contraste quente | 0,148 | 0,083–0,226 | 0,674 | 0,581–0,762 |

A PR-AUC aleatória de referência é `0,0745`. Portanto, os atributos térmicos
preservam alguma ordenação prospectiva. Entretanto, o melhor F1 médio foi
`0,203` por animal e `0,118` por data. Ainda não há um limiar operacional
estável.

### Janela cronometrada de 24 horas

Restaram apenas 27 positivos, prevalência de 3,53%. A ROI `15 x 15` com
regressão logística teve:

| Grupo | PR-AUC | Faixa bootstrap | ROC-AUC | Faixa bootstrap |
|---|---:|---:|---:|---:|
| Animal | 0,092 | 0,047–0,191 | 0,709 | 0,605–0,829 |
| Data | 0,065 | 0,026–0,127 | 0,620 | 0,450–0,753 |

O sinal observado em animais novos não foi confirmado entre datas: a faixa
temporal de ROC-AUC inclui `0,5`. A janela exata deve ser mantida como
sensibilidade exploratória, e não como resultado principal.

### Interpretação atualizada

Os resultados apoiam a existência de um sinal térmico prospectivo modesto,
mas não um detector confiável. A próxima coleta deve aumentar o número de
eventos, registrar o horário real da monta e reservar um período ou lote para
validação externa. O plano completo está em
[`roadmap_cientifico.md`](roadmap_cientifico.md).

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

Experimento de `dia atual + dia seguinte`:

```powershell
.\venv\Scripts\python.exe src/avaliar_janela_24h.py
.\venv\Scripts\python.exe src/bootstrap_janela_24h.py
```

As saídas ficam em `outputs/modeling_window_24h/`. `summary_metrics.csv`
contém as médias dos folds, `split_audit.csv` documenta a separação dos dados
e os dois arquivos `bootstrap_*.csv` registram as estimativas reamostradas.

Auditoria EXIF e alvos prospectivos:

```powershell
.\venv\Scripts\python.exe src/auditar_timestamps_exif.py

.\venv\Scripts\python.exe src/avaliar_janela_24h.py `
  --targets mount_next_day mount_within_24h `
  --groupings animal date `
  --feature-sets poi fixed_15 hot_contrast `
  --models logistic svm_rbf `
  --timestamp-audit outputs/exif_timestamp_audit/timestamp_audit_records.csv
```

Experimento personalizado principal:

```powershell
.\venv\Scripts\python.exe src/avaliar_alerta_personalizado.py `
  --models logistic svm_rbf
```

Comparação separada com boosting:

```powershell
.\venv\Scripts\python.exe src/avaliar_alerta_personalizado.py `
  --models hist_gradient_boosting `
  --bootstrap-iterations 0 `
  --output-dir outputs/modeling_personalized_alert_hgb
```

As saídas principais ficam em `outputs/modeling_personalized_alert/`, incluindo
métricas por fold, predições fora da amostra, bootstrap, eventos e ranking
diário. O boosting fica no diretório informado no segundo comando.

## Próximos passos

1. confirmar se o relógio da câmera estava deslocado e se foto/monta ocorreram
   na mesma sessão;
2. confirmar o tratamento de `FLIR0689`, `FLIR1106` e `FLIR1107`;
3. anotar 120 máscaras anatômicas estratificadas, com 30 revisadas por um
   segundo anotador;
4. comparar ROI fixa, limiar térmico e segmentação somente contra essas
   máscaras, sem consultar o rótulo de monta;
5. realizar análise longitudinal alinhada ao evento, com efeito por animal e
   controle de ambiente;
6. manter “monta na próxima coleta” como alvo preditivo primário e a janela
   EXIF de 24 h como sensibilidade;
7. congelar atributos, algoritmo e limiar antes de uma nova coleta;
8. coletar confirmação hormonal/ultrassonográfica em uma subcoorte;
9. validar externamente em outro período ou lote e reportar calibração,
   desempenho por evento e falsos alertas por 100 animal-dias.
