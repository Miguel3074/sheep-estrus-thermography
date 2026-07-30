# Sheep Estrus Thermography

Pipeline para reconstrução radiométrica, extração de regiões de interesse
(ROIs) da vulva e classificação preliminar do estro em ovelhas.

## Estado da base

A auditoria local encontrou:

- 829 registros da planilha associados a uma fotografia;
- 781 registros com um par de coordenadas do ponto de interesse (POI);
- 767 POIs utilizáveis pela ROI principal;
- 48 registros sem coordenadas e 14 marcações inválidas pendentes de revisão.

Entre as 14 marcações inválidas, 12 usam `(0,0)` como valor sentinela e duas
estão próximas demais da borda da matriz. Portanto, ainda há 62 fotografias
referenciadas na planilha sem uma ROI principal utilizável.

O pacote recebido continha 850 matrizes radiométricas `.npy` e 850
visualizações `.png`. Foram reconstruídas 25 matrizes ausentes a partir dos
JPEGs radiométricos locais e geradas as 25 visualizações correspondentes.
Assim, o conjunto local contém agora 875 matrizes e 875 PNGs. Das 829
fotografias referenciadas pela planilha, 828 possuem matriz; somente
`FLIR1107` não foi encontrada.

As imagens e matrizes não ficam versionadas no Git por causa do tamanho. Os
dados devem ser mantidos fora do repositório ou informados por `--data-dir`.

O estado exato das marcações, a distinção entre POI manual e ROI automática e
o procedimento para concluir as pendências estão registrados em
[`docs/status_anotacoes.md`](docs/status_anotacoes.md).

## Método de ROI adotado

As colunas `Coord_X` e `Coord_Y` de `planilha/planilha_anotada.CSV` representam
o centro da vulva na matriz térmica nativa de `80 x 60` pixels. A estratégia
pré-especificada é:

- método principal: janela fixa de `11 x 11` pixels, centrada no POI;
- sensibilidade espacial: janelas de `7 x 7` e `15 x 15` pixels;
- baselines: temperatura do próprio POI e leitura de ponto da planilha;
- comparação exploratória: crescimento de região com raio máximo de 5 pixels
  e tolerância de `±0,3 °C`.

A janela fixa é o método principal por ter tamanho, forma e posição
reproduzíveis. O crescimento por semente foi mantido como comparação, não como
máscara anatômica validada: 720 das 767 regiões processadas (93,9%) alcançaram
o limite espacial configurado, indicando que a semelhança térmica não produz
uma fronteira natural confiável nesta base.

Para cada escala são extraídos número de pixels, mínimo, média, mediana,
percentil 90, máximo e desvio-padrão, além dos gradientes da média, do p90 e do
máximo em relação à temperatura ambiente. A análise preditiva principal deve
usar `11 x 11`; `7 x 7`, `15 x 15`, POI e crescimento de região servem para
análise de sensibilidade e comparação.

A justificativa completa, as regras de exclusão e as referências estão em
[`docs/metodologia_roi.md`](docs/metodologia_roi.md).

## Execução

Ambiente recomendado:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements-modeling.txt
```

O script procura os dados nesta ordem:

1. caminho informado por `--data-dir`;
2. variável de ambiente `SHEEP_DATA_DIR`;
3. pasta `data/` na raiz do projeto;
4. pasta `../data/data` relativa à raiz do projeto.

Extração recomendada:

```powershell
python src/extrair_roi_multiescala.py
```

Com caminhos explícitos:

```powershell
python src/extrair_roi_multiescala.py `
  --csv planilha/planilha_anotada.CSV `
  --data-dir "C:\caminho\para\data" `
  --output-dir outputs/roi_multiescala
```

Saídas em `outputs/roi_multiescala/`:

- `roi_features_comparative.csv`: atributos das três escalas, POI e
  crescimento de região;
- `roi_method_summary.csv`: cobertura e resumo descritivo de cada método;
- `roi_errors.csv`: POIs ausentes ou inválidos;
- `revisao/`: 24 sobreposições distribuídas pela base para inspeção visual.

A escala `15 x 15` está disponível em 765 registros; dois POIs próximos da
borda continuam válidos nas escalas `7 x 7` e `11 x 11`.

O experimento isolado de crescimento por semente continua disponível em:

```powershell
python src/extrair_roi.py --max-radius 5 --tolerance 0.3
```

## Recuperação e anotação das pendências

Auditoria das matrizes ausentes:

```powershell
.\venv\Scripts\python.exe src/recuperar_matrizes_pendentes.py
```

Para criar somente arquivos ainda inexistentes:

```powershell
.\venv\Scripts\python.exe src/recuperar_matrizes_pendentes.py --apply
.\venv\Scripts\python.exe src/save_images.py
```

O processo não sobrescreve matrizes ou PNGs existentes. Após a recuperação,
61 dos 62 POIs pendentes estão prontos para marcação manual. A lista pode ser
recriada sem abrir interface gráfica:

```powershell
.\venv\Scripts\python.exe src/anotador.py --audit
```

Para marcar os POIs:

```powershell
.\venv\Scripts\python.exe src/anotador.py --limit 20
```

O anotador mostra a imagem visual e a matriz térmica, exige que o clique
comporte a janela principal `11 x 11`, cria backup datado do CSV e salva cada
clique atomicamente. Execute novamente para retomar.

## Modelagem preliminar

O rótulo principal é `Monta`. Provisoriamente, `true` é codificado como estro
e célula vazia como não estro. Essa interpretação deve ser confirmada com os
responsáveis pela coleta antes da análise científica final.

Execução reproduzida:

```powershell
.\venv\Scripts\python.exe src/modelar_estro.py `
  --group-by animal `
  --folds 5 `
  --repeats 5 `
  --rf-estimators 100 `
  --bootstrap-iterations 500
```

A comparação usa uma coorte comum de 765 registros, 65 positivos (8,5%) e 36
animais. Foram executadas 25 divisões estratificadas e agrupadas por animal:
nenhum animal aparece simultaneamente no treino e no teste. A análise
principal foi pré-especificada como ROI `11 x 11` com regressão logística.

Resultados médios nos 25 folds:

| Representação/modelo | PR-AUC | ROC-AUC | F1 | Sensibilidade | Especificidade |
|---|---:|---:|---:|---:|---:|
| `11 x 11` + regressão logística, principal | 0,192 | 0,620 | 0,200 | 0,559 | 0,636 |
| Ambiente + regressão logística | 0,144 | 0,630 | 0,226 | 0,593 | 0,664 |
| `15 x 15` + regressão logística | 0,220 | 0,642 | 0,210 | 0,574 | 0,645 |
| POI radiométrico + SVM | 0,218 | 0,688 | 0,241 | 0,561 | 0,720 |

O desempenho ainda é modesto e variável entre folds. A ROI principal aumentou
a PR-AUC em relação ao modelo apenas ambiental, mas não melhorou a ROC-AUC; o
ganho também não foi consistente contra o ponto radiométrico ou todas as
outras escalas. A janela `15 x 15` apresentou um sinal exploratório favorável
na regressão logística, porém não deve substituir retrospectivamente a análise
principal usando o mesmo teste.

Também foi executada uma validação agrupada por data. A auditoria encontrou
137 registros de fevereiro com anos incrementais entre 2026 e 2033. O padrão
dos dias, das fotografias e do lote indica erro de preenchimento; por isso o
pipeline preserva o CSV original e normaliza apenas o ano da cópia analítica
para 2025. Essa hipótese ainda deve ser confirmada com os responsáveis pela
coleta.

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

Após a normalização existem 24 datas. Na validação temporal, a ROI principal
`11 x 11` obteve PR-AUC `0,189` e ROC-AUC `0,603` com regressão logística. A
SVM obteve PR-AUC `0,179` e ROC-AUC `0,625`; a `Random Forest`, PR-AUC `0,148`
e ROC-AUC `0,580`. O POI radiométrico com regressão logística teve a maior
PR-AUC média (`0,212`), mas esse resultado é exploratório.

Quando o bootstrap foi agrupado por data, a diferença da ROI `11 x 11` contra
o modelo somente ambiental foi `+0,008` em PR-AUC, com faixa de
`-0,023` a `+0,048`. Portanto, o ganho térmico não se manteve conclusivo ao
bloquear datas inteiras.

Saídas em `outputs/modeling_grouped/`:

- `summary_metrics.csv`: médias, desvios e quantis dos 25 folds;
- `fold_metrics.csv`: métricas de cada divisão;
- `out_of_fold_predictions.csv`: todas as predições fora da amostra;
- `fold_assignments.csv`: animais de teste em cada divisão;
- `paired_bootstrap_summary.csv`: comparação pareada por reamostragem de
  animais;
- `primary_oof_diagnostics.png`: curvas ROC/PR e matriz de confusão;
- `feature_set_comparison.png`: comparação visual das representações.

As saídas equivalentes da validação temporal ficam em
`outputs/modeling_grouped_date/`, incluindo `date_audit.csv`, que documenta
cada data original e sua versão analítica.

## Testes

```powershell
python -m unittest discover -s tests -v
```

Os 16 testes cobrem centralização e limites das janelas fixas, conectividade e
limite espacial do crescimento por semente, cálculo dos atributos térmicos,
recuperação segura, auditoria das anotações, preparação dos rótulos,
normalização auditável das datas e ausência de vazamento entre datas.

## Próximas etapas

1. marcar manualmente os 61 POIs disponíveis;
2. localizar ou solicitar novamente a fotografia `FLIR1107`;
3. confirmar se célula vazia em `Monta` significa observação negativa;
4. confirmar que os anos 2026–2033 são erros de preenchimento e que o ano
   correto da coleta é 2025;
5. regenerar as ROIs e repetir as duas validações depois da revisão;
6. obter uma coleta independente ou um novo período de coleta;
7. congelar a especificação final antes do teste independente.
