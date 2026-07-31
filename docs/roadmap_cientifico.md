# Roadmap científico para detecção de estro por termografia

## Resposta objetiva

O projeto está no estágio de **viabilidade retrospectiva com validação
interna**. A reconstrução radiométrica, a anotação dos pontos, a extração de
atributos e os primeiros testes estão funcionais. Há sinal térmico acima do
acaso em alguns cenários, inclusive prospectivos, mas ainda não há um modelo
pronto para uso no campo.

Para chegar a conclusões científicas fortes, o trabalho precisa responder duas
perguntas diferentes:

1. **Associação fisiológica:** a temperatura vulvar muda antes ou durante o
   estro depois de controlar ambiente, indivíduo e dia de coleta?
2. **Predição operacional:** uma imagem obtida agora consegue alertar uma
   monta futura com sensibilidade útil e quantidade aceitável de falsos
   alertas em outro período ou lote?

Uma resposta positiva à primeira pergunta já produz um TCC científico válido.
A segunda exige validação externa e um desfecho biológico mais forte.

## O que a base permite afirmar hoje

### Auditoria temporal

Os JPEGs FLIR não contêm apenas a visualização exibida na câmera. Eles guardam
a matriz radiométrica, os parâmetros de conversão, uma imagem visível
incorporada e data/hora EXIF.

A auditoria reproduzível de 829 registros com fotografia encontrou:

- 828 JPEGs radiométricos resolvidos e um ausente (`FLIR1107`);
- data e hora EXIF disponíveis nos 828 JPEGs;
- 826 das 828 datas EXIF, ou 99,76%, iguais à data analítica com ano 2025;
- um erro claro de mês: `FLIR1106`, registrado como `01.02.2025`, foi criado
  em `01.03.2025 18:25:42`;
- uma divergência ainda não resolvida: `FLIR0689`, atribuída a `18.02.2025`,
  foi criada em `19.02.2025 16:49:52`;
- 23 datas reais de captura entre 12 de fevereiro e 6 de março de 2025.

Na coorte modelada, os 765 registros possuem EXIF. A correção pela data EXIF
muda somente `FLIR1106`.

Os horários absolutos concentram-se principalmente entre 16 h e 18 h, apesar
da lembrança de que a coleta ocorreu pela manhã. Antes de interpretar hora do
dia, deve-se confirmar se o relógio da câmera estava deslocado. O deslocamento
constante não invalida a ordem nem os intervalos entre fotografias.

### Auditoria radiométrica

Os 875 JPEGs radiométricos locais foram produzidos com a câmera
`FLIR C2 Education k`. Os metadados são consistentes quanto a:

- matriz térmica: `80 x 60`;
- imagem visível incorporada: `640 x 480`;
- emissividade: `0,98`;
- distância configurada: `0,50 m`;
- umidade relativa configurada: `50%`;
- temperatura atmosférica configurada: `20 °C`;
- transmissão da janela IR: `1,0`.

Emissividade e distância são plausíveis para o protocolo relatado. Entretanto,
umidade e temperatura atmosférica ficaram constantes em todas as imagens, o
que sugere valores padrão e não medições ambientais reais. A temperatura
aparente refletida variou entre 28 e 33 °C, mas nem sempre coincide com a
temperatura ambiente da planilha. Em uma distância curta de 0,50 m o efeito
atmosférico tende a ser menor, porém essa incerteza deve ser declarada e
testada em análise de sensibilidade.

A especificação oficial da câmera confirma sensor IR de `80 x 60`, imagem
visual/MSX, dados radiométricos no JPEG, NETD de 100 mK e acurácia nominal de
`±2 °C` ou 2% em condições especificadas. Isso significa que diferenças
absolutas pequenas não devem ser interpretadas isoladamente como precisão
fisiológica; comparações relativas, repetidas no mesmo animal e com protocolo
padronizado são mais defensáveis
([FLIR C2 Educational Kit](https://support.flir.com/DSDownload/Assets/T810226-en-US_A4-lr.pdf)).

### Resultados preditivos

Há 765 registros utilizáveis, 36 animais e forte desbalanceamento. Os três
alvos respondem perguntas distintas:

| Alvo | Positivos | Prevalência | Interpretação |
|---|---:|---:|---|
| Monta no próprio dia | 65 | 8,5% | classificação simultânea |
| Monta no dia atual ou seguinte | 101 | 13,2% | triagem ampla; não é totalmente prospectiva |
| Monta somente no dia seguinte | 57 | 7,45% | alerta prospectivo por próxima coleta |
| Monta em até 24 h pelo EXIF | 27 | 3,53% | sensibilidade; depende da hipótese de mesmo horário da observação |

No alvo prospectivo **somente no dia seguinte**, as predições fora da amostra
da regressão logística produziram:

| Bloqueio | Representação | PR-AUC | Faixa bootstrap | ROC-AUC | Faixa bootstrap |
|---|---|---:|---:|---:|---:|
| Animal | ROI `15 x 15` | 0,125 | 0,081–0,197 | 0,653 | 0,577–0,738 |
| Animal | Contraste quente | 0,129 | 0,078–0,211 | 0,669 | 0,592–0,747 |
| Data | ROI `15 x 15` | 0,133 | 0,075–0,197 | 0,661 | 0,558–0,753 |
| Data | Contraste quente | 0,148 | 0,083–0,226 | 0,674 | 0,581–0,762 |

A referência aleatória de PR-AUC é a prevalência, `0,0745`. Portanto, existe
alguma capacidade de ordenação prospectiva. Contudo, o melhor F1 médio ficou
em `0,203` por animal e `0,118` por data, com limiares instáveis.

Na janela EXIF de 24 h, a ROI fixa obteve ROC-AUC agregada `0,709`
(`0,605–0,829`) quando animais inteiros foram separados. Ao separar datas,
caiu para `0,620` (`0,450–0,753`). Com apenas 27 positivos, o resultado não se
replica de forma estável entre dias e deve permanecer uma sensibilidade
exploratória.

Na avaliação operacional anterior do alvo dia atual + seguinte, a ROI fixa
detectou aproximadamente 46,2% dos episódios com somente 10,2% de precisão
dos alertas e 23,3 falsos alertas por 100 animal-dias no bloqueio por data.
Assim, discriminação moderada não se converteu em um sistema útil no limiar
testado.

### Conclusão permitida pela base atual

> Os atributos térmicos vulvares contêm um sinal prospectivo modesto associado
> à ocorrência de monta na coleta seguinte, mas o pequeno número de eventos, a
> instabilidade entre datas, a ausência do horário exato da monta e a falta de
> validação externa impedem afirmar que a termografia isolada seja um detector
> confiável de estro.

Essa é uma conclusão científica útil. Um resultado negativo ou limitado,
quando obtido com avaliação correta, é melhor que uma acurácia alta produzida
por desbalanceamento ou vazamento.

## O que impede uma conclusão mais forte

1. **Desfecho imperfeito.** Monta observada é um marcador comportamental. A
   confirmação fisiológica ideal usa progesterona e/ou ultrassonografia. O
   estudo clássico em 20 ovelhas sincronizadas mediu duas vezes ao dia,
   acompanhou dinâmica folicular por ultrassom e registrou temperatura,
   umidade e WBGT
   ([de Freitas et al., 2018](https://doi.org/10.1016/j.theriogenology.2018.07.015)).
2. **Horário da monta ausente.** O horário da foto foi recuperado, mas o
   evento ainda é aproximado pelo horário da imagem positiva. Deve-se confirmar
   que foto e observação ocorreram na mesma sessão e registrar a ordem.
3. **Poucos eventos efetivos.** Há 65 dias positivos, mas aproximadamente 44
   episódios quando positivos consecutivos são unidos. O alvo prospectivo
   tem 57 positivos e o alvo cronometrado, somente 27.
4. **ROI não validada anatomicamente.** O POI é manual, mas a janela quadrada e
   o limiar térmico não são máscaras verdadeiras da vulva. O crescimento por
   região alcançou o limite espacial em 93,9% das imagens e não superou a
   janela fixa.
5. **Confusão por data/protocolo.** A temperatura ambiente sozinha já prevê
   parte do alvo. Como os animais foram sincronizados e observados nos mesmos
   dias, ambiente e data podem funcionar como marcadores indiretos do
   protocolo.
6. **Sem teste externo.** Todos os modelos foram avaliados dentro da mesma
   coleta. Separar animais e datas reduz otimismo, mas não substitui um novo
   período, lote ou fazenda.

## Tecnologias que valem a pena

### 1. Segmentação assistida, não segmentação por limiar

Os JPEGs possuem uma imagem térmica renderizada de `320 x 240` e uma imagem
visível incorporada de `640 x 480`. A imagem visível observada é escura em
parte da base; por isso, a imagem térmica renderizada deve continuar como
referência principal para a anotação anatômica.

O `SAM 3/3.1` pode acelerar a criação de máscaras usando ponto, caixa ou
exemplo visual, mas não deve ser considerado verdade anatômica automática.
Ele foi criado para segmentação orientada por prompts em imagens e vídeos e
não foi validado para vulvas ovinas em termogramas
([Meta SAM 3](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/)).

Protocolo recomendado:

1. selecionar 120 imagens estratificadas por lote, animal, data, classe,
   temperatura e qualidade;
2. desenhar a máscara anatômica manual, usando SAM apenas como ferramenta de
   contorno;
3. repetir 30 imagens com um segundo anotador, cego ao rótulo de monta;
4. medir Dice, IoU, distância de borda e concordância térmica;
5. comparar janela fixa, crescimento térmico, SAM assistido e um modelo
   supervisionado;
6. só então extrair atributos da máscara vencedora.

Uma U-Net pequena com aumento de dados é uma candidata depois que as máscaras
existirem; a arquitetura foi concebida para segmentação com poucas imagens
anotadas, mas ainda precisa de teste separado por animal e data
([Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)).

### 2. Modelo longitudinal e inferencial

Antes de redes profundas, o estudo deve modelar a trajetória térmica:

- centralizar a temperatura pela mediana anterior do próprio animal;
- calcular contraste contra região perivulvar e ambiente;
- alinhar observações pelo tempo até a monta;
- estimar curvas de `-72 h` a `+48 h`;
- usar modelo de efeitos mistos ou GEE com animal como agrupamento;
- testar interação entre tempo até o evento, lote e ambiente;
- apresentar tamanho de efeito e intervalo de confiança, não apenas
  classificação.

Essa análise responde à associação fisiológica e é mais compatível com 36
animais do que uma CNN com milhares de parâmetros.

### 3. Predição de baixa complexidade

O modelo principal deve permanecer uma regressão logística penalizada com
poucos atributos pré-especificados. SVM, Extra Trees e LightGBM podem ser
comparações secundárias, sempre dentro da validação agrupada aninhada.

Como há poucos eventos, testar dezenas de algoritmos e hiperparâmetros aumenta
o risco de selecionar ruído. O tamanho de uma próxima coleta deve ser
calculado para a precisão desejada de calibração, discriminação e utilidade,
em vez de usar somente uma regra fixa de “eventos por variável”
([Riley et al., 2020](https://doi.org/10.1136/bmj.m441);
[Riley et al., 2021](https://doi.org/10.1002/sim.9025)).

### 4. Fusão multimodal

O maior ganho provável está em adicionar fontes complementares:

- temperatura e umidade ambientais medidas no momento;
- temperatura retal;
- região ocular bilateral;
- movimento de cauda/exposição da vulva;
- acelerômetro ou vídeo comportamental;
- protocolo hormonal e tempo desde retirada do dispositivo.

Um sistema automatizado em bovinos combinou termografia e movimento da cauda
e usou progesterona como referência; a taxa de detecção variou conforme a
qualidade do evento e a janela temporal
([Perez Marquez et al., 2022](https://doi.org/10.1016/j.animal.2022.100585)).

Um trabalho de julho de 2026 em ovelhas combinou vídeos térmicos, temperatura
dos dois olhos, temperatura/umidade ambiente, rede ResNet18 modificada e
LightGBM, relatando mais de 98,39% para três fases do estro
([Zhang et al., 2026](https://doi.org/10.1016/j.compag.2026.111848)).
Esse número não é comparável ao nosso: são vídeos, outras regiões, sensores
ambientais e um protocolo de três classes. A arquitetura é uma direção para
uma **nova coleta multimodal**, não uma promessa para as 765 imagens atuais.

## Protocolo mínimo da próxima coleta

1. Salvar o JPEG radiométrico original, sem WhatsApp ou recompressão.
2. Sincronizar relógios da câmera, planilha e observadores.
3. Registrar horários de foto, teste de monta e qualquer intervenção.
4. Fotografar em horário fixo, idealmente duas vezes ao dia, e manter ordem
   padronizada entre teste e imagem.
5. Fixar distância, ângulo, orientação, emissividade e fundo; registrar
   temperatura, umidade, vento e exposição solar.
6. Marcar sujeira, umidade, oclusão, borrão, cauda e qualidade da imagem.
7. Confirmar estro/ovulação com progesterona e ultrassonografia em pelo menos
   uma subcoorte; a montagem sozinha permanece um desfecho comportamental.
8. Reservar previamente um lote ou período inteiro para teste externo, sem
   usá-lo para escolher ROI, atributos, algoritmo ou limiar.

Estudos em outras espécies mostram correlação importante entre ambiente e
temperatura vulvar e recomendam o período da manhã quando ele é menos afetado
pelo clima
([Ruediger et al., 2018](https://doi.org/10.1016/j.anireprosci.2018.08.023)).

## Critérios de conclusão

Os critérios abaixo são metas propostas para discussão com os orientadores,
não limites universais:

- **mensuração:** máscara automática próxima da concordância entre anotadores;
- **associação:** efeito temporal pré-especificado, com intervalo de confiança,
  coerência fisiológica e estabilidade entre lotes;
- **discriminação:** limite inferior do intervalo de PR-AUC acima da
  prevalência no teste externo;
- **calibração:** curva, intercepto e inclinação de calibração reportados;
- **operação:** sensibilidade por episódio e falsos alertas por 100
  animal-dias dentro de um custo definido antes do teste;
- **reprodutibilidade:** dados derivados, código, dependências, sementes e
  protocolo documentados.

PR-AUC deve continuar como métrica principal devido ao desbalanceamento
([Saito e Rehmsmeier, 2015](https://doi.org/10.1371/journal.pone.0118432)).
F1, sensibilidade, especificidade, precisão, MCC e desempenho por evento devem
ser apresentados juntos. Acurácia comum não deve selecionar o modelo.

O relato final seguirá, por adaptação, os itens de transparência,
agrupamento, avaliação, incerteza e ciência aberta do
[TRIPOD+AI](https://www.bmj.com/content/385/bmj-2023-078378), e a análise de
risco de viés será guiada pelo
[PROBAST+AI](https://www.bmj.com/content/388/bmj-2024-082505). Essas diretrizes
foram criadas para modelos clínicos humanos; no TCC serão usadas como estrutura
metodológica, não como alegação de conformidade veterinária formal.

## Sequência recomendada

### Etapa 1 — congelar e auditar a base atual

- confirmar relógio da câmera e ordem foto/monta;
- decidir o tratamento de `FLIR0689` e confirmar `FLIR1106`;
- registrar a limitação dos parâmetros radiométricos ambientais;
- finalizar ou excluir explicitamente as imagens sem POI confiável.

### Etapa 2 — validar a medição da vulva

- criar as 120 máscaras estratificadas;
- medir concordância entre anotadores;
- comparar ROI fixa, limiar e segmentação assistida sem usar o rótulo de monta.

### Etapa 3 — análise científica principal

- análise longitudinal/evento com efeitos por animal;
- alvo primário prospectivo “monta na próxima coleta”;
- alvo simultâneo e janela EXIF de 24 h como sensibilidades;
- regressão logística penalizada como modelo principal;
- bootstrap agrupado, calibração e avaliação por episódio.

### Etapa 4 — validação externa

- congelar atributos, modelo e limiar;
- avaliar um novo lote/período;
- reportar a diferença entre desempenho interno e externo.

### Etapa 5 — sistema multimodal

- somente depois da validação básica, acrescentar olho, ambiente, vídeo de
  cauda ou acelerômetro e comparar o ganho incremental com a termografia
  vulvar isolada.

## Como reproduzir as novas auditorias

```powershell
.\venv\Scripts\python.exe src\auditar_timestamps_exif.py

.\venv\Scripts\python.exe src\avaliar_janela_24h.py `
  --targets mount_next_day mount_within_24h `
  --groupings animal date `
  --feature-sets poi fixed_15 hot_contrast `
  --models logistic svm_rbf `
  --timestamp-audit outputs\exif_timestamp_audit\timestamp_audit_records.csv
```

Os resultados derivados ficam em `outputs/` e os dados brutos nunca são
sobrescritos.
