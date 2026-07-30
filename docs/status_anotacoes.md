# Status das anotações

Atualizado em 30 de julho de 2026.

## Resumo

A fonte oficial das marcações manuais é
[`planilha/planilha_anotada.CSV`](../planilha/planilha_anotada.CSV). Cada
marcação é um ponto de interesse (POI), armazenado nas colunas `Coord_X` e
`Coord_Y`, que representa o centro aproximado da vulva na matriz térmica de
`80 x 60` pixels.

| Situação | Quantidade |
|---|---:|
| Fotografias referenciadas na planilha | 829 |
| Registros com coordenadas preenchidas | 781 |
| POIs válidos para a ROI principal `11 x 11` | 767 |
| Coordenadas ausentes | 48 |
| Coordenadas sentinela `(0,0)` | 12 |
| Coordenadas inválidas próximas da borda | 2 |
| Total pendente de revisão | 62 |
| Pendências com matriz disponível para anotação | 61 |
| Pendência sem imagem/matriz disponível (`FLIR1107`) | 1 |

As 767 marcações válidas já existiam no CSV versionado. Nesta etapa não foram
criados manualmente os 61 pontos restantes. A recuperação de 25 matrizes
radiométricas tornou essas imagens processáveis, mas não determina
automaticamente o centro anatômico da vulva.

## O que foi calculado automaticamente

A partir de cada POI válido, o pipeline extrai:

- ROI principal quadrada de `11 x 11` pixels;
- ROIs de sensibilidade de `7 x 7` e `15 x 15` pixels;
- uma região por crescimento a partir da semente, usada somente como método de
  comparação;
- mínimo, média, mediana, percentil 90, máximo e desvio-padrão da temperatura.

Essas ROIs automáticas não substituem a marcação manual do centro da vulva.
O método e sua justificativa estão documentados em
[`metodologia_roi.md`](metodologia_roi.md).

## Arquivos versionados e arquivos locais

- `planilha/planilha_anotada.CSV`: coordenadas manuais e demais observações;
- `src/anotador.py`: interface para revisar e completar os POIs;
- `src/extrair_roi_multiescala.py`: geração das ROIs e dos atributos térmicos;
- `docs/metodologia_roi.md`: protocolo metodológico;
- `docs/resultados_preliminares_modelagem.md`: primeira validação agrupada por
  animal.

As imagens, matrizes `.npy`, sobreposições de conferência, resultados gerados e
backups do CSV permanecem fora do Git por causa do volume e para evitar
duplicação. Eles podem ser reproduzidos localmente pelos scripts do projeto.

No momento desta atualização, o arquivo de marcações possui SHA-256:

```text
657A416E482853011C8BDD689C0C73402E6F628FC1645AE6594216C206380CB9
```

## Como concluir as marcações pendentes

Primeiro, recrie a auditoria:

```powershell
.\venv\Scripts\python.exe src\anotador.py --audit
```

Depois, abra um lote de 20 imagens e clique no centro da vulva no painel
térmico:

```powershell
.\venv\Scripts\python.exe src\anotador.py --limit 20
```

O programa salva cada clique no mesmo CSV, cria um backup datado e permite
retomar o trabalho em outro momento. Após finalizar as 61 imagens disponíveis,
é necessário obter `FLIR1107`, marcar seu POI e executar novamente a auditoria.

## Critério de conclusão

A anotação estará completa quando:

1. a auditoria retornar zero POIs pendentes;
2. todos os pontos comportarem a ROI principal `11 x 11`;
3. uma inspeção visual por amostragem confirmar que os pontos estão no centro
   da vulva;
4. o CSV revisado for versionado em um novo commit;
5. as ROIs e a validação dos modelos forem regeneradas a partir do CSV final.
