# Premier League 2026/27 · Big 6

Análise e previsão de todos os jogos de Arsenal, Chelsea, Liverpool, Man City,
Man United e Tottenham até a rodada 38, só com dados gratuitos.

## Como usar

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                 # instala o pacote plbig6 a partir de src/
pytest -q                        # testes offline (7 testes)
python -m plbig6.atualizar       # baixa tudo e gera data/processed/*.csv
jupyter lab notebooks/
```

| Notebook | Para quê |
|---|---|
| `01_dados_e_tabela` | fontes usadas, tabela, casa x fora, perfil dos Big 6 |
| `02_previsoes_big6` | forças, próxima rodada, todos os jogos até a 38ª, simulação de título/top 4 |
| `03_escalacao_dia_do_jogo` | previsão sem x com a escalação que você informou |

## Rotina da temporada

* **Dia do jogo (você):** copie `data/escalacoes/MODELO_rodada_XX.csv` para
  `rodada_08.csv` (por exemplo) e liste os titulares. Rode o notebook 03.
* **Terça-feira (Claude):** roda `python -m plbig6.atualizar`, preenche em
  `data/manual/resultados_2026-27.csv` o que as fontes ainda não lançaram
  (só com placar confirmado em 2 sites) e sobe os arquivos atualizados.

## Fontes (todas gratuitas) e ordem de fallback

| Ordem | Fonte | Traz | Observação |
|---|---|---|---|
| 1 | football-data.co.uk | placar, chutes, no alvo, escanteios, cartões, odds | sem número da rodada |
| 2 | openfootball (GitHub) | calendário com rodadas + placar | atrasa alguns dias |
| 3 | API do Fantasy (FPL) | placar oficial + dados de jogador (xG, xA, lesões) | usada nas escalações |
| 4 | `data/manual/` | resultados verificados à mão | cobre atraso de todas |
| extra | Understat | xG por jogo | se falhar, segue sem xG |

Se uma fonte cair ou estourar o limite, o código tenta a próxima e usa cache
antigo quando existe (`sources/base.py`). Se duas fontes discordarem do
placar de um jogo, aparece um aviso de conflito.

**Jogo adiado** fica na rodada original: a rodada vem de
`data/calendario_2026-27.csv`, congelado uma única vez. Não usamos a rodada
do FPL porque ele move jogos adiados para outra rodada.

## O modelo e o quanto confiar nele

Dixon-Coles (Poisson + correção de placares baixos + peso maior para jogos
recentes), treinado com 2023-24 a 2026-27. Detalhes comentados em
`src/plbig6/model.py`.

Backtest (treina só com o passado, prevê cada rodada a partir da 4ª):

| Temporada | Log-loss modelo | Log-loss base | Acerto 1X2 modelo | Acerto base |
|---|---|---|---|---|
| 2024-25 | 0.992 | 1.077 | 52.6% | 41.1% |
| 2025-26 | 1.027 | 1.082 | 46.9% | 42.0% |

Base = chutar sempre a frequência casa/empate/fora da própria temporada
(350 jogos em cada). Testamos incluir a Championship no
treino para melhorar os promovidos e **piorou** nas duas temporadas, então
ficou desligado. Rodar de novo: `python -m plbig6.backtest 2025-26`.

Limitações conhecidas: o modelo só usa gols (xG ainda não entra no ajuste);
o ajuste por escalação é uma heurística não validada; casas de aposta
costumam ser mais precisas que qualquer modelo gratuito como este.
