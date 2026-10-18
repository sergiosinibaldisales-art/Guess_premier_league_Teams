# Brasileirão Série A 2026

Análise e previsão de todos os jogos da Série A 2026 (os 20 times, até a
rodada 38), só com dados gratuitos. Mesma estrutura e mesmo modelo do
projeto `pl-big6-2627` (Premier League), adaptados às fontes brasileiras.

## Como usar

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                      # instala o pacote brasileirao a partir de src/
pytest -q                             # testes offline (7 testes)
python -m brasileirao.rodada          # atualiza tudo + relatório da rodada atual
python -m brasileirao.rodada 32       # relatório de uma rodada específica
python -m brasileirao.lineups 32      # cria a escalação da rodada com os Prováveis do Cartola
jupyter lab notebooks/
```

Saídas: `relatorios/rodada_XX.md` (tudo sobre os jogos que faltam) e
`data/processed/*.csv` (partidas, tabela, previsões, simulação, forças).

| Notebook | Para quê |
|---|---|
| `01_rodada_atual` | relatório da rodada, modelo x mercado, escalação do dia |
| `02_temporada` | tabela, forças, simulação de título/G4/G6/Z4, todos os jogos até a 38ª |

## O que o relatório da rodada traz, por jogo

Data, horário e estádio; posição, pontos e últimos 5 dos dois times;
campanha do mandante em casa e do visitante fora; médias por jogo de gols,
finalizações, no alvo, escanteios, posse e faltas; confronto direto desde
2024; probabilidades do modelo (1X2, placares mais prováveis, +2,5 gols,
ambos marcam, odd justa); odds do mercado sem margem e a diferença para o
modelo; desfalques e dúvidas segundo o Cartola. No fim, tabela atual,
jogos atrasados e a simulação do campeonato.

## Rotina da temporada

* **Dia do jogo (você):** `python -m brasileirao.lineups 32` cria
  `data/escalacoes/rodada_32.csv` com os Prováveis do Cartola. Corrija com a
  escalação oficial e rode `python -m brasileirao.rodada 32`.
* **Depois da última partida da rodada:** `python -m brasileirao.rodada`
  atualiza placares e gera o relatório da próxima rodada. O que nenhuma
  fonte lançou vai em `data/manual/resultados_2026.csv`, só com placar
  confirmado em 2 sites.

## Fontes (todas gratuitas) e ordem de fallback

| Ordem | Fonte | Traz | Observação |
|---|---|---|---|
| 1 | Cartola FC (API pública) | calendário com rodada oficial, placar, estádio, jogadores e status | API não documentada |
| 2 | football-data.co.uk | placar + odds de fechamento; histórico 2024 e 2025 | sem rodada |
| 3 | openfootball (GitHub) | calendário + placar | atrasa algumas rodadas |
| 4 | ESPN (API pública) | finalizações, no alvo, escanteios, posse, faltas; odds da DraftKings | consulta dia a dia |
| 5 | `data/manual/` | resultados verificados à mão | cobre atraso de todas |

Toda fonte passa por `sources/base.py`: cache local, novas tentativas em
erro temporário e cache antigo se a fonte cair. Se duas fontes discordarem
do placar de um jogo, aparece um aviso de conflito. Na primeira execução
(18/10/2026) as quatro fontes de placar concordaram nos 304 jogos.

**Jogo adiado fica na rodada original:** a rodada vem de
`data/calendario_2026.csv`, congelado a partir do Cartola, que usa a rodada
da CBF. Exemplo real: Flamengo x Mirassol é da rodada 4 e foi jogado em
02/09; Chapecoense x Vasco (rodada 21) segue sem data e aparece como
atrasado.

## Escanteios

`src/brasileirao/corners.py` prevê os escanteios de cada time (média da liga
x índice a favor do time x índice cedido pelo adversário, com binomial
negativa). Backtest em 2026 (257 jogos): para o **total** de escanteios o
modelo não bate a frequência da liga (+9,5: log-loss 0.699 x 0.694); para
**quem tem mais escanteios** ajuda um pouco (0.881 x 0.900). Por isso o
relatório mostra a frequência da liga ao lado de cada linha. Rodar de novo:
`python -m brasileirao.corners`.

## O modelo e o quanto confiar nele

Dixon-Coles (Poisson + correção de placares baixos + peso maior para jogos
recentes), treinado com 2024, 2025 e o que já foi jogado de 2026. É o
mesmo `model.py` do projeto da Premier League.

Backtest (treina só com o passado, prevê cada rodada a partir da 4ª):

| Temporada | Jogos | Log-loss modelo | Log-loss base | Log-loss mercado* | Acerto modelo | Acerto mercado* |
|---|---|---|---|---|---|---|
| 2025 | 350 | 0.989 | 1.038 | 0.995 | 51.4% | 51.4% |
| 2026 (até a 31ª) | 274 | 1.011 | 1.048 | 0.990 | 51.8% | 52.8% |

\* mercado = odd de fechamento sem margem (Pinnacle, ou média das casas quando
não há Pinnacle), só nos jogos com odd (315 em 2025, 269 em 2026). Nesses
mesmos jogos o modelo fez 1.009 (2025) e 1.019 (2026), ou seja, pior que o mercado.
Testamos o peso do tempo (xi 0.001 a 0.005): diferença menor que 0.005, ficou 0.0019.
Base = chutar sempre a frequência casa/empate/fora da própria temporada.
Rodar de novo: `python -m brasileirao.backtest 2026`.

Leitura honesta: o modelo é claramente melhor que chute, mas **pior que o
mercado de apostas**. Quando ele discorda muito das odds, o mais provável é
que o mercado saiba de algo que ele não sabe (desfalque, time reserva por
causa de Libertadores/Copa do Brasil). Não use como sinal de aposta.

Limitações conhecidas: só usa gols (finalizações e posse aparecem no
relatório, mas não entram no ajuste); não sabe de rodízio de elenco; o ajuste
por escalação é uma heurística não validada, baseada em gols e assistências
do Cartola; as vagas de G6 dependem das copas e são aproximadas.
