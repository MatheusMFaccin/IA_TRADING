# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pipeline HFT (High-Frequency Trading) com Conselho de Especialistas (MoE - Mixture of Experts) para previsão de séries temporais financeiras. O projeto implementa um pipeline completo de ML quantitativo desde extração de dados até backtesting.

## Architecture

O pipeline consiste em 5 módulos principais:

1. **baixa_dados.py (Módulo 1)** - Extração de dados MetaTrader5 e geração de barras avançadas (Tick/Volume/Dollar Bars) via amostragem no espaço de atividade de mercado, não tempo.

2. **limpaArquivos.py (Módulo 2)** - Denoising multi-camada: Wavelet DWT → Kalman Adaptativo → Marchenko-Pastur (eigenvalue cleaning).

3. **calcula_alphas.py (Módulo 3)** - Engenharia de features de microestrutura com correlação cruzada < 0.40, Fractional Differentiation para estacionaridade, e Triple Barrier Method para labeling.

4. **llm.py (Módulo 4)** - Conselho de 3 especialistas (LSTM Seq2Seq, XGBoost, Transformer Decoder) com Stacking Aggregator (Ridge Regression) e Conformal Prediction.

5. **moe_gating.py (Módulo 5)** - Evolução do MoE com 4 especialistas (InceptionTime, LSTM, Transformer, Residual MLP) e Gating Network aprendível treinado end-to-end com Loss Morfológica (Soft-DTW + Curvature Penalty).

6. **backtest_engine.py (Módulo 6)** - Backtesting com Purged Walk-Forward Cross-Validation (Purge & Embargo para prevenir leakage) e métricas estatísticas (Sharpe, Sortino, IC, Calmar, Profit Factor).

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run individual modules (execute in order)
python baixa_dados.py      # Extrair dados
python limpaArquivos.py    # Denoising
python calcula_alphas.py   # Features
python llm.py              # Modelo MoE clássico
python moe_gating.py       # MoE com Gating Network v2
python backtest_engine.py  # Backtesting
```

## Key Technologies

- **Deep Learning**: TensorFlow/Keras, XGBoost
- **Signal Processing**: PyWavelets, pykalman
- **Data**: pandas, numpy, h5py, tables
- **Stationarity Tests**: statsmodels (FracDiff)
- **Indicators**: pandas-ta
- **Visualization**: matplotlib, seaborn, plotly

## Data Flow

```
MetaTrader5 → Barras Variáveis → Denoising → Features (Alphas) → MoE → Backtest
```

## Important Notes

- Dados armazenados em HDF5 (.h5) com compressão blosc:zstd
- Intervalo de confiança via Conformal Prediction
- Gating Network usa EMA(10) de Hurst Exponent e GK_Vol para detecção de regime
