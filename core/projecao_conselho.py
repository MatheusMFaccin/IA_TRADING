"""
===============================================================================
MÓDULO 6: PROJEÇÃO E VISUALIZAÇÃO — INTEGRAÇÃO COM CONSELHO DE ESPECIALISTAS
===============================================================================
Módulo de visualização e inferência que consome o Conselho de Especialistas
definido em llm.py. Gera gráficos de candlestick com projeção futura e
bandas de confiança via Conformal Prediction.

Mudanças vs. versão anterior:
    1. Modelos removidos (agora em llm.py) — separação de responsabilidades
    2. Bandas de confiança via Conformal Prediction (não mais divergência)
    3. Integração com predict_candles() de llm.py
    4. Suporte a visualização de predições individuais dos especialistas

    Gráf
Arquitetura:

    llm.py (Especialistas + Aggregator + Conformal)
        ↓
    projecao_conselho.py (Visualização + Inferência Rápida)
        ↓ico Candlestick + Bandas de Confiança

Referências:
    - Salinas et al. (2020): "DeepAR: Probabilistic Forecasting"
    - Lim et al. (2021): "Temporal Fusion Transformers"
===============================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
from typing import Optional, Dict, Any, List

warnings.filterwarnings('ignore')

# Imports do Conselho de Especialistas
from llm import (
    run_council_pipeline,
    predict_candles,
    relative_to_absolute,
    StackingAggregator,
    ConformalPredictor,
)


# =============================================================================
# VISUALIZAÇÃO: CANDLESTICK COM PROJEÇÃO + CONFORMAL BANDS
# =============================================================================

def plot_projection(
    df_recent: pd.DataFrame,
    projected_ohlc: np.ndarray,
    upper_band: Optional[np.ndarray] = None,
    lower_band: Optional[np.ndarray] = None,
    confidence_level: float = 0.90,
    individual_preds: Optional[Dict[str, np.ndarray]] = None,
    n_history: int = 80,
    title: str = "Conselho de Especialistas — USDJPY",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Gera gráfico de candlestick com projeção futura e bandas conformais.

    Barras históricas (verde/vermelho) + barras projetadas (azul/laranja)
    com bandas de confiança via Conformal Prediction.

    Parâmetros:
    -----------
    df_recent : pd.DataFrame
        Últimas barras reais com OHLC.
    projected_ohlc : np.ndarray
        Array (horizon, 4) com OHLC projetado absoluto.
    upper_band / lower_band : np.ndarray, optional
        Bandas de confiança conformais (horizon, 4).
    confidence_level : float
        Nível de confiança (para legenda).
    individual_preds : dict, optional
        Predições individuais dos especialistas para overlay.
    n_history : int
        Número de barras históricas a mostrar.
    """
    fig, ax = plt.subplots(figsize=(20, 9))

    # Configuração de fundo escuro premium
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    ax.tick_params(colors='#8b949e')
    ax.spines['bottom'].set_color('#30363d')
    ax.spines['left'].set_color('#30363d')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- BARRAS HISTÓRICAS ---
    hist = df_recent.tail(n_history).reset_index(drop=True)

    for i in range(len(hist)):
        o = hist['Open'].iloc[i]
        h = hist['High'].iloc[i]
        l = hist['Low'].iloc[i]
        c = hist['Close'].iloc[i]

        is_bull = c >= o
        color = '#26a69a' if is_bull else '#ef5350'

        # Sombra
        ax.plot([i, i], [l, h], color=color, linewidth=0.8, alpha=0.8)

        # Corpo
        body_bottom = min(o, c)
        body_height = abs(c - o)
        if body_height < 1e-10:
            body_height = (h - l) * 0.01

        rect = plt.Rectangle(
            (i - 0.35, body_bottom), 0.7, body_height,
            facecolor=color, edgecolor=color, alpha=0.9, linewidth=0.5
        )
        ax.add_patch(rect)

    # --- LINHA DIVISÓRIA ---
    divider_x = len(hist) - 0.5
    ax.axvline(x=divider_x, color='#58a6ff', linewidth=1.5,
               linestyle='--', alpha=0.7)

    # Posicionar texto AGORA
    y_range = ax.get_ylim()
    if y_range[0] == 0 and y_range[1] == 1:
        text_y = hist['Close'].min()
    else:
        text_y = y_range[0]

    ax.text(divider_x + 0.5, text_y,
            'AGORA', color='#58a6ff', fontsize=10, fontweight='bold',
            verticalalignment='bottom')

    # --- BARRAS PROJETADAS ---
    horizon = len(projected_ohlc)

    for k in range(horizon):
        x_pos = len(hist) + k
        o = projected_ohlc[k, 0]
        h = projected_ohlc[k, 1]
        l = projected_ohlc[k, 2]
        c = projected_ohlc[k, 3]

        is_bull = c >= o
        color = '#4fc3f7' if is_bull else '#ff8a65'

        # Sombra
        ax.plot([x_pos, x_pos], [l, h], color=color, linewidth=0.8, alpha=0.7)

        # Corpo
        body_bottom = min(o, c)
        body_height = abs(c - o)
        if body_height < 1e-10:
            body_height = (h - l) * 0.01

        rect = plt.Rectangle(
            (x_pos - 0.35, body_bottom), 0.7, body_height,
            facecolor=color, edgecolor=color, alpha=0.6, linewidth=0.5
        )
        ax.add_patch(rect)

    # --- BANDAS DE CONFIANÇA CONFORMAL ---
    if upper_band is not None and lower_band is not None:
        proj_x = np.arange(len(hist), len(hist) + horizon)

        # Banda externa (High upper → Low lower)
        ax.fill_between(
            proj_x, upper_band[:, 1], lower_band[:, 2],
            alpha=0.06, color='#58a6ff',
            label=f'Conformal Band ({confidence_level:.0%})'
        )

        # Banda Close (CI do close)
        ax.fill_between(
            proj_x, upper_band[:, 3], lower_band[:, 3],
            alpha=0.12, color='#58a6ff',
            label=f'Close CI ({confidence_level:.0%})'
        )

        # Linha central do Close projetado
        ax.plot(
            proj_x, projected_ohlc[:, 3],
            color='#58a6ff', linewidth=1.5, linestyle='-',
            alpha=0.8, label='Close Projetado'
        )

    # --- PREDIÇÕES INDIVIDUAIS (overlay) ---
    if individual_preds is not None:
        proj_x = np.arange(len(hist), len(hist) + horizon)
        colors_ind = {'lstm_seq2seq': '#ff6b6b', 'xgboost_multi': '#ffd93d',
                      'transformer': '#6bcb77'}
        labels_ind = {'lstm_seq2seq': 'LSTM Seq2Seq', 'xgboost_multi': 'XGBoost',
                      'transformer': 'Transformer'}

        for name, pred in individual_preds.items():
            if name in colors_ind:
                ax.plot(proj_x, pred[:, 3], color=colors_ind[name],
                        linewidth=0.8, alpha=0.4, linestyle='--',
                        label=labels_ind.get(name, name))

    # --- LEGENDA E LABELS ---
    legend_elements = [
        mpatches.Patch(facecolor='#26a69a', label='Histórico Bull'),
        mpatches.Patch(facecolor='#ef5350', label='Histórico Bear'),
        mpatches.Patch(facecolor='#4fc3f7', alpha=0.6, label='Projeção Bull'),
        mpatches.Patch(facecolor='#ff8a65', alpha=0.6, label='Projeção Bear'),
        mpatches.Patch(facecolor='#58a6ff', alpha=0.15,
                       label=f'Conformal Band ({confidence_level:.0%})'),
    ]
    ax.legend(
        handles=legend_elements, loc='upper left',
        facecolor='#161b22', edgecolor='#30363d',
        fontsize=9, labelcolor='#c9d1d9'
    )

    ax.set_title(title, fontsize=16, fontweight='bold',
                 color='#f0f6fc', pad=15)
    ax.set_xlabel('Tick Bar Index', fontsize=11, color='#8b949e')
    ax.set_ylabel('Preço', fontsize=11, color='#8b949e')
    ax.grid(True, alpha=0.1, color='#30363d')

    # Ajustar ylim
    all_prices: list = list(hist[['Open', 'High', 'Low', 'Close']].values.flatten())
    all_prices.extend(projected_ohlc.flatten())
    if upper_band is not None:
        all_prices.extend(upper_band.flatten())
        all_prices.extend(lower_band.flatten())
    margin = (max(all_prices) - min(all_prices)) * 0.05
    ax.set_ylim(min(all_prices) - margin, max(all_prices) + margin)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='#0d1117')
        print(f"  Gráfico salvo: {save_path}")

    plt.show()
    return fig


# =============================================================================
# PIPELINE DE PROJEÇÃO (INTEGRADO COM llm.py)
# =============================================================================

def run_projection_council(
    input_file: str = 'dataset_final.h5',
    window_size: int = 60,
    horizon: int = 15,
    purge_bars: int = 200,
    n_splits: int = 5,
    epochs: int = 50,
    batch_size: int = 64,
    confidence_level: float = 0.90,
    decay_lambda: float = 0.05,
    show_plot: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Pipeline completo: Treino + Projeção + Visualização.

    Delega o treino e inferência ao Conselho de Especialistas (llm.py)
    e se encarrega da visualização e formatação de resultados.

    Parâmetros:
    -----------
    input_file : str
        Arquivo HDF5 com features.
    window_size : int
        Janela de observação (lookback).
    horizon : int
        Barras futuras a projetar (H=15).
    purge_bars : int
        Gap entre treino e validação.
    n_splits : int
        Folds de walk-forward.
    epochs : int
        Épocas de treino.
    batch_size : int
        Tamanho do batch.
    confidence_level : float
        Nível de confiança para Conformal Prediction.
    decay_lambda : float
        Fator de decay do XGBoost.
    show_plot : bool
        Se True, gera visualização candlestick.

    Retorna:
    --------
    dict com: projected_ohlc, confidence_intervals, metrics, ensemble, models
    """
    # 1. Executar pipeline do Conselho
    results = run_council_pipeline(
        input_file=input_file,
        window_size=window_size,
        horizon=horizon,
        purge_bars=purge_bars,
        n_splits=n_splits,
        epochs=epochs,
        batch_size=batch_size,
        confidence_level=confidence_level,
        decay_lambda=decay_lambda
    )

    if results is None:
        return None

    # 2. Carregar dados para visualização
    df: Optional[pd.DataFrame] = None
    sources = [
        (input_file, 'features'),
        ('dataset_clean.h5', 'data'),
        ('dataset_final.h5', 'features'),
    ]
    for filepath, key in sources:
        try:
            df = pd.read_hdf(filepath, key=key)
            break
        except (FileNotFoundError, KeyError):
            continue

    if df is None:
        print("  ⚠️ Não foi possível carregar dados para visualização.")
        return results

    # 3. Resumo da projeção
    proj_ohlc = results['projected_ohlc']
    base_price = results['base_price']

    print(f"\n{'═' * 65}")
    print(f"  RESUMO DA PROJEÇÃO")
    print(f"{'═' * 65}")

    direction = "ALTA ↑" if proj_ohlc[-1, 3] > base_price else "BAIXA ↓"
    delta_pct = (proj_ohlc[-1, 3] - base_price) / base_price * 100
    max_h = float(np.max(proj_ohlc[:, 1]))
    min_l = float(np.min(proj_ohlc[:, 2]))
    range_pct = (max_h - min_l) / base_price * 100

    print(f"  Direção projetada:       {direction}")
    print(f"  Variação projetada:      {delta_pct:+.4f}%")
    print(f"  Range projetado:         {range_pct:.4f}%")
    print(f"  High máximo projetado:   {max_h:.5f}")
    print(f"  Low mínimo projetado:    {min_l:.5f}")
    print(f"  Confiança:               {confidence_level:.0%} (Conformal)")

    # Pesos do aggregator
    agg = results.get('aggregator')
    if agg is not None:
        print(f"  Shrinkage Factor:        {agg.shrinkage:.3f}")
        print(f"  Pesos: LSTM={agg.weights[0]:.3f} | "
              f"XGB={agg.weights[1]:.3f} | "
              f"TFM={agg.weights[2]:.3f}")

    print(f"{'═' * 65}")

    # 4. Visualização
    if show_plot:
        plot_projection(
            df_recent=df,
            projected_ohlc=proj_ohlc,
            upper_band=results.get('confidence_upper'),
            lower_band=results.get('confidence_lower'),
            confidence_level=confidence_level,
            individual_preds=results.get('individual_predictions'),
            n_history=80,
            title=f"Conselho de Especialistas: Projeção {horizon} Barras — USDJPY",
            save_path='projecao_movimento.png'
        )

    return results


# =============================================================================
# PROJEÇÃO RÁPIDA (Sem Retreino)
# =============================================================================

def quick_project(
    df: pd.DataFrame,
    models: Dict[str, Any],
    aggregator: StackingAggregator,
    conformal: Optional[ConformalPredictor] = None,
    window_size: int = 60,
    horizon: int = 15,
    confidence_level: float = 0.90,
    feature_cols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Projeção rápida usando modelos já treinados.
    Útil para atualização em tempo real sem retreinar.

    Delega ao predict_candles() de llm.py.

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame com dados atualizados (OHLC + features).
    models : dict
        {'lstm': Model, 'xgb': Forecaster, 'transformer': Model}
    aggregator : StackingAggregator
        Meta-learner do llm.py.
    conformal : ConformalPredictor, optional
        Para intervalos de confiança.
    """
    return predict_candles(
        df=df,
        horizon=horizon,
        confidence_level=confidence_level,
        models=models,
        aggregator=aggregator,
        conformal=conformal,
        window_size=window_size,
        feature_cols=feature_cols
    )


# =============================================================================
# EXECUÇÃO
# =============================================================================
if __name__ == "__main__":
    results = run_projection_council(
        input_file='dataset_final.h5',
        window_size=60,
        horizon=15,
        purge_bars=200,
        n_splits=5,
        epochs=50,
        batch_size=64,
        confidence_level=0.90,
        decay_lambda=0.05,
        show_plot=True
    )
