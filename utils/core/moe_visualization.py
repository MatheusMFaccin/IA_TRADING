# -*- coding: utf-8 -*-
"""
===============================================================================
VISUALIZAÇÃO STANDALONE: MoE GATING — ANÁLISE PÓS-TREINO
===============================================================================
Script de visualização 100% independente do processo de treino.

Carrega resultados diretamente de:
    · results_inference.parquet   — predições, bandas conformais, gating
    · moe_config.json             — hiperparâmetros e fold_metrics

NÃO importa TensorFlow, moe_gating ou calcula_alphas.

Painéis disponíveis:
    1. Candlestick histórico + barras projetadas (MoE) + bandas conformais
    2. Overlay dos 4 especialistas individuais vs MoE ponderado (Close)
    3. Pesos do Gating ao longo do tempo (barras empilhadas)
    4. (Comparação) MoE vs Real + Erro residual + Gating por fold

Hierarquia de chamadas:
    moe_gating.py      — definições de modelo (NÃO importado aqui)
    moe_to_daily.py    — treino + salva .parquet + .json
    moe_visualization.py ★ ESTE ARQUIVO ★ — lê .parquet → plots
===============================================================================
"""

from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
import warnings
from typing import Optional, Dict, Any, List

from plotly.subplots import make_subplots
import plotly.graph_objects as go


warnings.filterwarnings('ignore')

# Paleta de cores dos especialistas
EXPERT_COLORS = {
    'inception':   '#f72585',   # Rosa vibrante — CNN
    'lstm':        '#4cc9f0',   # Azul ciano — LSTM
    'transformer': '#7209b7',   # Roxo — Transformer
    'mlp':         '#4ade80',   # Verde — MLP
    'moe':         '#fbbf24',   # Âmbar — MoE ensemble
}
EXPERT_LABELS = {
    'inception':   'E1: InceptionTime',
    'lstm':        'E2: LSTM Seq2Seq',
    'transformer': 'E3: Transformer',
    'mlp':         'E4: Residual MLP',
    'moe':         'MoE Ponderado',
}

# ── Cores / estilo base usados em ambos os gráficos ─────────────────────────
BG      = "#0d1117"
PAPER   = "#0d1117"
GRID    = "rgba(48,54,61,0.4)"
TICK_C  = "#8b949e"
TITLE_C = "#f0f6fc"

AXIS_DEFAULTS = dict(
    gridcolor=GRID,
    zerolinecolor=GRID,
    tickfont=dict(color=TICK_C, size=9),
    showline=True,
    linecolor="#30363d",
)


# =============================================================================
# CARREGAMENTO DE ARTEFATOS (Standalone — sem TensorFlow)
# =============================================================================

def load_inference_results(
    parquet_path: str = '../exports/inference/results_inference.parquet',
) -> Dict[str, Any]:
    """
    Carrega o arquivo de resultados .parquet e reconstrói o dicionário
    ``results`` compatível com as funções de plot existentes.

    O .parquet contém:
        step, Open_proj..Close_proj, Open_lower..Close_lower,
        Open_upper..Close_upper, Close_inception..Close_mlp,
        gating_inception..gating_mlp, base_price

    Retorna dict com:
        projected_ohlc, confidence_lower, confidence_upper,
        base_price, individual_predictions, gating_weights_final,
        gating_labels, confidence_level
    """
    df = pd.read_parquet(parquet_path)
    horizon = len(df)

    # ── Projeção OHLC ────────────────────────────────────────────────────
    proj_cols = ['Open_proj', 'High_proj', 'Low_proj', 'Close_proj']
    projected_ohlc = df[proj_cols].values  # (H, 4)

    # ── Bandas conformais ────────────────────────────────────────────────
    lower_cols = ['Open_lower', 'High_lower', 'Low_lower', 'Close_lower']
    upper_cols = ['Open_upper', 'High_upper', 'Low_upper', 'Close_upper']

    confidence_lower = df[lower_cols].values if all(c in df.columns for c in lower_cols) else None
    confidence_upper = df[upper_cols].values if all(c in df.columns for c in upper_cols) else None

    # ── Predições individuais ────────────────────────────────────────────
    ind_preds = {}
    for key in ['inception', 'lstm', 'transformer', 'mlp']:
        col = f'Close_{key}'
        if col in df.columns:
            # Reconstruir (H, 4) com Close do expert; copiar OHLC do MoE
            # para Open/High/Low (aproximação para visualização)
            expert_arr = projected_ohlc.copy()
            expert_arr[:, 3] = df[col].values
            ind_preds[key] = expert_arr

    # ── Gating weights ───────────────────────────────────────────────────
    gating_cols = ['gating_inception', 'gating_lstm', 'gating_transformer', 'gating_mlp']
    if all(c in df.columns for c in gating_cols):
        gating_final = df[gating_cols].iloc[0].values
    else:
        gating_final = np.ones(4) / 4

    # ── Base price ───────────────────────────────────────────────────────
    base_price = df['base_price'].iloc[0] if 'base_price' in df.columns else projected_ohlc[0, 0]

    # ── Metadados (podem estar em .attrs se o engine suportar) ──────────
    confidence_level = df.attrs.get('confidence_level', 0.90) if hasattr(df, 'attrs') else 0.90

    gating_labels_json = df.attrs.get('gating_labels', None) if hasattr(df, 'attrs') else None
    if gating_labels_json:
        try:
            gating_labels = json.loads(gating_labels_json)
        except (json.JSONDecodeError, TypeError):
            gating_labels = ['InceptionTime', 'LSTM', 'Transformer', 'ResidualMLP']
    else:
        gating_labels = ['InceptionTime', 'LSTM', 'Transformer', 'ResidualMLP']

    print(f"  📂 Resultados carregados: {parquet_path} ({horizon} steps)")

    return {
        'projected_ohlc':         projected_ohlc,
        'confidence_lower':       confidence_lower,
        'confidence_upper':       confidence_upper,
        'base_price':             base_price,
        'confidence_level':       confidence_level,
        'individual_predictions': ind_preds,
        'gating_weights_final':   gating_final,
        'gating_labels':          gating_labels,
    }


def load_config(
    config_path: str = '../models/EURUSD/moe_config.json',
) -> Dict[str, Any]:
    """
    Carrega o arquivo de configuração JSON do pipeline.

    Retorna dict com hiperparâmetros, fold_metrics, gating_weights_final.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"  📂 Config carregada: {config_path}")
    return data


# =============================================================================
# PLOT PRINCIPAL: 3 PAINÉIS  (Plotly)
# =============================================================================

def plot_moe_analysis(
    df_history: pd.DataFrame,
    results: Dict[str, Any],
    n_history: int = 80,
    save_path: Optional[str] = '../exports/plots/moe_analysis.html',
    title_suffix: str = "USD/JPY",
) -> go.Figure:
    """
    Gera o gráfico de análise em 3 painéis do MoE Gating usando Plotly.

    Painéis:
        1. Candlestick histórico + projeção MoE + bandas conformais
        2. Close de cada especialista vs MoE ponderado
        3. Pesos do Gating (por fold ou barra)

    Parâmetros:
    -----------
    df_history : pd.DataFrame
        DataFrame com dados históricos reais (OHLC + features).
    results : dict
        Output de load_inference_results() ou run_moe_pipeline().
    n_history : int
        Número de barras históricas a exibir.
    save_path : str, optional
        Caminho para salvar o gráfico (.html ou .png/.webp via write_image).
    """
    proj_ohlc    = results['projected_ohlc']
    upper_band   = results.get('confidence_upper')
    lower_band   = results.get('confidence_lower')
    ind_preds    = results.get('individual_predictions', {})
    gating_final = results.get('gating_weights_final', np.ones(4) / 4)
    gat_labels   = results.get('gating_labels',
                                ['InceptionTime', 'LSTM', 'Transformer', 'MLP'])
    fold_metrics = results.get('fold_metrics', [])
    base_price   = results.get('base_price', proj_ohlc[0, 0])
    horizon      = len(proj_ohlc)

    # Dados históricos
    hist = df_history.tail(n_history).reset_index(drop=True)
    hist_ohlc = hist[['Open', 'High', 'Low', 'Close']].values
    n_hist = len(hist_ohlc)

    # Direção e variação
    direction = "↑ ALTA" if proj_ohlc[-1, 3] > base_price else "↓ BAIXA"
    delta_pct = (proj_ohlc[-1, 3] - base_price) / base_price * 100

    # ── Layout ──────────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.50, 0.28, 0.22],
        shared_xaxes=False,
        vertical_spacing=0.07,
        subplot_titles=(
            (f"MoE Gating Network — {title_suffix}   │   "
             f"Projeção: {horizon} barras   │   {direction} {delta_pct:+.4f}%"),
            "Comparação de Especialistas — Close Projetado",
            "Pesos do Gating — Especialização por Regime",
        ),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # PAINEL 1 — Candlestick Histórico + Projeção MoE
    # ══════════════════════════════════════════════════════════════════════════

    hist_x = list(range(n_hist))
    proj_x = list(range(n_hist, n_hist + horizon))

    # Histórico — candlestick
    fig.add_trace(
        go.Candlestick(
            x=hist_x,
            open=hist_ohlc[:, 0],
            high=hist_ohlc[:, 1],
            low=hist_ohlc[:, 2],
            close=hist_ohlc[:, 3],
            name="Histórico",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a",
            decreasing_fillcolor="#ef5350",
            opacity=0.85,
            showlegend=True,
        ),
        row=1, col=1,
    )

    # Divisor "AGORA"
    div_x = n_hist - 0.5
    fig.add_vline(
        x=div_x, row=1, col=1,
        line=dict(color="#58a6ff", width=1.8, dash="dash"),
        opacity=0.75,
        annotation_text="AGORA →",
        annotation_font=dict(color="#58a6ff", size=10),
        annotation_position="bottom right",
    )

    # Projeção MoE — OHLC
    fig.add_trace(
        go.Ohlc(
            x=proj_x,
            open=proj_ohlc[:, 0],
            high=proj_ohlc[:, 1],
            low=proj_ohlc[:, 2],
            close=proj_ohlc[:, 3],
            name="Projeção MoE",
            increasing_line_color="#4fc3f7",
            decreasing_line_color="#ff8a65",
            opacity=0.65,
            showlegend=True,
        ),
        row=1, col=1,
    )

    # Linha Close projetado (MoE)
    fig.add_trace(
        go.Scatter(
            x=proj_x,
            y=proj_ohlc[:, 3],
            mode="lines",
            line=dict(color=EXPERT_COLORS['moe'], width=2.2),
            name="MoE Close",
            opacity=0.9,
        ),
        row=1, col=1,
    )

    # Especialistas individuais (Close overlay no painel 1)
    expert_keys = ['inception', 'lstm', 'transformer', 'mlp']
    for key in expert_keys:
        if key in ind_preds:
            pred = ind_preds[key]
            close_vals = pred[:, 3] if pred.ndim == 2 else pred[:horizon, 3]
            fig.add_trace(
                go.Scatter(
                    x=proj_x,
                    y=close_vals,
                    mode="lines",
                    line=dict(color=EXPERT_COLORS[key], width=1.0, dash="dash"),
                    opacity=0.45,
                    name=EXPERT_LABELS[key],
                    showlegend=True,
                ),
                row=1, col=1,
            )

    # Bandas conformais
    if upper_band is not None and lower_band is not None:
        ci_label = f"CI ({results.get('confidence_level', 0.9):.0%})"
        # Banda externa (High/Low)
        fig.add_trace(
            go.Scatter(
                x=proj_x + proj_x[::-1],
                y=list(upper_band[:, 1]) + list(lower_band[::-1, 2]),
                fill="toself",
                fillcolor="rgba(88,166,255,0.05)",
                line=dict(width=0),
                name=ci_label + " (H/L)",
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1, col=1,
        )
        # Banda interna (Close)
        fig.add_trace(
            go.Scatter(
                x=proj_x + proj_x[::-1],
                y=list(upper_band[:, 3]) + list(lower_band[::-1, 3]),
                fill="toself",
                fillcolor="rgba(88,166,255,0.10)",
                line=dict(width=0),
                name=ci_label,
                hoverinfo="skip",
                showlegend=True,
            ),
            row=1, col=1,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # PAINEL 2 — Close de cada especialista (comparação)
    # ══════════════════════════════════════════════════════════════════════════

    bar_x = list(range(1, horizon + 1))
    moe_close = proj_ohlc[:, 3]

    fig.add_trace(
        go.Scatter(
            x=bar_x,
            y=moe_close,
            mode="lines+markers",
            line=dict(color=EXPERT_COLORS['moe'], width=2.5),
            marker=dict(size=4),
            name="MoE Ponderado (p2)",
            showlegend=True,
        ),
        row=2, col=1,
    )

    for key in expert_keys:
        if key in ind_preds:
            pred = ind_preds[key]
            c = pred[:, 3] if pred.ndim == 2 else pred[:horizon, 3]
            fig.add_trace(
                go.Scatter(
                    x=bar_x,
                    y=c,
                    mode="lines+markers",
                    line=dict(color=EXPERT_COLORS[key], width=1.2, dash="dash"),
                    marker=dict(size=3),
                    opacity=0.7,
                    name=EXPERT_LABELS[key] + " (p2)",
                    showlegend=True,
                ),
                row=2, col=1,
            )

    # Linha base (último preço real)
    fig.add_hline(
        y=base_price, row=2, col=1,
        line=dict(color="#8b949e", width=0.8, dash="dot"),
        opacity=0.6,
        annotation_text=f"Base: {base_price:.4f}",
        annotation_font=dict(color="#8b949e", size=9),
        annotation_position="bottom right",
    )

    # ══════════════════════════════════════════════════════════════════════════
    # PAINEL 3 — Pesos do Gating
    # ══════════════════════════════════════════════════════════════════════════

    gating_weights_series = results.get('gating_weights_series')
    colors_g = [EXPERT_COLORS[k] for k in ['inception', 'lstm', 'transformer', 'mlp']]

    if gating_weights_series is not None and len(gating_weights_series) == horizon:
        # Pesos variam por barra — stackplot via stacked area
        w_arr = np.array(gating_weights_series)  # (H, 4)
        for i, (lbl, col) in enumerate(zip(gat_labels, colors_g)):
            fig.add_trace(
                go.Scatter(
                    x=bar_x,
                    y=w_arr[:, i],
                    mode="lines",
                    line=dict(color=col, width=0),
                    fill="tonexty" if i > 0 else "tozeroy",
                    fillcolor=col.replace(")", ",0.8)").replace("rgb", "rgba")
                             if "rgb" in col else col,
                    name=lbl,
                    stackgroup="gating",
                    opacity=0.8,
                ),
                row=3, col=1,
            )
    elif fold_metrics:
        # Pesos ao longo dos folds (evolução do aprendizado)
        folds_x = [m['fold'] for m in fold_metrics]
        gw = np.array([m['gating_weights'] for m in fold_metrics])
        for i, (label, color) in enumerate(zip(gat_labels, colors_g)):
            fig.add_trace(
                go.Scatter(
                    x=folds_x,
                    y=gw[:, i],
                    mode="lines+markers",
                    line=dict(color=color, width=2),
                    marker=dict(size=6),
                    name=label,
                    opacity=0.85,
                ),
                row=3, col=1,
            )
    else:
        # Último recurso: barras com os pesos finais
        fig.add_trace(
            go.Bar(
                x=gat_labels,
                y=gating_final,
                marker_color=colors_g,
                opacity=0.85,
                name="Pesos Gating",
                text=[f"{w:.3f}" for w in gating_final],
                textposition="outside",
                textfont=dict(color="#f0f6fc", size=11),
            ),
            row=3, col=1,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Layout global
    # ══════════════════════════════════════════════════════════════════════════

    fig.update_layout(
        height=1050,
        width=1400,
        paper_bgcolor=PAPER,
        plot_bgcolor=BG,
        font=dict(color=TICK_C),
        title=dict(
            text=(
                "MoE Gating Network │ EMA(10)[Hurst, GK_Vol] → Softmax(4) │ "
                "Loss: Soft-DTW + Curvature Penalty"
            ),
            font=dict(size=13, color="#8b949e"),
            x=0.5,
        ),
        legend=dict(
            bgcolor="#161b22",
            bordercolor="#30363d",
            borderwidth=1,
            font=dict(color="#c9d1d9", size=9),
        ),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
    )

    # Aplicar estilo em todos os eixos
    for i in range(1, 4):
        fig.update_xaxes(**AXIS_DEFAULTS, row=i, col=1)
        fig.update_yaxes(**AXIS_DEFAULTS, row=i, col=1)

    # Rótulos
    fig.update_yaxes(title_text="Preço",          title_font=dict(color=TICK_C, size=11), row=1, col=1)
    fig.update_yaxes(title_text="Close (absoluto)", title_font=dict(color=TICK_C, size=10), row=2, col=1)
    fig.update_xaxes(title_text="Barra Futura",   title_font=dict(color=TICK_C, size=10), row=2, col=1)

    if fold_metrics:
        fig.update_xaxes(title_text="Fold (Walk-Forward)", title_font=dict(color=TICK_C, size=10), row=3, col=1)
        fig.update_yaxes(title_text="Peso Médio",          title_font=dict(color=TICK_C, size=10), row=3, col=1)
    else:
        fig.update_yaxes(title_text="Peso Softmax",        title_font=dict(color=TICK_C, size=10), row=3, col=1)

    # Estilo dos títulos dos subplots
    for ann in fig.layout.annotations:
        ann.update(font=dict(color="#c9d1d9", size=12))

    # ── Salvar / exibir ─────────────────────────────────────────────────────
    if save_path:
        if save_path.endswith(".html"):
            fig.write_html(save_path)
        else:
            fig.write_image(save_path, scale=2)
        print(f"  📊 Gráfico salvo: {save_path}")

    fig.show()
    return fig


# =============================================================================
# COMPARAÇÃO: MoE vs REALIDADE (plot de 4 painéis — usa dados pré-carregados)
# =============================================================================


def plot_comparison_vs_real(
    df_history: pd.DataFrame,
    results: Dict[str, Any],
    real_ohlc: np.ndarray,
    n_history: int = 60,
    save_path: Optional[str] = None,
) -> go.Figure:
    """
    Plota MoE vs realidade com painel de erro e pesos do Gating usando Plotly.

    Painéis
    -------
    1. Candlesticks históricos + projeção MoE (azul) + real (verde) lado a lado
    2. Close de cada especialista vs real
    3. Erro por barra (Projetado − Real) com IC conformal
    4. Pesos do Gating por fold (linha) ou barra única se não houver folds
    """
    proj_abs   = results["projected_ohlc"]           # (H, 4)
    lower_abs  = results.get("confidence_lower")      # (H, 4) ou None
    upper_abs  = results.get("confidence_upper")      # (H, 4) ou None
    ind_preds  = results.get("individual_predictions", {})
    gat_w      = results.get("gating_weights_final", np.ones(4) / 4)
    gat_labels = results.get("gating_labels", ["Inc", "LSTM", "TFM", "MLP"])
    metrics    = results.get("metrics", {})
    fold_met   = results.get("fold_metrics", [])
    base_price = results.get("base_price", proj_abs[0, 0])

    horizon = len(proj_abs)
    n_real  = min(horizon, len(real_ohlc))

    hist     = df_history.tail(n_history).reset_index(drop=True)
    hist_ohlc = hist[["Open", "High", "Low", "Close"]].values
    n_hist   = len(hist_ohlc)

    bar_x    = np.arange(1, horizon + 1)          # eixo-x painéis 2-4

    # ── Títulos das métricas ───────────────────────────────────────────────────
    mae_c   = metrics.get("mae_close",           float("nan"))
    dir_acc = metrics.get("dir_accuracy",         float("nan"))
    cov     = metrics.get("conformal_coverage",   float("nan"))
    flat    = metrics.get("flat_line_rate",        float("nan"))
    curv    = metrics.get("mean_curvature",        float("nan"))

    title_p1 = (
        f"MoE vs Real │ MAE={mae_c:.5f} │ Dir={dir_acc:.0%} │ "
        f"Cov={cov:.0%} │ Flat={flat:.0%} │ Curv={curv:.6f}"
    )

    # ── Figura com 4 linhas ────────────────────────────────────────────────────
    fig = make_subplots(
        rows=4, cols=1,
        row_heights=[0.44, 0.22, 0.17, 0.17],
        shared_xaxes=False,
        vertical_spacing=0.06,
        subplot_titles=(
            title_p1,
            "Especialistas Individuais vs Real",
            "Erro por Barra (Projetado − Real)",
            "Evolução dos Pesos do Gating — Aprendizado por Regime",
        ),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # PAINEL 1 — Histórico + Projeção + Real
    # ══════════════════════════════════════════════════════════════════════════

    hist_x = list(range(n_hist))
    fig.add_trace(
        go.Candlestick(
            x=hist_x,
            open=hist_ohlc[:, 0],
            high=hist_ohlc[:, 1],
            low=hist_ohlc[:, 2],
            close=hist_ohlc[:, 3],
            name="Histórico",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a",
            decreasing_fillcolor="#ef5350",
            opacity=0.85,
            showlegend=True,
        ),
        row=1, col=1,
    )

    div_x = n_hist - 0.5
    fig.add_vline(
        x=div_x, row=1, col=1,
        line=dict(color="#58a6ff", width=1.8, dash="dash"),
        opacity=0.75,
    )

    proj_x_float = [n_hist + k for k in range(horizon)]

    fig.add_trace(
        go.Ohlc(
            x=[n_hist + k - 0.18 for k in range(horizon)],
            open=proj_abs[:, 0],
            high=proj_abs[:, 1],
            low=proj_abs[:, 2],
            close=proj_abs[:, 3],
            name="Projeção MoE",
            increasing_line_color="#4fc3f7",
            decreasing_line_color="#ff8a65",
            opacity=0.6,
            showlegend=True,
        ),
        row=1, col=1,
    )

    if n_real > 0:
        fig.add_trace(
            go.Ohlc(
                x=[n_hist + k + 0.18 for k in range(n_real)],
                open=real_ohlc[:n_real, 0],
                high=real_ohlc[:n_real, 1],
                low=real_ohlc[:n_real, 2],
                close=real_ohlc[:n_real, 3],
                name="Real",
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350",
                opacity=0.9,
                showlegend=True,
            ),
            row=1, col=1,
        )

    if upper_abs is not None and lower_abs is not None:
        fig.add_trace(
            go.Scatter(
                x=proj_x_float + proj_x_float[::-1],
                y=list(upper_abs[:, 3]) + list(lower_abs[::-1, 3]),
                fill="toself",
                fillcolor="rgba(88,166,255,0.10)",
                line=dict(width=0),
                name="IC Conformal",
                showlegend=True,
                hoverinfo="skip",
            ),
            row=1, col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=proj_x_float,
            y=proj_abs[:, 3],
            mode="lines",
            line=dict(color=EXPERT_COLORS["moe"], width=2),
            name="Close Projetado (MoE)",
            opacity=0.9,
        ),
        row=1, col=1,
    )
    if n_real > 0:
        fig.add_trace(
            go.Scatter(
                x=proj_x_float[:n_real],
                y=real_ohlc[:n_real, 3],
                mode="lines",
                line=dict(color="#ffd93d", width=2),
                name="Close Real",
                opacity=0.9,
            ),
            row=1, col=1,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # PAINEL 2 — Especialistas individuais vs Real
    # ══════════════════════════════════════════════════════════════════════════

    fig.add_trace(
        go.Scatter(
            x=bar_x,
            y=proj_abs[:, 3],
            mode="lines+markers",
            line=dict(color=EXPERT_COLORS["moe"], width=2.5),
            marker=dict(size=4),
            name="MoE",
        ),
        row=2, col=1,
    )
    if n_real > 0:
        fig.add_trace(
            go.Scatter(
                x=bar_x[:n_real],
                y=real_ohlc[:n_real, 3],
                mode="lines+markers",
                line=dict(color="#ffd93d", width=2),
                marker=dict(symbol="square", size=4),
                name="Real (p2)",
                showlegend=True,
            ),
            row=2, col=1,
        )
    for key in ["inception", "lstm", "transformer", "mlp"]:
        if key in ind_preds:
            arr = ind_preds[key]
            c_vals = arr[:, 3] if arr.ndim == 2 else arr[:horizon, 3]
            fig.add_trace(
                go.Scatter(
                    x=bar_x,
                    y=c_vals,
                    mode="lines",
                    line=dict(color=EXPERT_COLORS[key], width=1, dash="dash"),
                    opacity=0.65,
                    name=EXPERT_LABELS[key],
                ),
                row=2, col=1,
            )

    fig.add_hline(
        y=base_price, row=2, col=1,
        line=dict(color="#8b949e", width=0.8, dash="dot"),
        opacity=0.5,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # PAINEL 3 — Erro por barra
    # ══════════════════════════════════════════════════════════════════════════

    if n_real > 0:
        errors = proj_abs[:n_real, 3] - real_ohlc[:n_real, 3]
        bar_colors = [
            "#26a69a" if e >= 0 else "#ef5350" for e in errors
        ]
        fig.add_trace(
            go.Bar(
                x=bar_x[:n_real],
                y=errors,
                marker_color=bar_colors,
                opacity=0.75,
                name="Erro por Barra",
                showlegend=False,
            ),
            row=3, col=1,
        )

        if upper_abs is not None and lower_abs is not None:
            ci_half = (upper_abs[:n_real, 3] - lower_abs[:n_real, 3]) / 2
            fig.add_trace(
                go.Scatter(
                    x=list(bar_x[:n_real]) + list(bar_x[:n_real][::-1]),
                    y=list(ci_half) + list(-ci_half[::-1]),
                    fill="toself",
                    fillcolor="rgba(88,166,255,0.12)",
                    line=dict(width=0),
                    name="CI Conformal (p3)",
                    hoverinfo="skip",
                ),
                row=3, col=1,
            )

    fig.add_hline(
        y=0, row=3, col=1,
        line=dict(color="#8b949e", width=0.8),
        opacity=0.5,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # PAINEL 4 — Pesos do Gating
    # ══════════════════════════════════════════════════════════════════════════

    colors_g = [EXPERT_COLORS[k] for k in ["inception", "lstm", "transformer", "mlp"]]

    if fold_met:
        folds_x  = [m["fold"] for m in fold_met]
        gw_arr   = np.array([m["gating_weights"] for m in fold_met])
        for i, (lbl, col) in enumerate(zip(gat_labels, colors_g)):
            fig.add_trace(
                go.Scatter(
                    x=folds_x,
                    y=gw_arr[:, i],
                    mode="lines+markers",
                    line=dict(color=col, width=2),
                    marker=dict(size=6),
                    name=lbl,
                ),
                row=4, col=1,
            )
    else:
        fig.add_trace(
            go.Bar(
                x=gat_labels,
                y=gat_w,
                marker_color=colors_g,
                opacity=0.85,
                name="Pesos Gating",
                text=[f"{w:.3f}" for w in gat_w],
                textposition="outside",
                textfont=dict(color="#f0f6fc", size=10),
            ),
            row=4, col=1,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Layout global
    # ══════════════════════════════════════════════════════════════════════════

    fig.update_layout(
        height=1050,
        width=1400,
        paper_bgcolor=PAPER,
        plot_bgcolor=BG,
        font=dict(color=TICK_C),
        title=dict(
            text=(
                "MoE Gating Network │ Soft-DTW + Curvature Penalty │ "
                "EMA(10)[Hurst, GK_Vol] → Gating → Weighted Output"
            ),
            font=dict(size=13, color="#8b949e"),
            x=0.5,
        ),
        legend=dict(
            bgcolor="#161b22",
            bordercolor="#30363d",
            borderwidth=1,
            font=dict(color="#c9d1d9", size=9),
        ),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
    )

    for i in range(1, 5):
        fig.update_xaxes(**AXIS_DEFAULTS, row=i, col=1)
        fig.update_yaxes(**AXIS_DEFAULTS, row=i, col=1)

    fig.update_yaxes(title_text="Preço",           title_font=dict(color=TICK_C, size=11), row=1, col=1)
    fig.update_yaxes(title_text="Close",           title_font=dict(color=TICK_C, size=10), row=2, col=1)
    fig.update_yaxes(title_text="Erro (preço)",    title_font=dict(color=TICK_C, size=10), row=3, col=1)
    fig.update_yaxes(title_text="Peso (Softmax)",  title_font=dict(color=TICK_C, size=10), row=4, col=1)
    fig.update_xaxes(title_text="Fold (Walk-Forward)", title_font=dict(color=TICK_C, size=10), row=4, col=1)

    fig.update_layout(xaxis_rangeslider_visible=False)

    for ann in fig.layout.annotations:
        ann.update(font=dict(color="#c9d1d9", size=12))

    if save_path:
        if save_path.endswith(".html"):
            fig.write_html(save_path)
        else:
            fig.write_image(save_path, scale=2)
        print(f"📊 Gráfico salvo: {save_path}")

    return fig


# =============================================================================
# CORES ADICIONAIS — Daily Comparison
# =============================================================================

CANDLE_BULL = "#26a69a"
CANDLE_BEAR = "#ef5350"
MOE_BULL    = "#4fc3f7"
MOE_BEAR    = "#ff8a65"
MOE_PATH    = "#fbbf24"


# =============================================================================
# CLASSE: DailyMoEComparer — Comparação Tick-to-Daily
# =============================================================================


class DailyMoEComparer:
    """
    Gera gráfico Plotly comparativo: Candles Diários Reais vs Projeção MoE.

    Painéis
    -------
    1. Candlestick real + "Shadow Candles" da projeção MoE + Expected Path
    2. Heatmap de pesos do Gating por dia (se disponível)
    """

    def __init__(self, aggregator=None):
        self.aggregator = aggregator

    def plot(
        self,
        daily_real: pd.DataFrame,
        daily_proj: pd.DataFrame,
        gate_weights=None,
        gate_labels=None,
        n_history: int = 30,
        title: str = "USD/JPY — Candles Diários vs Projeção MoE",
        save_path: Optional[str] = "moe_daily_comparison.html",
    ) -> go.Figure:
        if gate_labels is None:
            gate_labels = ["InceptionTime", "LSTM", "Transformer", "ResidualMLP"]

        has_gating = gate_weights is not None and len(gate_weights) > 0
        n_rows = 2 if has_gating else 1
        row_heights = [0.70, 0.30] if has_gating else [1.0]

        fig = make_subplots(
            rows=n_rows, cols=1,
            row_heights=row_heights,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=(
                [title, "Pesos do Gating (Diário)"] if has_gating else [title]
            ),
        )

        hist = daily_real.tail(n_history)

        # ── Candles reais ────────────────────────────────────────────────
        fig.add_trace(
            go.Candlestick(
                x=hist.index,
                open=hist["Open"], high=hist["High"],
                low=hist["Low"],  close=hist["Close"],
                name="Real (Diário)",
                increasing_line_color=CANDLE_BULL,
                decreasing_line_color=CANDLE_BEAR,
                increasing_fillcolor=CANDLE_BULL,
                decreasing_fillcolor=CANDLE_BEAR,
                opacity=0.90, showlegend=True,
            ),
            row=1, col=1,
        )

        # ── Divisor Histórico / Projeção ─────────────────────────────────
        if len(hist) > 0 and len(daily_proj) > 0:
            dt_diff = pd.Timedelta(daily_proj.index[0] - hist.index[-1])
            div_date = pd.Timestamp(hist.index[-1]) + (dt_diff / 2)
            # Plotly internamente faz sum([x]) → precisa de int (Unix ms)
            div_date_ms = int(div_date.timestamp() * 1000)
            fig.add_vline(
                x=div_date_ms, row=1, col=1,
                line=dict(color="#58a6ff", width=1.8, dash="dash"),
                opacity=0.75,
                annotation_text="AGORA →",
                annotation_font=dict(color="#58a6ff", size=10),
                annotation_position="bottom right",
            )

        # ── Shadow Candles (Projeção MoE) ────────────────────────────────
        if len(daily_proj) > 0:
            fig.add_trace(
                go.Candlestick(
                    x=daily_proj.index,
                    open=daily_proj["Open"], high=daily_proj["High"],
                    low=daily_proj["Low"],  close=daily_proj["Close"],
                    name="Projeção MoE (Diário)",
                    increasing_line_color=MOE_BULL,
                    decreasing_line_color=MOE_BEAR,
                    increasing_fillcolor=MOE_BULL,
                    decreasing_fillcolor=MOE_BEAR,
                    opacity=0.55, showlegend=True,
                ),
                row=1, col=1,
            )

            # ── Expected Path ─────────────────────────────────────────────
            fig.add_trace(
                go.Scatter(
                    x=daily_proj.index, y=daily_proj["Close"],
                    mode="lines+markers",
                    line=dict(color=MOE_PATH, width=2.5, dash="dot"),
                    marker=dict(size=6, symbol="diamond"),
                    name="Expected Path (Close MoE)", opacity=0.9,
                ),
                row=1, col=1,
            )

            # ── Conexão último real → primeiro projetado ──────────────────
            if len(hist) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=[hist.index[-1], daily_proj.index[0]],
                        y=[hist["Close"].iloc[-1], daily_proj["Open"].iloc[0]],
                        mode="lines",
                        line=dict(color="#58a6ff", width=1.2, dash="dot"),
                        showlegend=False, hoverinfo="skip",
                    ),
                    row=1, col=1,
                )

        # ── Anotação de tendência ─────────────────────────────────────────
        if len(daily_proj) > 0 and len(hist) > 0:
            base_p = hist["Close"].iloc[-1]
            final_p = daily_proj["Close"].iloc[-1]
            delta_pct = (final_p - base_p) / base_p * 100
            direction = "↑ ALTA" if delta_pct > 0 else "↓ BAIXA"
            fig.add_annotation(
                x=daily_proj.index[-1], y=final_p,
                text=(f"<b>{direction}</b><br>{delta_pct:+.3f}%<br>{len(daily_proj)}d"),
                showarrow=True, arrowhead=2, arrowcolor=MOE_PATH,
                font=dict(color=TITLE_C, size=10),
                bgcolor="rgba(22,27,34,0.85)", bordercolor="#30363d",
                row=1, col=1,
            )

        # ── Painel 2: Gating por dia ──────────────────────────────────────
        if has_gating:
            import numpy as _np
            gw = _np.array(gate_weights)
            n_days_gw = min(len(gw), len(daily_proj))
            if n_days_gw > 0:
                gw_days = gw[:n_days_gw]
                dates_gw = daily_proj.index[:n_days_gw]
                colors_g = [
                    EXPERT_COLORS["inception"], EXPERT_COLORS["lstm"],
                    EXPERT_COLORS["transformer"], EXPERT_COLORS["mlp"],
                ]
                for i, (label, color) in enumerate(zip(gate_labels, colors_g)):
                    fig.add_trace(
                        go.Bar(
                            x=dates_gw, y=gw_days[:, i],
                            name=label, marker_color=color, opacity=0.85,
                            text=[f"{w:.2f}" for w in gw_days[:, i]],
                            textposition="inside",
                            textfont=dict(size=8, color=TITLE_C),
                        ),
                        row=2, col=1,
                    )
                fig.update_layout(barmode="stack")

        # ── Layout global ─────────────────────────────────────────────────
        fig.update_layout(
            height=750 if has_gating else 550, width=1300,
            paper_bgcolor=PAPER, plot_bgcolor=BG,
            font=dict(color=TICK_C),
            title=dict(
                text="Tick-to-Daily Consolidation │ MoE Gating Network │ USD/JPY",
                font=dict(size=14, color="#8b949e"), x=0.5,
            ),
            legend=dict(
                bgcolor="#161b22", bordercolor="#30363d", borderwidth=1,
                font=dict(color="#c9d1d9", size=9),
                orientation="h", y=-0.08,
            ),
            hovermode="x unified",
            xaxis_rangeslider_visible=False,
        )

        for i in range(1, n_rows + 1):
            fig.update_xaxes(**AXIS_DEFAULTS, row=i, col=1)
            fig.update_yaxes(**AXIS_DEFAULTS, row=i, col=1)

        fig.update_yaxes(
            title_text="Preço", title_font=dict(color=TICK_C, size=11),
            row=1, col=1,
        )
        if has_gating:
            fig.update_yaxes(
                title_text="Peso (Softmax)", title_font=dict(color=TICK_C, size=10),
                row=2, col=1,
            )

        for ann in fig.layout.annotations:
            ann.update(font=dict(color="#c9d1d9", size=12))

        fig.update_layout(xaxis_rangeslider_visible=False)

        if save_path:
            if save_path.endswith(".html"):
                fig.write_html(save_path)
            else:
                fig.write_image(save_path, scale=2)
            print(f"  📊 Gráfico salvo: {save_path}")

        fig.show()
        return fig


# =============================================================================
# ANÁLISE MORFOLÓGICA
# =============================================================================


def _compute_morphology_match(
    daily_real: pd.DataFrame,
    daily_proj: pd.DataFrame,
) -> dict:
    """Compara tendência macro da projeção MoE com os candles reais."""
    import numpy as _np

    overlap = daily_real.index.intersection(daily_proj.index)
    result = {
        "overlap_days": len(overlap),
        "trend_match": False,
        "mae_close": float("nan"),
        "direction_accuracy": float("nan"),
    }

    if len(overlap) == 0:
        if len(daily_real) > 0 and len(daily_proj) > 0:
            real_trend = daily_real["Close"].iloc[-1] > daily_real["Open"].iloc[-1]
            proj_trend = daily_proj["Close"].iloc[-1] > daily_proj["Open"].iloc[0]
            result["trend_match"] = real_trend == proj_trend
        return result

    real_ov = daily_real.loc[overlap]
    proj_ov = daily_proj.loc[overlap]

    result["mae_close"] = float(
        _np.mean(_np.abs(real_ov["Close"].values - proj_ov["Close"].values))
    )

    real_dir = _np.sign(real_ov["Close"].values - real_ov["Open"].values)
    proj_dir = _np.sign(proj_ov["Close"].values - proj_ov["Open"].values)
    result["direction_accuracy"] = float(_np.mean(real_dir == proj_dir))

    real_trend = real_ov["Close"].iloc[-1] > real_ov["Open"].iloc[0]
    proj_trend = proj_ov["Close"].iloc[-1] > proj_ov["Open"].iloc[0]
    result["trend_match"] = bool(real_trend == proj_trend)

    return result


def _print_daily_report(
    daily_real: pd.DataFrame,
    daily_proj: pd.DataFrame,
    morph: dict,
) -> None:
    """Imprime relatório da consolidação diária."""
    print(f"\n{'═' * 65}")
    print("  TICK-TO-DAILY — RELATÓRIO DE CONSOLIDAÇÃO")
    print(f"{'═' * 65}")
    print(f"  Candles diários reais:     {len(daily_real)}")
    print(f"  Candles diários projetados: {len(daily_proj)}")
    print(f"  Dias sobrepostos:          {morph['overlap_days']}")

    if morph["overlap_days"] > 0:
        print(f"  MAE Close (diário):        {morph['mae_close']:.5f}")
        print(f"  Acurácia Direcional:       {morph['direction_accuracy']:.0%}")

    trend_icon = "✅" if morph["trend_match"] else "❌"
    print(f"  Tendência Macro confirma:  {trend_icon} {morph['trend_match']}")

    if len(daily_proj) > 0 and len(daily_real) > 0:
        base = daily_real["Close"].iloc[-1]
        target = daily_proj["Close"].iloc[-1]
        delta = (target - base) / base * 100
        direction = "ALTA ↑" if delta > 0 else "BAIXA ↓"
        print(f"  Projeção macro:            {direction} ({delta:+.3f}%)")
        print(f"  Período projetado:         {daily_proj.index[0].date()} → "
              f"{daily_proj.index[-1].date()} ({len(daily_proj)}d)")

    print(f"{'═' * 65}")


# =============================================================================
# ORQUESTRADOR: run_daily_comparison
# =============================================================================


def run_daily_comparison(
    df_tick_bars: pd.DataFrame,
    moe_results: dict,
    timestamp_col: str = "Date",
    n_history_days: int = 30,
    min_ticks_per_candle: int = 10,
    save_path: str = "moe_daily_comparison.html",
) -> dict:
    """
    Pipeline completo: tick bars → daily candles → comparação visual com MoE.

    Parameters
    ----------
    df_tick_bars : pd.DataFrame
        Tick bars com OHLC e timestamps.
    moe_results : dict
        Output de run_moe_pipeline() (projected_ohlc, gating_weights_final, etc.)
    timestamp_col : str
        Coluna de timestamp nos tick bars.
    n_history_days : int
        Número de candles diários históricos a plotar.
    min_ticks_per_candle : int
        Mínimo de ticks para formar um candle válido.
    save_path : str
        Caminho para salvar a figura HTML.

    Returns
    -------
    dict
        daily_real, daily_proj, figure, morphology_match.
    """
    import numpy as _np
    # Import lazy para evitar circular import (moe_to_daily importa moe_visualization
    # apenas no bloco __main__)
    from moe_to_daily import TimeSeriesAggregator

    print("═" * 65)
    print("  📊 TICK-TO-DAILY — CONSOLIDAÇÃO + COMPARAÇÃO MoE")
    print("═" * 65)

    aggregator = TimeSeriesAggregator(min_ticks_per_candle=min_ticks_per_candle)

    # 1. Agregar tick bars reais → diário
    print("  Agregando tick bars → candles diários...")
    daily_real = aggregator.aggregate_ticks_to_daily(
        df_tick_bars, timestamp_col=timestamp_col
    )
    print(f"  ✅ {len(daily_real)} candles diários reais formados")

    # 2. Estimar timestamps para a projeção MoE
    proj_ohlc = moe_results["projected_ohlc"]
    horizon = len(proj_ohlc)

    if timestamp_col in df_tick_bars.columns:
        ts_series = pd.to_datetime(df_tick_bars[timestamp_col])
    elif isinstance(df_tick_bars.index, pd.DatetimeIndex):
        ts_series = df_tick_bars.index.to_series()
    else:
        raise ValueError("Não foi possível identificar a coluna de timestamps.")

    med_diff_val = ts_series.diff().median()
    avg_bar_duration = (
        pd.Timedelta(minutes=5) if pd.isna(med_diff_val)
        else pd.Timedelta(med_diff_val)
    )
    last_ts = ts_series.iloc[-1]
    print(f"  Duração média entre tick bars: {avg_bar_duration}")

    proj_timestamps = TimeSeriesAggregator.estimate_projection_timestamps(
        last_timestamp=pd.Timestamp(last_ts),
        horizon=horizon,
        avg_bar_duration=avg_bar_duration,
    )

    # 3. Agregar projeção MoE → diário
    print("  Convertendo projeção MoE (tick) → candles diários...")
    daily_proj = aggregator.aggregate_projection_to_daily(
        proj_ohlc=proj_ohlc, timestamps=proj_timestamps
    )
    print(f"  ✅ {len(daily_proj)} candles diários projetados formados")

    # 4. Gating weights por dia
    gw_series = moe_results.get("gating_weights_series")
    gate_weights_daily = None

    if gw_series is not None and len(gw_series) == horizon:
        gw_df = pd.DataFrame(
            gw_series,
            columns=["w_inc", "w_lstm", "w_tfm", "w_mlp"],
            index=proj_timestamps,
        )
        if gw_df.index.tz is None:
            gw_df.index = gw_df.index.tz_localize("UTC")
        gw_df.index = gw_df.index.tz_convert("US/Eastern")
        gw_df.index = gw_df.index - pd.Timedelta(hours=17)
        gw_daily = gw_df.resample("1D").mean()
        gw_daily.index = gw_daily.index + pd.Timedelta(hours=17)
        gw_daily.index = gw_daily.index.normalize()
        gw_daily = gw_daily.dropna()
        gate_weights_daily = gw_daily.values
    else:
        gw_final = moe_results.get("gating_weights_final")
        if gw_final is not None:
            gate_weights_daily = _np.tile(gw_final, (len(daily_proj), 1))

    # 5. Análise morfológica
    morphology_match = _compute_morphology_match(daily_real, daily_proj)

    # 6. Plot
    comparer = DailyMoEComparer(aggregator=aggregator)
    fig = comparer.plot(
        daily_real=daily_real,
        daily_proj=daily_proj,
        gate_weights=gate_weights_daily,
        gate_labels=moe_results.get(
            "gating_labels",
            ["InceptionTime", "LSTM", "Transformer", "ResidualMLP"],
        ),
        n_history=n_history_days,
        save_path=save_path,
    )

    # 7. Relatório
    _print_daily_report(daily_real, daily_proj, morphology_match)

    return {
        "daily_real": daily_real,
        "daily_proj": daily_proj,
        "figure": fig,
        "morphology_match": morphology_match,
    }


# =============================================================================
# ROLLING BACKTEST: PROJEÇÃO MoE vs REALIDADE — ÚLTIMOS 30 DIAS
# =============================================================================


def run_rolling_backtest_30d(
    df_data: pd.DataFrame,
    model_path: str = '../models/EURUSD/moe_model.keras',
    config_path: str = '../models/EURUSD/moe_config.json',
    n_days: int = 30,
    timestamp_col: str = 'Date',
    save_path: str = '../exports/plots/moe_rolling_30d.html',
) -> dict:
    """
    Executa backtest rolling dos últimos ``n_days`` dias, comparando a
    projeção MoE (1-step-ahead por dia) com o que realmente aconteceu.

    Pipeline
    --------
    1. Carrega o modelo MoE treinado (.keras)
    2. Prepara sequências com ``prepare_moe_data``
    3. Para cada um dos últimos N pontos de dados que correspondem a um
       novo dia calendário, executa inferência e coleta:
       - Close projetado (passo 1 do horizonte)
       - Close real (próxima barra)
       - Pesos do Gating nesse ponto
    4. Agrega por dia e plota comparação de 30 dias

    Parameters
    ----------
    df_data : pd.DataFrame
        DataFrame com tick bars (OHLC + features + timestamps).
    model_path : str
        Caminho para o modelo .keras treinado.
    config_path : str
        Caminho para o moe_config.json.
    n_days : int
        Número de dias a comparar (default=30).
    timestamp_col : str
        Nome da coluna de timestamp.
    save_path : str
        Caminho para salvar o HTML do gráfico.

    Returns
    -------
    dict
        daily_projected, daily_real, metrics, figure
    """
    import tensorflow as tf
    from moe_gating import (
        MoEForecaster, MoEConfig, prepare_moe_data,
        hybrid_financial_loss, morphological_loss, _find_regime_col_indices,
    )
    from llm import relative_to_absolute

    print("\n" + "═" * 65)
    print("  📊 ROLLING BACKTEST — MoE vs REALIDADE (últimos 30 dias)")
    print("═" * 65)

    # ── 1. Carregar config ───────────────────────────────────────────────
    config_data = {}
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

    window_size = config_data.get('window_size', 60)
    horizon = config_data.get('horizon', 15)

    # ── 2. Preparar dados ────────────────────────────────────────────────
    df = df_data.copy()
    if 'Returns' not in df.columns:
        df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df.dropna(inplace=True)

    X_seq, y_targets, base_prices, feat_names = prepare_moe_data(
        df, window_size=window_size, horizon=horizon
    )
    n_samples = len(X_seq)
    print(f"  Sequências totais: {n_samples} | Window: {window_size} | Horizon: {horizon}")

    # ── 3. Carregar modelo treinado ──────────────────────────────────────
    if not os.path.exists(model_path):
        print(f"  ❌ Modelo '{model_path}' não encontrado.")
        print(f"  Execute moe_to_daily.py primeiro para treinar.")
        return {}

    print(f"  Carregando modelo: {model_path}...")

    # Registrar loss customizada e layers customizadas
    from moe_gating import (
        InceptionBlock, CausalSelfAttentionMoE,
        ResidualMLPBlock, RegimeGatingNetwork,
    )
    custom_objects = {
        'hybrid_financial_loss': hybrid_financial_loss,
        'morphological_loss': morphological_loss,
        'InceptionBlock': InceptionBlock,
        'CausalSelfAttentionMoE': CausalSelfAttentionMoE,
        'ResidualMLPBlock': ResidualMLPBlock,
        'RegimeGatingNetwork': RegimeGatingNetwork,
        'MoEForecaster': MoEForecaster,
    }

    # Tentar carregar; se falhar com custom loss, compilar manualmente
    try:
        moe_model = tf.keras.models.load_model(
            model_path, custom_objects=custom_objects, compile=False
        )
        # Recompilar com a loss híbrida financeira
        alpha_val = config_data.get('loss_alpha', 0.5)
        beta_val = config_data.get('loss_beta', 0.2)
        gamma_val = config_data.get('loss_gamma', 1.0)
        delta_val = config_data.get('huber_delta', 1.0)
        moe_model.compile(
            optimizer='adam',
            loss=lambda y_t, y_p: hybrid_financial_loss(
                y_t, y_p, alpha_val, beta_val, gamma_val, delta_val
            ),
        )
        print(f"  ✅ Modelo carregado com sucesso.")
    except Exception as e:
        print(f"  ❌ Falha ao carregar modelo: {e}")
        return {}

    # ── 4. Identificar pontos de dados correspondentes a dias distintos ──
    # Precisamos mapear cada sequência a um dia calendário.
    # O base_price de cada sequência corresponde ao Close da última barra
    # da janela, e o target da sequência são as próximas H barras.
    # Usamos timestamps se disponíveis, caso contrário usamos índices.

    # Calcular quantas sequências precisamos para cobrir n_days de dias
    # Cada sequência i tem target no intervalo [i + window_size, i + window_size + horizon)
    # Precisamos de sequências cujos targets (ao menos step 0) sejam distintos em dias.

    has_timestamps = False
    if timestamp_col in df_data.columns:
        ts_series = pd.to_datetime(df_data[timestamp_col])
        has_timestamps = True
    elif isinstance(df_data.index, pd.DatetimeIndex):
        ts_series = df_data.index.to_series().reset_index(drop=True)
        has_timestamps = True

    # Obter as datas dos targets (step 0 de cada sequência)
    df_reset = df.reset_index(drop=True)

    if has_timestamps:
        if timestamp_col in df_reset.columns:
            all_ts = pd.to_datetime(df_reset[timestamp_col])
        elif isinstance(df_data.index, pd.DatetimeIndex):
            all_ts = pd.Series(df_data.index).reset_index(drop=True)
            # Ajustar tamanho após dropna
            all_ts = all_ts.iloc[:len(df_reset)]
        else:
            all_ts = None
            has_timestamps = False

    # Para cada sequência i, o target step 0 está no índice (i + window_size) dos dados
    target_indices = np.arange(window_size, window_size + n_samples)

    if has_timestamps and all_ts is not None and len(all_ts) > 0:
        # Mapear cada sequência a um dia
        target_dates = all_ts.iloc[target_indices].dt.date.values
        unique_dates = np.unique(target_dates)

        # Pegar os últimos n_days dias únicos
        last_n_dates = unique_dates[-n_days:]
        print(f"  Dias únicos disponíveis: {len(unique_dates)} | Selecionados: {len(last_n_dates)}")

        # Para cada dia, pegar a ÚLTIMA sequência daquele dia
        # (mais informativa — usa dados mais recentes)
        day_seq_indices = []
        for d in last_n_dates:
            mask = (target_dates == d)
            indices = np.where(mask)[0]
            if len(indices) > 0:
                day_seq_indices.append(indices[-1])  # última sequência do dia

        day_seq_indices = np.array(day_seq_indices)
    else:
        # Sem timestamps: pegar últimas n_samples uniformemente espaçadas
        # Assumir que o dataset tem ~X barras por dia
        # Heurística: pegar uma sequência a cada (total / n_days) posições
        step = max(1, n_samples // (n_days * 3))
        day_seq_indices = np.arange(
            max(0, n_samples - n_days * step),
            n_samples,
            step
        )[-n_days:]
        print(f"  Sem timestamps — espaçamento heurístico: {step} barras/ponto")

    n_points = len(day_seq_indices)
    print(f"  Pontos de comparação: {n_points}")

    if n_points == 0:
        print("  ❌ Dados insuficientes para comparação.")
        return {}

    # ── 5. Inferência rolling ────────────────────────────────────────────
    print("  Executando inferência rolling...")

    daily_results = []
    batch_size = 32

    for batch_start in range(0, n_points, batch_size):
        batch_end = min(batch_start + batch_size, n_points)
        batch_indices = day_seq_indices[batch_start:batch_end]

        X_batch = X_seq[batch_indices]
        y_batch = y_targets[batch_indices]
        bp_batch = base_prices[batch_indices]

        # Projeção MoE
        proj_rel = moe_model(X_batch, training=False).numpy()

        # Pesos do gating
        try:
            gating_w = moe_model.get_gating_weights(X_batch)
        except Exception:
            gating_w = np.ones((len(batch_indices), 4)) / 4

        for j, seq_idx in enumerate(batch_indices):
            base_p = float(bp_batch[j])

            # Converter projeção para absoluto
            proj_abs = relative_to_absolute(proj_rel[j], base_p)  # (H, 4)
            real_abs = relative_to_absolute(y_batch[j], base_p)    # (H, 4)

            # Pegar step 0 (próxima barra) da projeção e do real
            proj_close_step0 = float(proj_abs[0, 3])  # Close projetado
            real_close_step0 = float(real_abs[0, 3])   # Close real

            proj_ohlc_step0 = proj_abs[0]  # [O, H, L, C]
            real_ohlc_step0 = real_abs[0]  # [O, H, L, C]

            # Se temos horizonte >= 1, podemos pegar todas as H barras
            # projetadas para comparar com as H barras reais
            entry = {
                'seq_idx': int(seq_idx),
                'base_price': base_p,
                'proj_close': proj_close_step0,
                'real_close': real_close_step0,
                'proj_ohlc': proj_ohlc_step0.tolist(),
                'real_ohlc': real_ohlc_step0.tolist(),
                'error': proj_close_step0 - real_close_step0,
                'error_pct': (proj_close_step0 - real_close_step0) / real_close_step0 * 100,
                'direction_correct': (
                    (proj_close_step0 > base_p) == (real_close_step0 > base_p)
                ),
                'gating_weights': gating_w[j].tolist(),
            }

            # Timestamp se disponível
            if has_timestamps and all_ts is not None:
                tidx = seq_idx + window_size
                if tidx < len(all_ts):
                    entry['date'] = all_ts.iloc[tidx]

            daily_results.append(entry)

    # ── 6. Construir DataFrame de resultados ─────────────────────────────
    df_backtest = pd.DataFrame(daily_results)

    # Métricas globais
    mae_close = np.mean(np.abs(df_backtest['error'].values))
    rmse_close = np.sqrt(np.mean(df_backtest['error'].values ** 2))
    dir_acc = df_backtest['direction_correct'].mean()
    mean_err_pct = np.mean(np.abs(df_backtest['error_pct'].values))

    print(f"\n  ═══ MÉTRICAS DO BACKTEST ROLLING ({n_points} pontos) ═══")
    print(f"  MAE Close:             {mae_close:.5f}")
    print(f"  RMSE Close:            {rmse_close:.5f}")
    print(f"  MAPE Close:            {mean_err_pct:.3f}%")
    print(f"  Acurácia Direcional:   {dir_acc:.0%}")
    print(f"  {'═' * 50}")

    metrics = {
        'mae_close': mae_close,
        'rmse_close': rmse_close,
        'mape_close': mean_err_pct,
        'direction_accuracy': dir_acc,
        'n_points': n_points,
    }

    # ── 7. Plot ──────────────────────────────────────────────────────────
    fig = plot_rolling_30d_comparison(
        df_backtest=df_backtest,
        metrics=metrics,
        save_path=save_path,
    )

    return {
        'df_backtest': df_backtest,
        'metrics': metrics,
        'figure': fig,
    }


def plot_rolling_30d_comparison(
    df_backtest: pd.DataFrame,
    metrics: dict,
    save_path: Optional[str] = '../exports/plots/moe_rolling_30d.html',
) -> go.Figure:
    """
    Plota a comparação rolling de 30 dias: Projeção MoE vs Realidade.

    Painéis
    -------
    1. Close Projetado vs Close Real (linhas sobrepostas)
    2. Candles OHLC lado a lado (Projeção vs Real)
    3. Erro por dia (barras) + MAE acumulado (linha)
    4. Pesos do Gating por dia (barras empilhadas)
    """
    n_points = len(df_backtest)
    has_dates = 'date' in df_backtest.columns

    if has_dates:
        x_axis = pd.to_datetime(df_backtest['date'])
        x_label = "Data"
    else:
        x_axis = np.arange(1, n_points + 1)
        x_label = "Ponto de Comparação"

    # Métricas para título
    mae_c = metrics.get('mae_close', 0)
    dir_acc = metrics.get('direction_accuracy', 0)
    mape = metrics.get('mape_close', 0)

    title_p1 = (
        f"Close: Projeção MoE vs Real │ MAE={mae_c:.5f} │ "
        f"Dir={dir_acc:.0%} │ MAPE={mape:.3f}%"
    )

    fig = make_subplots(
        rows=4, cols=1,
        row_heights=[0.32, 0.28, 0.20, 0.20],
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            title_p1,
            "Candles OHLC — Projeção (azul) vs Real (verde)",
            "Erro por Dia (Projetado − Real) + MAE Acumulado",
            "Pesos do Gating — Especialização por Regime",
        ),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # PAINEL 1 — Close Projetado vs Close Real (linhas)
    # ══════════════════════════════════════════════════════════════════════════

    proj_close = df_backtest['proj_close'].values
    real_close = df_backtest['real_close'].values

    # Linha Real
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=real_close,
            mode="lines+markers",
            line=dict(color="#26a69a", width=2.5),
            marker=dict(size=5, symbol="circle"),
            name="Close Real",
            opacity=0.95,
        ),
        row=1, col=1,
    )

    # Linha Projeção MoE
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=proj_close,
            mode="lines+markers",
            line=dict(color=EXPERT_COLORS['moe'], width=2.5, dash="dot"),
            marker=dict(size=5, symbol="diamond"),
            name="Close Projetado (MoE)",
            opacity=0.95,
        ),
        row=1, col=1,
    )

    # Banda de erro (fill between)
    fig.add_trace(
        go.Scatter(
            x=pd.concat([pd.Series(x_axis), pd.Series(x_axis[::-1])]) if has_dates
              else np.concatenate([x_axis, x_axis[::-1]]),
            y=np.concatenate([
                np.maximum(proj_close, real_close),
                np.minimum(proj_close, real_close)[::-1]
            ]),
            fill="toself",
            fillcolor="rgba(251,191,36,0.08)",
            line=dict(width=0),
            name="Zona de Erro",
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1, col=1,
    )

    # Marcar acertos direcionais
    for i in range(n_points):
        if df_backtest['direction_correct'].iloc[i]:
            fig.add_trace(
                go.Scatter(
                    x=[x_axis.iloc[i] if has_dates else x_axis[i]],
                    y=[real_close[i]],
                    mode="markers",
                    marker=dict(
                        size=10, symbol="circle-open",
                        color="#26a69a", line=dict(width=2, color="#26a69a")
                    ),
                    showlegend=(i == 0),
                    name="Direção ✓" if i == 0 else None,
                    hoverinfo="skip",
                ),
                row=1, col=1,
            )

    # ══════════════════════════════════════════════════════════════════════════
    # PAINEL 2 — Candles OHLC lado a lado
    # ══════════════════════════════════════════════════════════════════════════

    proj_ohlc = np.array(df_backtest['proj_ohlc'].tolist())  # (N, 4)
    real_ohlc = np.array(df_backtest['real_ohlc'].tolist())  # (N, 4)

    if has_dates:
        # Offset para side-by-side candles
        dt_half = pd.Timedelta(hours=6)
        x_proj = x_axis + dt_half
        x_real = x_axis + dt_half
    else:
        x_proj = x_axis + 0.15
        x_real = x_axis + 0.15

    # Candles reais
    fig.add_trace(
        go.Candlestick(
            x=x_real,
            open=real_ohlc[:, 0], high=real_ohlc[:, 1],
            low=real_ohlc[:, 2], close=real_ohlc[:, 3],
            name="Real (OHLC)",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a",
            decreasing_fillcolor="#ef5350",
            opacity=0.85,
        ),
        row=2, col=1,
    )

    # Candles projeção
    fig.add_trace(
        go.Candlestick(
            x=x_proj,
            open=proj_ohlc[:, 0], high=proj_ohlc[:, 1],
            low=proj_ohlc[:, 2], close=proj_ohlc[:, 3],
            name="Projeção MoE (OHLC)",
            increasing_line_color="#4fc3f7",
            decreasing_line_color="#ff8a65",
            increasing_fillcolor="#4fc3f7",
            decreasing_fillcolor="#ff8a65",
            opacity=0.70,
        ),
        row=2, col=1,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # PAINEL 3 — Erro diário (barras) + MAE acumulado (linha)
    # ══════════════════════════════════════════════════════════════════════════

    errors = df_backtest['error'].values
    bar_colors = ["#26a69a" if e >= 0 else "#ef5350" for e in errors]

    fig.add_trace(
        go.Bar(
            x=x_axis,
            y=errors,
            marker_color=bar_colors,
            opacity=0.75,
            name="Erro (Proj − Real)",
            showlegend=True,
        ),
        row=3, col=1,
    )

    # MAE acumulado (rolling)
    cumulative_mae = np.cumsum(np.abs(errors)) / np.arange(1, n_points + 1)
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=cumulative_mae,
            mode="lines",
            line=dict(color="#f72585", width=2, dash="dash"),
            name="MAE Acumulado",
            yaxis="y2",
        ),
        row=3, col=1,
    )

    # Zero line
    fig.add_hline(
        y=0, row=3, col=1,
        line=dict(color="#8b949e", width=0.8),
        opacity=0.5,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # PAINEL 4 — Pesos do Gating por dia
    # ══════════════════════════════════════════════════════════════════════════

    gating_arr = np.array(df_backtest['gating_weights'].tolist())  # (N, 4)
    gate_labels = ['InceptionTime', 'LSTM', 'Transformer', 'ResidualMLP']
    gate_keys = ['inception', 'lstm', 'transformer', 'mlp']
    colors_g = [EXPERT_COLORS[k] for k in gate_keys]

    for i, (label, color) in enumerate(zip(gate_labels, colors_g)):
        fig.add_trace(
            go.Bar(
                x=x_axis,
                y=gating_arr[:, i],
                name=label,
                marker_color=color,
                opacity=0.85,
                text=[f"{w:.2f}" for w in gating_arr[:, i]],
                textposition="inside",
                textfont=dict(size=7, color=TITLE_C),
            ),
            row=4, col=1,
        )

    fig.update_layout(barmode="stack")

    # ══════════════════════════════════════════════════════════════════════════
    # Layout global
    # ══════════════════════════════════════════════════════════════════════════

    fig.update_layout(
        height=1200,
        width=1400,
        paper_bgcolor=PAPER,
        plot_bgcolor=BG,
        font=dict(color=TICK_C),
        title=dict(
            text=(
                f"Rolling Backtest MoE — Últimos {n_points} dias │ "
                f"MAE={mae_c:.5f} │ Acurácia Direcional={dir_acc:.0%} │ "
                f"MAPE={mape:.3f}%"
            ),
            font=dict(size=14, color="#f0f6fc"),
            x=0.5,
        ),
        legend=dict(
            bgcolor="#161b22",
            bordercolor="#30363d",
            borderwidth=1,
            font=dict(color="#c9d1d9", size=9),
            orientation="h",
            y=-0.05,
        ),
        hovermode="x unified",
    )

    for i in range(1, 5):
        fig.update_xaxes(**AXIS_DEFAULTS, row=i, col=1)
        fig.update_yaxes(**AXIS_DEFAULTS, row=i, col=1)

    fig.update_yaxes(title_text="Close (Preço)", title_font=dict(color=TICK_C, size=11), row=1, col=1)
    fig.update_yaxes(title_text="Preço (OHLC)", title_font=dict(color=TICK_C, size=10), row=2, col=1)
    fig.update_yaxes(title_text="Erro (preço)", title_font=dict(color=TICK_C, size=10), row=3, col=1)
    fig.update_yaxes(title_text="Peso (Softmax)", title_font=dict(color=TICK_C, size=10), row=4, col=1)
    fig.update_xaxes(title_text=x_label, title_font=dict(color=TICK_C, size=10), row=4, col=1)

    # Desativar range sliders dos candlesticks
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
        xaxis3_rangeslider_visible=False,
        xaxis4_rangeslider_visible=False,
    )

    for ann in fig.layout.annotations:
        ann.update(font=dict(color="#c9d1d9", size=12))

    if save_path:
        if save_path.endswith(".html"):
            fig.write_html(save_path)
        else:
            fig.write_image(save_path, scale=2)
        print(f"  📊 Gráfico salvo: {save_path}")

    fig.show()
    return fig


# =============================================================================
# GHOST PROJECTION — PROJEÇÃO FUTURA OUT-OF-SAMPLE
# =============================================================================


def plot_ghost_projection(
    df_history: pd.DataFrame,
    results: Dict[str, Any],
    n_history: int = 60,
    timestamp_col: str = 'Date',
    bars_per_day: int = 50,
    save_path: Optional[str] = '../exports/plots/moe_ghost_projection.html',
    title_suffix: str = "USD/JPY",
) -> go.Figure:
    """
    Plota a projeção futura (out-of-sample) do MoE como "Ghost Candles"
    semitransparentes numa zona além do último dado real.

    O eixo temporal futuro é estimado dinamicamente a partir da frequência
    média das últimas 50 barras, adaptando-se ao regime de atividade do
    mercado no momento da inferência.

    Painéis
    -------
    1. Candlestick Histórico + Ghost Candles (projeção) + CI Conformal
       + Vline "Momento de Inferência" + Anotação de horizonte temporal
    2. Close dos 4 especialistas + MoE ponderado (fase futura)
    3. Pesos do Gating (stacked ou barras)

    Parameters
    ----------
    df_history : pd.DataFrame
        DataFrame com dados históricos reais (OHLC + timestamps).
    results : dict
        Output de load_inference_results() — projected_ohlc, CI bands, etc.
    n_history : int
        Número de barras históricas a exibir antes do momento de inferência.
    timestamp_col : str
        Nome da coluna de timestamp nos dados históricos.
    bars_per_day : int
        Calibração do pipeline (default=50 barras/dia). Usado para estimar
        o horizonte temporal se timestamps não estiverem disponíveis.
    save_path : str, optional
        Caminho para salvar o gráfico (.html).
    title_suffix : str
        Sufixo do título (par de moedas).

    Returns
    -------
    go.Figure
    """
    proj_ohlc    = results['projected_ohlc']          # (H, 4)
    upper_band   = results.get('confidence_upper')     # (H, 4) ou None
    lower_band   = results.get('confidence_lower')     # (H, 4) ou None
    ind_preds    = results.get('individual_predictions', {})
    gating_final = results.get('gating_weights_final', np.ones(4) / 4)
    gat_labels   = results.get('gating_labels',
                                ['InceptionTime', 'LSTM', 'Transformer', 'MLP'])
    base_price   = results.get('base_price', proj_ohlc[0, 0])
    conf_level   = results.get('confidence_level', 0.90)
    horizon      = len(proj_ohlc)

    # ── Dados históricos ────────────────────────────────────────────────
    hist = df_history.tail(n_history).copy()
    hist_ohlc = hist[['Open', 'High', 'Low', 'Close']].values
    n_hist = len(hist_ohlc)

    # ── Detectar timestamps ─────────────────────────────────────────────
    has_timestamps = False
    last_dt = None
    avg_delta = None

    # Tentar coluna explícita
    if timestamp_col in hist.columns:
        ts_series = pd.to_datetime(hist[timestamp_col])
        has_timestamps = True
    elif isinstance(hist.index, pd.DatetimeIndex):
        ts_series = hist.index.to_series()
        has_timestamps = True

    if has_timestamps:
        last_dt = ts_series.iloc[-1]
        # Estimar delta_t a partir das últimas min(50, n) barras
        n_sample = min(50, len(ts_series))
        delta_total = (ts_series.iloc[-1] - ts_series.iloc[-n_sample]).total_seconds()
        avg_delta_sec = delta_total / max(n_sample - 1, 1)
        avg_delta = pd.Timedelta(seconds=avg_delta_sec)

        # Eixo X histórico com timestamps reais
        hist_x = ts_series.values

        # Gerar eixo X futuro
        future_x = pd.DatetimeIndex([
            last_dt + pd.Timedelta(seconds=avg_delta_sec * i)
            for i in range(1, horizon + 1)
        ])

        # Horizonte estimado em horas
        total_horizon_hours = (avg_delta_sec * horizon) / 3600.0

        # Conexão: último ponto real → primeiro ponto projetado
        bridge_x = [last_dt, future_x[0]]
    else:
        # Fallback: eixo numérico
        hist_x = np.arange(n_hist)
        future_x = np.arange(n_hist, n_hist + horizon)
        avg_delta_sec = (24 * 3600) / bars_per_day
        total_horizon_hours = (avg_delta_sec * horizon) / 3600.0
        bridge_x = [n_hist - 1, n_hist]

    # ── Direção e variação ──────────────────────────────────────────────
    direction = "↑ ALTA" if proj_ohlc[-1, 3] > base_price else "↓ BAIXA"
    delta_pct = (proj_ohlc[-1, 3] - base_price) / base_price * 100

    # ══════════════════════════════════════════════════════════════════════
    # LAYOUT
    # ══════════════════════════════════════════════════════════════════════

    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.55, 0.25, 0.20],
        shared_xaxes=False,
        vertical_spacing=0.07,
        subplot_titles=(
            (f"Ghost Projection — {title_suffix}   │   "
             f"Horizonte: {horizon} barras (~{total_horizon_hours:.1f}h)   │   "
             f"{direction} {delta_pct:+.4f}%"),
            "Especialistas — Projeção Futura (Close)",
            "Pesos do Gating — Regime Atual",
        ),
    )

    # ══════════════════════════════════════════════════════════════════════
    # PAINEL 1 — Candlestick Histórico + Ghost Candles + CI
    # ══════════════════════════════════════════════════════════════════════

    # ── Candlestick Histórico ────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=hist_x,
            open=hist_ohlc[:, 0],
            high=hist_ohlc[:, 1],
            low=hist_ohlc[:, 2],
            close=hist_ohlc[:, 3],
            name="Histórico (Real)",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a",
            decreasing_fillcolor="#ef5350",
            opacity=0.90,
            showlegend=True,
        ),
        row=1, col=1,
    )

    # ── Linha Vertical: Momento de Inferência (T₀) ──────────────────────
    if has_timestamps:
        div_x = last_dt
    else:
        div_x = n_hist - 0.5

    fig.add_vline(
        x=div_x, row=1, col=1,
        line=dict(color="#58a6ff", width=2.2, dash="dash"),
        opacity=0.85,
    )

    # Anotação do momento de inferência
    fig.add_annotation(
        x=div_x,
        y=1.02,
        yref="y domain",
        text="<b>T₀ Inferência</b>",
        showarrow=False,
        font=dict(color="#58a6ff", size=11),
        bgcolor="rgba(13,17,23,0.85)",
        bordercolor="#58a6ff",
        borderwidth=1,
        borderpad=4,
        row=1, col=1,
    )

    # ── Sombreamento da Zona Futura (Ghost Zone) ─────────────────────────
    if has_timestamps:
        zone_x0 = last_dt
        zone_x1 = future_x[-1]
    else:
        zone_x0 = n_hist - 0.5
        zone_x1 = n_hist + horizon + 0.5

    fig.add_vrect(
        x0=zone_x0, x1=zone_x1,
        fillcolor="rgba(88,166,255,0.03)",
        line=dict(width=0),
        row=1, col=1,
    )

    # ── Ponte: último Close real → primeiro Open projetado ───────────────
    fig.add_trace(
        go.Scatter(
            x=bridge_x,
            y=[hist_ohlc[-1, 3], proj_ohlc[0, 0]],
            mode="lines",
            line=dict(color="#58a6ff", width=1.5, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1, col=1,
    )

    # ── Ghost Candles (Projeção Futura — semitransparentes) ──────────────
    fig.add_trace(
        go.Candlestick(
            x=future_x,
            open=proj_ohlc[:, 0],
            high=proj_ohlc[:, 1],
            low=proj_ohlc[:, 2],
            close=proj_ohlc[:, 3],
            name="Ghost Projection (MoE)",
            increasing_line_color="#4fc3f7",
            decreasing_line_color="#ff8a65",
            increasing_fillcolor="rgba(79,195,247,0.25)",
            decreasing_fillcolor="rgba(255,138,101,0.25)",
            opacity=0.55,
            showlegend=True,
        ),
        row=1, col=1,
    )

    # ── Ghost Path: Close projetado (linha tracejada âmbar) ──────────────
    fig.add_trace(
        go.Scatter(
            x=future_x,
            y=proj_ohlc[:, 3],
            mode="lines+markers",
            line=dict(color=EXPERT_COLORS['moe'], width=2.5, dash="dot"),
            marker=dict(size=5, symbol="diamond"),
            name="Ghost Path (Close MoE)",
            opacity=0.90,
        ),
        row=1, col=1,
    )

    # ── Intervalo de Confiança Conformal ─────────────────────────────────
    if upper_band is not None and lower_band is not None:
        ci_label = f"IC Conformal ({conf_level:.0%})"

        # Banda externa (High / Low)
        future_x_list = list(future_x)
        fig.add_trace(
            go.Scatter(
                x=future_x_list + future_x_list[::-1],
                y=list(upper_band[:, 1]) + list(lower_band[::-1, 2]),
                fill="toself",
                fillcolor="rgba(88,166,255,0.04)",
                line=dict(width=0),
                name=ci_label + " (H/L)",
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1, col=1,
        )

        # Banda interna (Close)
        fig.add_trace(
            go.Scatter(
                x=future_x_list + future_x_list[::-1],
                y=list(upper_band[:, 3]) + list(lower_band[::-1, 3]),
                fill="toself",
                fillcolor="rgba(88,166,255,0.10)",
                line=dict(width=0.5, color="rgba(88,166,255,0.3)"),
                name=ci_label,
                hoverinfo="skip",
                showlegend=True,
            ),
            row=1, col=1,
        )

        # Linhas de borda do CI (superior e inferior do Close)
        fig.add_trace(
            go.Scatter(
                x=future_x_list,
                y=upper_band[:, 3],
                mode="lines",
                line=dict(color="rgba(88,166,255,0.35)", width=1, dash="dot"),
                name="CI Upper",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=future_x_list,
                y=lower_band[:, 3],
                mode="lines",
                line=dict(color="rgba(88,166,255,0.35)", width=1, dash="dot"),
                name="CI Lower",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1, col=1,
        )

    # ── Anotação de Horizonte Temporal ───────────────────────────────────
    # Posicionar no meio da zona de projeção
    if has_timestamps:
        annot_x = future_x[horizon // 2]
    else:
        annot_x = n_hist + horizon // 2

    # Altura: abaixo do low mínimo projetado
    proj_low_min = proj_ohlc[:, 2].min()
    if lower_band is not None:
        proj_low_min = min(proj_low_min, lower_band[:, 2].min())

    fig.add_annotation(
        x=annot_x,
        y=proj_low_min,
        text=(
            f"<b>🔮 Horizonte Estimado: ~{total_horizon_hours:.1f}h</b><br>"
            f"<span style='font-size:10px; color:#8b949e'>"
            f"{horizon} barras × {avg_delta_sec / 60:.0f} min/barra<br>"
            f"Δt médio baseado nas últimas {min(50, n_hist)} barras</span>"
        ),
        showarrow=True,
        arrowhead=2,
        arrowcolor="#fbbf24",
        ax=0, ay=50,
        font=dict(color="#f0f6fc", size=11),
        bgcolor="rgba(22,27,34,0.92)",
        bordercolor="#fbbf24",
        borderwidth=1,
        borderpad=6,
        row=1, col=1,
    )

    # ── Anotação de target final ─────────────────────────────────────────
    final_close = proj_ohlc[-1, 3]
    fig.add_annotation(
        x=future_x[-1],
        y=final_close,
        text=(
            f"<b>{direction}</b><br>"
            f"{delta_pct:+.4f}%<br>"
            f"<span style='font-size:10px'>{final_close:.5f}</span>"
        ),
        showarrow=True,
        arrowhead=2,
        arrowcolor=EXPERT_COLORS['moe'],
        ax=40, ay=-30,
        font=dict(color=TITLE_C, size=10),
        bgcolor="rgba(22,27,34,0.90)",
        bordercolor=EXPERT_COLORS['moe'],
        borderwidth=1,
        borderpad=4,
        row=1, col=1,
    )

    # ══════════════════════════════════════════════════════════════════════
    # PAINEL 2 — Close de cada Especialista (Projeção Futura)
    # ══════════════════════════════════════════════════════════════════════

    bar_x = list(range(1, horizon + 1))
    moe_close = proj_ohlc[:, 3]

    fig.add_trace(
        go.Scatter(
            x=bar_x,
            y=moe_close,
            mode="lines+markers",
            line=dict(color=EXPERT_COLORS['moe'], width=2.5),
            marker=dict(size=5, symbol="diamond"),
            name="MoE Ponderado (p2)",
            showlegend=True,
        ),
        row=2, col=1,
    )

    expert_keys = ['inception', 'lstm', 'transformer', 'mlp']
    for key in expert_keys:
        if key in ind_preds:
            pred = ind_preds[key]
            c = pred[:, 3] if pred.ndim == 2 else pred[:horizon, 3]
            fig.add_trace(
                go.Scatter(
                    x=bar_x,
                    y=c,
                    mode="lines+markers",
                    line=dict(color=EXPERT_COLORS[key], width=1.2, dash="dash"),
                    marker=dict(size=3),
                    opacity=0.70,
                    name=EXPERT_LABELS[key] + " (p2)",
                    showlegend=True,
                ),
                row=2, col=1,
            )

    # Linha base (último preço real)
    fig.add_hline(
        y=base_price, row=2, col=1,
        line=dict(color="#8b949e", width=0.8, dash="dot"),
        opacity=0.6,
        annotation_text=f"Base: {base_price:.5f}",
        annotation_font=dict(color="#8b949e", size=9),
        annotation_position="bottom right",
    )

    # ══════════════════════════════════════════════════════════════════════
    # PAINEL 3 — Pesos do Gating
    # ══════════════════════════════════════════════════════════════════════

    gating_weights_series = results.get('gating_weights_series')
    colors_g = [EXPERT_COLORS[k] for k in expert_keys]

    if gating_weights_series is not None and len(gating_weights_series) == horizon:
        w_arr = np.array(gating_weights_series)  # (H, 4)
        for i, (lbl, col) in enumerate(zip(gat_labels, colors_g)):
            fig.add_trace(
                go.Scatter(
                    x=bar_x,
                    y=w_arr[:, i],
                    mode="lines",
                    line=dict(color=col, width=0),
                    fill="tonexty" if i > 0 else "tozeroy",
                    fillcolor=col.replace(")", ",0.8)").replace("rgb", "rgba")
                             if "rgb" in col else col,
                    name=lbl,
                    stackgroup="gating",
                    opacity=0.8,
                ),
                row=3, col=1,
            )
    else:
        # Barras estáticas com pesos finais do Gating
        fig.add_trace(
            go.Bar(
                x=gat_labels,
                y=gating_final,
                marker_color=colors_g,
                opacity=0.85,
                name="Pesos Gating",
                text=[f"{w:.3f}" for w in gating_final],
                textposition="outside",
                textfont=dict(color="#f0f6fc", size=11),
            ),
            row=3, col=1,
        )

    # ══════════════════════════════════════════════════════════════════════
    # LAYOUT GLOBAL
    # ══════════════════════════════════════════════════════════════════════

    fig.update_layout(
        height=1100,
        width=1450,
        paper_bgcolor=PAPER,
        plot_bgcolor=BG,
        font=dict(color=TICK_C),
        title=dict(
            text=(
                "Ghost Projection │ MoE Gating Network │ "
                "Hybrid Loss: Soft-DTW + Curvature + Huber │ "
                f"Horizonte: ~{total_horizon_hours:.1f}h"
            ),
            font=dict(size=13, color="#8b949e"),
            x=0.5,
        ),
        legend=dict(
            bgcolor="#161b22",
            bordercolor="#30363d",
            borderwidth=1,
            font=dict(color="#c9d1d9", size=9),
        ),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
    )

    # Aplicar estilo em todos os eixos
    for i in range(1, 4):
        fig.update_xaxes(**AXIS_DEFAULTS, row=i, col=1)
        fig.update_yaxes(**AXIS_DEFAULTS, row=i, col=1)

    # Rótulos
    fig.update_yaxes(
        title_text="Preço",
        title_font=dict(color=TICK_C, size=11),
        row=1, col=1,
    )
    fig.update_yaxes(
        title_text="Close (absoluto)",
        title_font=dict(color=TICK_C, size=10),
        row=2, col=1,
    )
    fig.update_xaxes(
        title_text="Barra Futura",
        title_font=dict(color=TICK_C, size=10),
        row=2, col=1,
    )

    if gating_weights_series is not None and len(gating_weights_series) == horizon:
        fig.update_yaxes(
            title_text="Peso (Softmax)",
            title_font=dict(color=TICK_C, size=10),
            row=3, col=1,
        )
    else:
        fig.update_xaxes(
            title_text="Especialista",
            title_font=dict(color=TICK_C, size=10),
            row=3, col=1,
        )
        fig.update_yaxes(
            title_text="Peso (Softmax)",
            title_font=dict(color=TICK_C, size=10),
            row=3, col=1,
        )

    if has_timestamps:
        fig.update_xaxes(
            title_text="Timestamp",
            title_font=dict(color=TICK_C, size=10),
            row=1, col=1,
        )

    # Estilo dos títulos dos subplots
    for ann in fig.layout.annotations:
        ann.update(font=dict(color="#c9d1d9", size=12))

    # ── Salvar / exibir ─────────────────────────────────────────────────
    if save_path:
        if save_path.endswith(".html"):
            fig.write_html(save_path)
        else:
            fig.write_image(save_path, scale=2)
        print(f"  📊 Ghost Projection salvo: {save_path}")

    fig.show()
    return fig


# =============================================================================
# DAILY PROJECTION — CONVERSÃO TICK BARS → CANDLES DIÁRIOS
# =============================================================================

def _resample_to_daily(
    df: pd.DataFrame,
    timestamp_col: str = 'Date',
    ny_close_hour: int = 17,
    min_ticks: int = 5,
) -> pd.DataFrame:
    """
    Converte tick/volume bars em candles diários segundo NY Close (17h ET).

    Aplica offset de -17h para que pd.resample('1D') corte no horário correto
    do mercado Forex. Filtra dias com menos de ``min_ticks`` barras.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com OHLC + timestamps.
    timestamp_col : str
        Coluna de timestamp.
    ny_close_hour : int
        Hora de corte do dia Forex (US/Eastern).
    min_ticks : int
        Mínimo de barras para formar um candle válido.

    Returns
    -------
    pd.DataFrame
        Candles diários com DatetimeIndex normalizado.
    """
    tmp = df.copy()

    # Resolver timestamps
    if timestamp_col in tmp.columns:
        tmp[timestamp_col] = pd.to_datetime(tmp[timestamp_col])
        tmp = tmp.set_index(timestamp_col)
    elif not isinstance(tmp.index, pd.DatetimeIndex):
        # Fallback: gerar timestamps sintéticos (5min spacing)
        n = len(tmp)
        base_ts = pd.Timestamp.now(tz='US/Eastern').normalize() - pd.Timedelta(days=60)
        tmp.index = pd.date_range(base_ts, periods=n, freq='5min')

    if tmp.index.tz is None:
        tmp.index = tmp.index.tz_localize('UTC')
    tmp.index = tmp.index.tz_convert('US/Eastern')

    # Offset NY Close
    tmp.index = tmp.index - pd.Timedelta(hours=ny_close_hour)

    # Contagem para filtragem
    tick_counts = tmp.resample('1D').size()

    # Agregação OHLC
    agg = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
    if 'Volume' in tmp.columns:
        agg['Volume'] = 'sum'
    daily = tmp.resample('1D').agg(agg)

    # Reverter offset
    daily.index = daily.index + pd.Timedelta(hours=ny_close_hour)
    daily.index = daily.index.normalize()
    daily.index.name = 'Date'

    # Filtrar dias incompletos
    valid_dates = tick_counts[tick_counts >= min_ticks].index + pd.Timedelta(hours=ny_close_hour)
    valid_dates = valid_dates.normalize()
    daily = daily[daily.index.isin(valid_dates)]
    daily = daily.dropna(subset=['Open', 'Close'])

    return daily


def _identify_today_bars(
    df: pd.DataFrame,
    timestamp_col: str = 'Date',
    ny_close_hour: int = 17,
) -> pd.DataFrame:
    """
    Identifica as barras intradiárias que pertencem ao dia corrente (D0)
    segundo o calendário NY Close.

    Returns
    -------
    pd.DataFrame
        Subset de ``df`` contendo apenas as barras do dia atual.
    """
    tmp = df.copy()
    if timestamp_col in tmp.columns:
        tmp['_ts'] = pd.to_datetime(tmp[timestamp_col])
    elif isinstance(tmp.index, pd.DatetimeIndex):
        tmp['_ts'] = tmp.index
    else:
        return tmp.tail(0)  # sem timestamps → vazio

    if tmp['_ts'].dt.tz is None:
        tmp['_ts'] = tmp['_ts'].dt.tz_localize('UTC')
    tmp['_ts'] = tmp['_ts'].dt.tz_convert('US/Eastern')

    # Dia Forex = 17h ET anterior → 17h ET atual
    last_ts = tmp['_ts'].iloc[-1]
    ny_close_today = last_ts.normalize() + pd.Timedelta(hours=ny_close_hour)
    if last_ts >= ny_close_today:
        day_start = ny_close_today
    else:
        day_start = ny_close_today - pd.Timedelta(days=1)

    mask = tmp['_ts'] >= day_start
    return df.loc[mask]


def plot_daily_projection(
    df_history: pd.DataFrame,
    results: Dict[str, Any],
    n_history_days: int = 30,
    timestamp_col: str = 'Date',
    bars_per_day: int = 50,
    ny_close_hour: int = 17,
    save_path: Optional[str] = '../exports/plots/moe_daily_projection.html',
    title_suffix: str = "USD/JPY",
) -> go.Figure:
    """
    Projeta o MoE de tick bars para Gráfico Diário com candle híbrido.

    Converte as 15 barras projetadas pela IA para o espaço diário,
    fundindo-as com os dados reais já ocorridos no dia atual (D0) para
    formar um "Candle Diário Projetado" visualmente distinto.

    Painéis
    -------
    1. **Candles Diários** — Histórico completo + candle D0 híbrido
       (borda tracejada, semitransparente) + CI do Close daily
    2. **Close Diário** — Linha de Close + CI conformal do daily
    3. **Decomposição Intraday** — Barras intradiárias reais vs projetadas
       que compõem o candle D0

    Parameters
    ----------
    df_history : pd.DataFrame
        Tick/Volume bars históricas com OHLC + timestamps.
    results : dict
        Output de load_inference_results() — projected_ohlc, CI bands, etc.
    n_history_days : int
        Número de candles diários históricos a exibir.
    timestamp_col : str
        Coluna de timestamp.
    bars_per_day : int
        Calibração do pipeline (default=50 barras/dia).
    ny_close_hour : int
        Hora NY Close (default=17 → 17h00 ET).
    save_path : str, optional
        Caminho para salvar o gráfico.
    title_suffix : str
        Par de moedas para o título.

    Returns
    -------
    go.Figure
    """
    proj_ohlc    = results['projected_ohlc']          # (H, 4)
    upper_band   = results.get('confidence_upper')     # (H, 4) ou None
    lower_band   = results.get('confidence_lower')     # (H, 4) ou None
    base_price   = results.get('base_price', proj_ohlc[0, 0])
    conf_level   = results.get('confidence_level', 0.90)
    horizon      = len(proj_ohlc)

    print(f"  ── Daily Projection: {horizon} tick bars → candle diário ──")

    # ═══════════════════════════════════════════════════════════════════════
    # 1. RESAMPLE HISTÓRICO → DAILY
    # ═══════════════════════════════════════════════════════════════════════
    has_timestamps = (
        (timestamp_col in df_history.columns) or
        isinstance(df_history.index, pd.DatetimeIndex)
    )

    if has_timestamps:
        daily_hist = _resample_to_daily(
            df_history, timestamp_col=timestamp_col,
            ny_close_hour=ny_close_hour, min_ticks=5,
        )
        today_bars = _identify_today_bars(
            df_history, timestamp_col=timestamp_col,
            ny_close_hour=ny_close_hour,
        )
    else:
        # Fallback sem timestamps: agrupar por blocos de bars_per_day barras
        n = len(df_history)
        n_days_est = max(1, n // bars_per_day)
        daily_rows = []
        for i in range(n_days_est):
            start = i * bars_per_day
            end = min(start + bars_per_day, n)
            chunk = df_history.iloc[start:end]
            if len(chunk) < 5:
                continue
            daily_rows.append({
                'Open': chunk['Open'].iloc[0],
                'High': chunk['High'].max(),
                'Low': chunk['Low'].min(),
                'Close': chunk['Close'].iloc[-1],
            })
        base_date = pd.Timestamp.now().normalize() - pd.Timedelta(days=n_days_est)
        daily_hist = pd.DataFrame(daily_rows)
        daily_hist.index = pd.date_range(base_date, periods=len(daily_rows), freq='1D')
        daily_hist.index.name = 'Date'
        today_bars = df_history.tail(bars_per_day)

    print(f"  Candles diários históricos: {len(daily_hist)}")

    # ═══════════════════════════════════════════════════════════════════════
    # 2. CONSTRUIR CANDLE DIÁRIO PROJETADO (D0 HÍBRIDO)
    # ═══════════════════════════════════════════════════════════════════════

    # Dados reais de hoje (parciais)
    if len(today_bars) > 0:
        real_open  = float(today_bars['Open'].iloc[0])
        real_high  = float(today_bars['High'].max())
        real_low   = float(today_bars['Low'].min())
        real_close = float(today_bars['Close'].iloc[-1])
        n_real_bars = len(today_bars)
    else:
        real_open = real_high = real_low = real_close = base_price
        n_real_bars = 0

    # Projeção MoE (15 barras futuras)
    proj_high  = float(np.max(proj_ohlc[:, 1]))
    proj_low   = float(np.min(proj_ohlc[:, 2]))
    proj_close = float(proj_ohlc[-1, 3])

    # Candle D0 híbrido: fusão real + projeção
    d0_open  = real_open
    d0_high  = max(real_high, proj_high)
    d0_low   = min(real_low, proj_low)
    d0_close = proj_close  # Close = última barra projetada

    # Barras restantes no dia (estimativa)
    bars_remaining = max(0, bars_per_day - n_real_bars)
    pct_day_complete = n_real_bars / bars_per_day * 100

    print(f"  Barras reais hoje (D0): {n_real_bars}/{bars_per_day} "
          f"({pct_day_complete:.0f}% do dia)")
    print(f"  Barras projetadas: {horizon} "
          f"(cobrindo {horizon / bars_per_day * 100:.0f}% adicional)")

    # CI do Close diário
    ci_close_low = ci_close_high = None
    if lower_band is not None and upper_band is not None:
        ci_close_low  = float(np.min(lower_band[:, 2]))   # Min Low CI
        ci_close_high = float(np.max(upper_band[:, 1]))   # Max High CI
        ci_daily_close_low  = float(lower_band[-1, 3])    # CI do Close final
        ci_daily_close_high = float(upper_band[-1, 3])

    # ═══════════════════════════════════════════════════════════════════════
    # 3. PLOTAR com Plotly (3 Painéis)
    # ═══════════════════════════════════════════════════════════════════════

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.06,
        row_heights=[0.50, 0.25, 0.25],
        subplot_titles=[
            "📅 Gráfico Diário + Candle Projetado (D0)",
            "📈 Close Diário + Intervalo de Confiança",
            "🔬 Decomposição Intraday do D0 (Real + Projeção MoE)",
        ],
    )

    # ───────────────────────────────────────────────────────────────────────
    # PAINEL 1: CANDLESTICK DIÁRIO + D0 HÍBRIDO
    # ───────────────────────────────────────────────────────────────────────

    daily_show = daily_hist.tail(n_history_days).copy()

    # Remover o último dia se for o mesmo que D0 (pois vamos plotar o híbrido)
    if len(daily_show) > 0:
        last_daily_date = daily_show.index[-1]
        if has_timestamps:
            d0_date = pd.Timestamp.now(tz='US/Eastern').normalize()
        else:
            d0_date = last_daily_date + pd.Timedelta(days=1)

        # Se o último candle diário é de hoje, removê-lo pois vamos substituir
        if last_daily_date == d0_date:
            daily_show = daily_show.iloc[:-1]

    # Candles diários históricos
    for idx, row in daily_show.iterrows():
        x_val = idx
        o, h, l, c = row['Open'], row['High'], row['Low'], row['Close']
        is_bull = c >= o
        color = '#26a69a' if is_bull else '#ef5350'

        # Sombra
        fig.add_trace(go.Scatter(
            x=[x_val, x_val], y=[l, h],
            mode='lines', line=dict(color=color, width=1),
            showlegend=False, hoverinfo='skip',
        ), row=1, col=1)

        # Corpo
        fig.add_trace(go.Bar(
            x=[x_val], y=[abs(c - o) if abs(c - o) > 1e-10 else (h - l) * 0.01],
            base=[min(o, c)], marker=dict(color=color, opacity=0.85,
                                           line=dict(color=color, width=0.5)),
            width=0.7 * 86400000,  # 70% de um dia em ms
            showlegend=False, hovertemplate=
                f"<b>{idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else idx}</b><br>"
                f"O: {o:.5f}<br>H: {h:.5f}<br>L: {l:.5f}<br>C: {c:.5f}<extra></extra>",
        ), row=1, col=1)

    # ── Candle D0 HÍBRIDO (real + projeção) ──────────────────────────────

    is_bull_d0 = d0_close >= d0_open
    d0_color = '#4fc3f7' if is_bull_d0 else '#ff8a65'

    # Sombra D0
    fig.add_trace(go.Scatter(
        x=[d0_date, d0_date], y=[d0_low, d0_high],
        mode='lines', line=dict(color=d0_color, width=2, dash='dot'),
        showlegend=False, hoverinfo='skip',
    ), row=1, col=1)

    # Corpo D0 (semitransparente com borda tracejada)
    d0_body = abs(d0_close - d0_open)
    if d0_body < 1e-10:
        d0_body = (d0_high - d0_low) * 0.01

    fig.add_trace(go.Bar(
        x=[d0_date], y=[d0_body],
        base=[min(d0_open, d0_close)],
        marker=dict(
            color=d0_color, opacity=0.35,
            line=dict(color=d0_color, width=2),
            pattern=dict(shape="/", solidity=0.3),
        ),
        width=0.7 * 86400000,
        name='D0 Projetado',
        showlegend=True,
        hovertemplate=(
            f"<b>D0 — Candle Projetado</b><br>"
            f"O (real): {d0_open:.5f}<br>"
            f"H (max): {d0_high:.5f}<br>"
            f"L (min): {d0_low:.5f}<br>"
            f"C (MoE): {d0_close:.5f}<br>"
            f"Real: {n_real_bars} barras | Proj: {horizon} barras"
            f"<extra></extra>"
        ),
    ), row=1, col=1)

    # ── Porção Real dentro do D0 (overlay sólido) ────────────────────────
    if n_real_bars > 0:
        real_body = abs(real_close - real_open)
        if real_body < 1e-10:
            real_body = (real_high - real_low) * 0.01
        real_bull = real_close >= real_open
        real_color = '#26a69a' if real_bull else '#ef5350'

        fig.add_trace(go.Bar(
            x=[d0_date], y=[real_body],
            base=[min(real_open, real_close)],
            marker=dict(color=real_color, opacity=0.7,
                        line=dict(color='white', width=1)),
            width=0.3 * 86400000,
            name=f'D0 Real ({n_real_bars} barras)',
            showlegend=True,
            hovertemplate=(
                f"<b>D0 — Parcial Real</b><br>"
                f"O: {real_open:.5f}<br>H: {real_high:.5f}<br>"
                f"L: {real_low:.5f}<br>C: {real_close:.5f}"
                f"<extra></extra>"
            ),
        ), row=1, col=1)

    # ── CI Band no D0 ────────────────────────────────────────────────────
    if ci_close_low is not None:
        # Sombra de incerteza (vertical)
        fig.add_trace(go.Scatter(
            x=[d0_date, d0_date], y=[ci_close_low, ci_close_high],
            mode='lines',
            line=dict(color='#58a6ff', width=6, dash='solid'),
            opacity=0.15,
            name=f'CI {conf_level:.0%} (Daily)',
            showlegend=True,
            hovertemplate=(
                f"<b>CI {conf_level:.0%} Diário</b><br>"
                f"Close Low: {ci_daily_close_low:.5f}<br>"
                f"Close High: {ci_daily_close_high:.5f}"
                f"<extra></extra>"
            ),
        ), row=1, col=1)

    # Anotação no D0
    direction = "↑ ALTA" if d0_close > d0_open else "↓ BAIXA"
    delta_pct = (d0_close - real_close) / real_close * 100

    fig.add_annotation(
        x=d0_date, y=d0_high, yshift=20,
        text=(f"<b>D0 — {direction}</b><br>"
              f"Close MoE: {d0_close:.5f} ({delta_pct:+.3f}%)<br>"
              f"{n_real_bars} barras reais + {horizon} projetadas"),
        showarrow=True, arrowhead=2, arrowcolor='#58a6ff',
        font=dict(color='#f0f6fc', size=10),
        bgcolor='rgba(22,27,34,0.85)', bordercolor='#58a6ff',
        borderwidth=1, borderpad=6, arrowwidth=1.5,
        row=1, col=1,
    )

    # ───────────────────────────────────────────────────────────────────────
    # PAINEL 2: CLOSE DIÁRIO + CI
    # ───────────────────────────────────────────────────────────────────────

    close_dates = list(daily_show.index) + [d0_date]
    close_vals  = list(daily_show['Close'].values) + [d0_close]

    fig.add_trace(go.Scatter(
        x=close_dates, y=close_vals,
        mode='lines+markers',
        line=dict(color='#fbbf24', width=2),
        marker=dict(size=4, color='#fbbf24'),
        name='Close Diário',
        showlegend=True,
    ), row=2, col=1)

    # CI band para o D0
    if ci_daily_close_low is not None:
        fig.add_trace(go.Scatter(
            x=[d0_date, d0_date],
            y=[ci_daily_close_low, ci_daily_close_high],
            mode='lines',
            line=dict(color='#58a6ff', width=12),
            opacity=0.2,
            showlegend=False,
        ), row=2, col=1)

        # Marcadores de CI
        fig.add_trace(go.Scatter(
            x=[d0_date, d0_date],
            y=[ci_daily_close_low, ci_daily_close_high],
            mode='markers',
            marker=dict(symbol='line-ew', size=10, color='#58a6ff',
                        line=dict(width=2, color='#58a6ff')),
            showlegend=False,
            hovertemplate=(
                f"CI {conf_level:.0%}<br>"
                f"{{y:.5f}}<extra></extra>"
            ),
        ), row=2, col=1)

    # D0 marker distinto
    fig.add_trace(go.Scatter(
        x=[d0_date], y=[d0_close],
        mode='markers',
        marker=dict(size=12, color='#f72585', symbol='diamond',
                    line=dict(color='white', width=1.5)),
        name='Close Projetado D0',
        showlegend=True,
        hovertemplate=f"<b>D0 Close MoE</b><br>{d0_close:.5f}<extra></extra>",
    ), row=2, col=1)

    # Linha de referência (último Close real)
    fig.add_hline(
        y=real_close, row=2, col=1,
        line=dict(color='#8b949e', width=1, dash='dot'),
        annotation_text=f"Último Close Real: {real_close:.5f}",
        annotation_font=dict(color='#8b949e', size=9),
    )

    # ───────────────────────────────────────────────────────────────────────
    # PAINEL 3: DECOMPOSIÇÃO INTRADAY DO D0
    # ───────────────────────────────────────────────────────────────────────

    # Barras reais intraday de hoje
    real_intraday = today_bars.tail(min(50, len(today_bars)))
    n_intra_real = len(real_intraday)

    for i in range(n_intra_real):
        o = float(real_intraday['Open'].iloc[i])
        h = float(real_intraday['High'].iloc[i])
        l = float(real_intraday['Low'].iloc[i])
        c = float(real_intraday['Close'].iloc[i])
        is_bull = c >= o
        color = '#26a69a' if is_bull else '#ef5350'

        fig.add_trace(go.Scatter(
            x=[i, i], y=[l, h],
            mode='lines', line=dict(color=color, width=0.8),
            showlegend=False, hoverinfo='skip',
        ), row=3, col=1)

        body_h = max(abs(c - o), (h - l) * 0.005)
        fig.add_trace(go.Bar(
            x=[i], y=[body_h], base=[min(o, c)],
            marker=dict(color=color, opacity=0.85),
            width=0.7, showlegend=False,
            hovertemplate=f"Bar {i+1}<br>C: {c:.5f}<extra></extra>",
        ), row=3, col=1)

    # Linha divisória (T0)
    div_x = n_intra_real - 0.5
    fig.add_vline(
        x=div_x, row=3, col=1,
        line=dict(color='#58a6ff', width=2, dash='dash'),
    )
    fig.add_annotation(
        x=div_x + 0.5, y=real_close, yshift=15,
        text="<b>T₀ Inferência</b>",
        font=dict(color='#58a6ff', size=9),
        showarrow=False, bgcolor='rgba(22,27,34,0.8)',
        bordercolor='#58a6ff', borderwidth=1, borderpad=3,
        row=3, col=1,
    )

    # Barras projetadas (Ghost)
    for k in range(horizon):
        x_pos = n_intra_real + k
        o = float(proj_ohlc[k, 0])
        h = float(proj_ohlc[k, 1])
        l = float(proj_ohlc[k, 2])
        c = float(proj_ohlc[k, 3])
        is_bull = c >= o
        color = '#4fc3f7' if is_bull else '#ff8a65'

        fig.add_trace(go.Scatter(
            x=[x_pos, x_pos], y=[l, h],
            mode='lines', line=dict(color=color, width=0.8),
            showlegend=False, hoverinfo='skip',
        ), row=3, col=1)

        body_h = max(abs(c - o), (h - l) * 0.005)
        fig.add_trace(go.Bar(
            x=[x_pos], y=[body_h], base=[min(o, c)],
            marker=dict(color=color, opacity=0.3),
            width=0.7, showlegend=False,
            hovertemplate=f"Proj {k+1}/{horizon}<br>C: {c:.5f}<extra></extra>",
        ), row=3, col=1)

    # CI no painel intraday
    if upper_band is not None and lower_band is not None:
        proj_x = list(range(n_intra_real, n_intra_real + horizon))
        fig.add_trace(go.Scatter(
            x=proj_x + proj_x[::-1],
            y=list(upper_band[:, 3]) + list(lower_band[::-1, 3]),
            fill='toself', fillcolor='rgba(88,166,255,0.08)',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False, hoverinfo='skip',
        ), row=3, col=1)

    # Zona futura sombreada
    fig.add_vrect(
        x0=div_x, x1=n_intra_real + horizon,
        row=3, col=1,
        fillcolor='rgba(88,166,255,0.04)',
        line_width=0,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 4. LAYOUT
    # ═══════════════════════════════════════════════════════════════════════

    fig.update_layout(
        title=dict(
            text=(f"<b>📅 Daily Projection — MoE Gating Network</b>  "
                  f"<span style='color:#8b949e'>|</span>  "
                  f"<span style='color:#fbbf24'>{title_suffix}</span>  "
                  f"<span style='color:#8b949e'>|</span>  "
                  f"<span style='color:{'#4ade80' if is_bull_d0 else '#ef5350'}'>"
                  f"D0: {direction} {delta_pct:+.3f}%</span>"),
            font=dict(color=TITLE_C, size=16),
            x=0.5,
        ),
        height=1100,
        paper_bgcolor=PAPER,
        plot_bgcolor=BG,
        font=dict(color=TICK_C, size=10),
        legend=dict(
            bgcolor='rgba(22,27,34,0.85)',
            bordercolor='#30363d',
            borderwidth=1,
            font=dict(color='#c9d1d9', size=9),
            orientation='h',
            yanchor='bottom', y=1.02, xanchor='right', x=1,
        ),
        barmode='overlay',
        # Toggle buttons: Intraday / Daily view
        updatemenus=[dict(
            type='buttons',
            direction='left',
            x=0.01, y=1.01, xanchor='left', yanchor='bottom',
            bgcolor='#161b22', bordercolor='#30363d',
            font=dict(color='#c9d1d9', size=10),
            buttons=[
                dict(
                    label='📅 Daily + Intraday',
                    method='update',
                    args=[{'visible': True}],
                ),
            ],
        )],
    )

    # Eixos
    for row_n in [1, 2, 3]:
        fig.update_xaxes(
            **AXIS_DEFAULTS,
            row=row_n, col=1,
        )
        fig.update_yaxes(
            **AXIS_DEFAULTS,
            title=dict(text='Preço' if row_n < 3 else 'Preço (intraday)',
                       font=dict(color=TICK_C, size=10)),
            row=row_n, col=1,
        )

    # Títulos dos subplots
    for ann in fig['layout']['annotations']:
        ann['font'] = dict(color='#c9d1d9', size=12)

    # Watermark com métricas
    fig.add_annotation(
        text=(f"Barras/dia: {bars_per_day} | "
              f"D0 real: {n_real_bars} barras ({pct_day_complete:.0f}%) | "
              f"Projeção: {horizon} barras | "
              f"CI: {conf_level:.0%} Conformal"),
        xref='paper', yref='paper', x=0.5, y=-0.02,
        showarrow=False,
        font=dict(color='#484f58', size=9),
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 5. SALVAR
    # ═══════════════════════════════════════════════════════════════════════

    if save_path:
        fig.write_html(
            save_path,
            include_plotlyjs='cdn',
            full_html=True,
            config={'displayModeBar': True, 'scrollZoom': True},
        )
        print(f"  📊 Daily Projection salvo: {save_path}")

    fig.show()
    return fig

# =============================================================================
# EXECUÇÃO DIRETA — TOTALMENTE STANDALONE
# =============================================================================
if __name__ == "__main__":

    print("=" * 65)
    print("  MoE VISUALIZATION — ANÁLISE PÓS-TREINO (Standalone)")
    print("=" * 65)

    # ── Caminhos default ─────────────────────────────────────────────────
    RESULTS_PATH = '../exports/inference/results_inference.parquet'
    CONFIG_PATH  = '../models/EURUSD/moe_config.json'
    DATA_FILE    = '../data/final/dataset_final.h5'
    DATA_KEY     = 'data'
    MODEL_PATH   = '../models/EURUSD/moe_model.keras'

    # ── 1. Carregar resultados de inferência ─────────────────────────────
    if not os.path.exists(RESULTS_PATH):
        print(f"  ❌ {RESULTS_PATH} não encontrado.")
        print(f"  Execute moe_to_daily.py primeiro para treinar e salvar checkpoints.")
        exit(1)

    results = load_inference_results(RESULTS_PATH)

    # ── 2. Carregar config (fold_metrics etc.) ───────────────────────────
    config_data = {}
    if os.path.exists(CONFIG_PATH):
        config_data = load_config(CONFIG_PATH)
        # Injetar fold_metrics no results dict para o plot
        if 'fold_metrics' in config_data:
            results['fold_metrics'] = config_data['fold_metrics']
        # Descobrir data file do config
        DATA_FILE = config_data.get('input_file', DATA_FILE)
        DATA_KEY = config_data.get('input_key', DATA_KEY)
    else:
        print(f"  ⚠️ {CONFIG_PATH} não encontrado — usando defaults.")

    # ── 3. Carregar dados históricos do HDF5 original ────────────────────
    df_history = None
    try:
        # Tentar ler do arquivo original
        df_history = pd.read_hdf(DATA_FILE, key=DATA_KEY)
        print(f"  📂 Histórico: {len(df_history)} barras de '{DATA_FILE}'")
    except FileNotFoundError:
        # Fallback: tentar outros arquivos
        for fallback in ['../data/final/dataset_final.h5', '../data/processed/dataset_clean.h5']:
            try:
                for key in ['data', 'features']:
                    try:
                        df_history = pd.read_hdf(fallback, key=key)
                        print(f"  📂 Histórico (fallback): {len(df_history)} barras de '{fallback}'")
                        break
                    except KeyError:
                        continue
                if df_history is not None:
                    break
            except FileNotFoundError:
                continue

    if df_history is None:
        print("  ⚠️ Nenhum arquivo de dados históricos encontrado.")
        print("  Os gráficos serão gerados sem candlesticks históricos.")
        # Criar df_history mínimo com base_price
        bp = results['base_price']
        df_history = pd.DataFrame({
            'Open': [bp], 'High': [bp], 'Low': [bp], 'Close': [bp]
        })

    # ── 4. Gerar plots ──────────────────────────────────────────────────
    print(f"\n  Gerando gráfico de análise MoE...")
    plot_moe_analysis(
        df_history=df_history,
        results=results,
        n_history=80,
        save_path='../exports/plots/moe_analysis.html',
    )

    # ── 5. Ghost Projection (Projeção Futura Out-of-Sample) ──────────────
    print(f"\n  Gerando Ghost Projection (projeção futura)...")
    plot_ghost_projection(
        df_history=df_history,
        results=results,
        n_history=60,
        save_path='../exports/plots/moe_ghost_projection.html',
    )

    # ── 6. Daily Projection (Tick Bars → Candle Diário Projetado) ────────
    print(f"\n  Gerando Daily Projection (visão diária)...")
    plot_daily_projection(
        df_history=df_history,
        results=results,
        n_history_days=30,
        save_path='../exports/plots/moe_daily_projection.html',
    )

    # ── 7. Rolling Backtest 30 dias (Projeção vs Real) ───────────────────
    if os.path.exists(MODEL_PATH) and len(df_history) > 100:
        print(f"\n  Gerando backtest rolling de 30 dias...")
        backtest_results = run_rolling_backtest_30d(
            df_data=df_history,
            model_path=MODEL_PATH,
            config_path=CONFIG_PATH,
            n_days=30,
            save_path='../exports/plots/moe_rolling_30d.html',
        )
    else:
        print(f"\n  ⚠️ Modelo '{MODEL_PATH}' não encontrado ou dados insuficientes.")
        print(f"  Pulando backtest rolling. Execute moe_to_daily.py primeiro.")
        # Fallback: comparação diária (sem modelo)
        run_daily_comparison(
            df_tick_bars=df_history,
            moe_results=results,
            save_path='../exports/plots/moe_daily_comparison.html',
        )

    print(f"\n  ✅ Visualização completa!")
