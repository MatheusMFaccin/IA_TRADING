# -*- coding: utf-8 -*-
"""
===============================================================================
MoE INFERENCE TOOL — Projeção Futura Standalone (15 barras)
===============================================================================
Carrega modelo MoE treinado (.keras), config (.json) e dataset (.h5)
para gerar projeção das próximas 15 barras (≈ 1 dia de mercado).

Uso:
    python moe_inference_tool.py
    python moe_inference_tool.py --model moe_model.keras --config moe_config.json
===============================================================================
"""
from __future__ import annotations
import os, sys, json, argparse, warnings, logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple

warnings.filterwarnings('ignore')

logger = logging.getLogger('MoE_Inference')
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(h)

EXPERT_NAMES = ['InceptionTime', 'LSTM Seq2Seq', 'Transformer', 'ResidualMLP']
SEP = '═' * 65


# =============================================================================
# 1. CARREGAMENTO DE CONFIG + MODELO + DADOS
# =============================================================================

def load_config(path: str) -> dict:
    """Carrega moe_config.json."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_model(model_path: str, config: dict):
    """Carrega modelo MoE com custom objects."""
    import tensorflow as tf
    from moe_gating import (
        InceptionBlock, CausalSelfAttentionMoE, ResidualMLPBlock,
        RegimeGatingNetwork, MoEForecaster,
        hybrid_financial_loss, morphological_loss,
    )
    custom_objects = {
        'InceptionBlock': InceptionBlock,
        'CausalSelfAttentionMoE': CausalSelfAttentionMoE,
        'ResidualMLPBlock': ResidualMLPBlock,
        'RegimeGatingNetwork': RegimeGatingNetwork,
        'MoEForecaster': MoEForecaster,
        'hybrid_financial_loss': hybrid_financial_loss,
        'morphological_loss': morphological_loss,
    }
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
    a = config.get('loss_alpha', 0.5)
    b = config.get('loss_beta', 0.2)
    g = config.get('loss_gamma', 1.0)
    d = config.get('huber_delta', 1.0)
    model.compile(optimizer='adam', loss=lambda yt, yp: hybrid_financial_loss(yt, yp, a, b, g, d))
    return model


def load_dataset(data_path: str, data_key: str) -> pd.DataFrame:
    """Carrega dataset HDF5."""
    df = pd.read_hdf(data_path, key=data_key)
    if 'Returns' not in df.columns:
        df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df.dropna(inplace=True)
    return df


# =============================================================================
# 2. PROCESSAMENTO DE DADOS PARA INFERÊNCIA
# =============================================================================

def check_data_freshness(df: pd.DataFrame, max_hours: int = 24):
    """Emite alerta se dados estiverem defasados."""
    if 'Date' in df.columns:
        last_ts = pd.to_datetime(df['Date'].iloc[-1])
        age = datetime.utcnow() - last_ts.to_pydatetime().replace(tzinfo=None)
        hours = age.total_seconds() / 3600
        if hours > max_hours:
            print(f"\n  ⚠️  ALERTA: Dados defasados em {hours:.1f}h (último: {last_ts})")
            print(f"  ⚠️  A projeção pode NÃO refletir condições atuais de mercado.\n")
            return False
        else:
            print(f"  ✅ Dados atualizados (última barra: {last_ts}, idade: {hours:.1f}h)")
            return True
    print("  ⚠️  Sem coluna 'Date' — impossível verificar frescura dos dados.")
    return True


def prepare_inference_window(df: pd.DataFrame, window_size: int) -> Tuple[np.ndarray, float, list]:
    """Extrai última janela de features para inferência."""
    from moe_gating import _find_regime_col_indices
    candidates = [
        'Alpha_Returns_norm', 'Alpha_Returns',
        'Alpha_GK_Vol_norm', 'Alpha_GK_Vol',
        'Alpha_Hurst_norm', 'Alpha_Hurst',
        'Alpha_Amihud_norm', 'Alpha_Amihud',
        'Alpha_CDF_VPIN_norm', 'Alpha_CDF_VPIN',
        'Alpha_MA_Cross',
        'Alpha_Spread_norm', 'Alpha_Spread',
        'Alpha_VWAP_norm', 'Alpha_VWAP',
        'Alpha_OI_norm', 'Alpha_OI',
        'Alpha_Close_FracDiff_norm', 'Alpha_Close_FracDiff',
        'Alpha_Kyle_Lambda_norm', 'Alpha_Kyle_Lambda',
        'Returns',
    ]
    feat_cols = [c for c in candidates if c in df.columns]
    if not feat_cols:
        feat_cols = ['Returns']

    tail = df.tail(window_size)
    if len(tail) < window_size:
        raise ValueError(f"Dados insuficientes: {len(tail)}/{window_size} barras disponíveis.")

    X = tail[feat_cols].values.astype('float32')
    X = X.reshape(1, window_size, -1)
    base_price = float(tail['Close'].iloc[-1])
    return X, base_price, feat_cols


# =============================================================================
# 3. FORWARD PASS + DENORMALIZAÇÃO
# =============================================================================

def run_inference(model, X: np.ndarray, base_price: float) -> Dict[str, Any]:
    """Executa inferência e retorna resultados denormalizados."""
    # Projeção MoE (retornos relativos)
    proj_rel = model.predict(X, verbose=0)[0]  # (H, 4)

    # Denormalizar: Price = base * (1 + delta)
    proj_abs = base_price * (1.0 + proj_rel)

    # Pesos do Gating
    try:
        gating_w = model.get_gating_weights(X)[0]
    except Exception:
        gating_w = np.ones(4) / 4.0

    # Predições individuais dos especialistas
    try:
        ind = model.get_individual_predictions(X)
        ind_abs = {k: base_price * (1.0 + v.numpy()[0]) for k, v in ind.items()}
    except Exception:
        ind_abs = {}

    target_close = float(proj_abs[-1, 3])
    delta_pct = (target_close - base_price) / base_price * 100
    direction = 'BULLISH ↑' if delta_pct > 0 else 'BEARISH ↓'

    # Confiança = peso do especialista dominante
    dominant_idx = int(np.argmax(gating_w))
    confidence = float(gating_w[dominant_idx]) * 100

    return {
        'proj_rel': proj_rel,
        'proj_abs': proj_abs,
        'base_price': base_price,
        'target_close': target_close,
        'delta_pct': delta_pct,
        'direction': direction,
        'gating_weights': gating_w,
        'dominant_expert': EXPERT_NAMES[dominant_idx],
        'dominant_weight': confidence,
        'individual_abs': ind_abs,
    }


# =============================================================================
# 4. TERMINAL REPORT
# =============================================================================

def print_terminal_report(res: Dict[str, Any], horizon: int):
    """Imprime métricas no terminal."""
    print(f"\n{SEP}")
    print("  🧠 MoE INFERENCE TOOL — PROJEÇÃO FUTURA")
    print(SEP)
    print(f"  Base Price (T₀):       {res['base_price']:.5f}")
    print(f"  Preço Alvo (T+{horizon}):    {res['target_close']:.5f}")
    print(f"  Variação Esperada:     {res['delta_pct']:+.4f}%")
    print(f"  Direção (Daily Bias):  {res['direction']}")
    print(f"  Confiança do Modelo:   {res['dominant_weight']:.1f}%")
    print(f"  Especialista Dominante: {res['dominant_expert']} ({res['dominant_weight']:.1f}%)")
    print(f"\n  Pesos do Gating Network:")
    for name, w in zip(EXPERT_NAMES, res['gating_weights']):
        bar = '█' * int(w * 40)
        print(f"    {name:15s} │ {w:.4f} │ {bar}")

    print(f"\n  Projeção OHLC ({horizon} barras):")
    print(f"  {'Bar':>4s}  {'Open':>10s}  {'High':>10s}  {'Low':>10s}  {'Close':>10s}  {'Δ%':>8s}")
    print(f"  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*8}")
    bp = res['base_price']
    for k in range(horizon):
        o, h, l, c = res['proj_abs'][k]
        d = (c - bp) / bp * 100
        print(f"  {k+1:4d}  {o:10.5f}  {h:10.5f}  {l:10.5f}  {c:10.5f}  {d:+7.3f}%")
    print(SEP)


# =============================================================================
# 5. VISUALIZAÇÃO PLOTLY
# =============================================================================

def build_projection_chart(
    df_history: pd.DataFrame,
    res: Dict[str, Any],
    horizon: int,
    n_history: int = 20,
    save_path: str = 'projecao_futura.html',
) -> None:
    """Gera gráfico Plotly com histórico + projeção ghost + gating."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    BG = "#0d1117"
    PAPER = "#0d1117"
    GRID_C = "#21262d"
    TICK_C = "#8b949e"
    EXPERT_COLORS = ['#f72585', '#4cc9f0', '#7209b7', '#4ade80']
    MOE_COLOR = '#fbbf24'

    proj_abs = res['proj_abs']
    base_price = res['base_price']
    gating_w = res['gating_weights']
    ind_abs = res['individual_abs']

    # Histórico recente
    hist = df_history.tail(n_history).copy()
    n_hist = len(hist)

    # Eixo X: numérico (barras) — histórico [0..n_hist-1], projeção [n_hist..n_hist+H-1]
    has_ts = 'Date' in hist.columns
    if has_ts:
        ts = pd.to_datetime(hist['Date'])
        last_ts = ts.iloc[-1]
        delta = ts.diff().median()
        if pd.isna(delta):
            delta = pd.Timedelta(minutes=5)
        hist_x = ts.values
        fut_x = pd.DatetimeIndex([last_ts + delta * (i+1) for i in range(horizon)])
        conn_x = [ts.iloc[-1], fut_x[0]]
    else:
        hist_x = np.arange(n_hist)
        fut_x = np.arange(n_hist, n_hist + horizon)
        conn_x = [n_hist - 1, n_hist]

    # Subplots: 3 rows
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.55, 0.25, 0.20],
        shared_xaxes=False,
        vertical_spacing=0.06,
        subplot_titles=(
            f"Projeção MoE — {horizon} Barras │ Bias: {res['direction']} ({res['delta_pct']:+.3f}%)",
            "Close por Especialista (Projeção)",
            f"Gating Network │ Dominante: {res['dominant_expert']} ({res['dominant_weight']:.1f}%)",
        ),
    )

    # ── Painel 1: Candlestick histórico + Ghost candles ──
    fig.add_trace(go.Candlestick(
        x=hist_x,
        open=hist['Open'], high=hist['High'],
        low=hist['Low'], close=hist['Close'],
        name='Histórico Real',
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
        increasing_fillcolor='#26a69a', decreasing_fillcolor='#ef5350',
    ), row=1, col=1)

    # Ghost candles (projeção)
    fig.add_trace(go.Candlestick(
        x=fut_x,
        open=proj_abs[:, 0], high=proj_abs[:, 1],
        low=proj_abs[:, 2], close=proj_abs[:, 3],
        name='Projeção MoE (Ghost)',
        increasing_line_color='rgba(251,191,36,0.7)',
        decreasing_line_color='rgba(239,83,80,0.5)',
        increasing_fillcolor='rgba(251,191,36,0.25)',
        decreasing_fillcolor='rgba(239,83,80,0.15)',
        opacity=0.8,
    ), row=1, col=1)

    # Linha de conexão
    fig.add_trace(go.Scatter(
        x=conn_x,
        y=[base_price, float(proj_abs[0, 0])],
        mode='lines', line=dict(color='#fbbf24', width=1.5, dash='dot'),
        showlegend=False, hoverinfo='skip',
    ), row=1, col=1)

    # Linha de base T0
    fig.add_hline(y=base_price, row=1, col=1,
                  line=dict(color='#8b949e', width=0.8, dash='dash'),
                  annotation_text=f"T₀ = {base_price:.3f}",
                  annotation_font=dict(color='#8b949e', size=9))

    # Vline separador
    if has_ts:
        fig.add_vline(x=int(pd.Timestamp(last_ts).timestamp() * 1000),
                       row=1, col=1, line=dict(color='#fbbf24', width=1, dash='dash'))

    # ── Painel 2: Close por especialista ──
    expert_keys = ['inception', 'lstm', 'transformer', 'mlp']
    for i, key in enumerate(expert_keys):
        if key in ind_abs:
            close_vals = ind_abs[key][:, 3]
            fig.add_trace(go.Scatter(
                x=fut_x, y=close_vals,
                mode='lines', name=EXPERT_NAMES[i],
                line=dict(color=EXPERT_COLORS[i], width=1.5),
                opacity=0.85,
            ), row=2, col=1)

    # MoE ponderado
    fig.add_trace(go.Scatter(
        x=fut_x, y=proj_abs[:, 3],
        mode='lines+markers', name='MoE Ponderado',
        line=dict(color=MOE_COLOR, width=2.5),
        marker=dict(size=4),
    ), row=2, col=1)

    # ── Painel 3: Gating weights (barras) ──
    for i, name in enumerate(EXPERT_NAMES):
        fig.add_trace(go.Bar(
            x=[name], y=[gating_w[i]],
            name=name, marker_color=EXPERT_COLORS[i],
            text=f"{gating_w[i]:.3f}", textposition='auto',
            textfont=dict(color='white', size=11),
            showlegend=False,
        ), row=3, col=1)

    # ── Layout global ──
    axis_style = dict(gridcolor=GRID_C, zerolinecolor=GRID_C,
                      tickfont=dict(color=TICK_C, size=9))
    fig.update_layout(
        height=950, width=1300,
        paper_bgcolor=PAPER, plot_bgcolor=BG,
        font=dict(color=TICK_C, family='Inter, sans-serif'),
        title=dict(
            text="MoE Inference Tool │ Projeção Futura │ USD/JPY",
            font=dict(size=15, color='#c9d1d9'), x=0.5,
        ),
        legend=dict(
            bgcolor='#161b22', bordercolor='#30363d', borderwidth=1,
            font=dict(color='#c9d1d9', size=9), orientation='h', y=-0.05,
        ),
        hovermode='x unified',
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
        xaxis3_rangeslider_visible=False,
    )

    for i in range(1, 4):
        fig.update_xaxes(**axis_style, row=i, col=1)
        fig.update_yaxes(**axis_style, row=i, col=1)

    fig.update_yaxes(title_text="Preço", row=1, col=1)
    fig.update_yaxes(title_text="Close", row=2, col=1)
    fig.update_yaxes(title_text="Peso (Softmax)", row=3, col=1)

    for ann in fig.layout.annotations:
        ann.update(font=dict(color='#c9d1d9', size=12))

    fig.write_html(save_path)
    print(f"\n  📊 Gráfico salvo: {save_path}")
    fig.show()


# =============================================================================
# 6. MAIN — ORQUESTRADOR
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='MoE Inference Tool — Projeção Futura')
    parser.add_argument('--model', default='../models/USDJPY/moe_model.keras', help='Caminho do modelo .keras')
    parser.add_argument('--config', default='../models/USDJPY/moe_config.json', help='Caminho do config .json')
    parser.add_argument('--data', default='../data/final/dataset_final.h5', help='Caminho do dataset .h5')
    parser.add_argument('--data-key', default='features', help='Key do HDF5')
    parser.add_argument('--output', default='../exports/plots/projecao_futura.html', help='Caminho do grafico HTML')
    parser.add_argument('--n-history', type=int, default=20, help='Barras historicas no grafico')
    args = parser.parse_args()

    print(f"\n{SEP}")
    print("  🧠 MoE INFERENCE TOOL — STANDALONE PROJECTION")
    print(f"  Modelo:  {args.model}")
    print(f"  Config:  {args.config}")
    print(f"  Dataset: {args.data}")
    print(SEP)

    # Validar arquivos
    for path, label in [(args.model, 'Modelo'), (args.config, 'Config'), (args.data, 'Dataset')]:
        if not os.path.exists(path):
            print(f"\n  ❌ {label} não encontrado: {path}")
            sys.exit(1)

    # 1. Config
    print("\n  [1/5] Carregando configuração...")
    cfg = load_config(args.config)
    window_size = cfg.get('window_size', 60)
    horizon = cfg.get('horizon', 15)
    print(f"        Window: {window_size} | Horizon: {horizon}")

    # 2. Dataset
    print("  [2/5] Carregando dataset...")
    df = load_dataset(args.data, args.data_key)
    print(f"        {len(df)} barras | {df.shape[1]} colunas")
    check_data_freshness(df)

    # 3. Modelo
    print("  [3/5] Carregando modelo MoE...")
    model = load_model(args.model, cfg)
    print("        ✅ Modelo carregado com sucesso.")

    # 4. Inferência
    print("  [4/5] Executando inferência...")
    X, base_price, feat_cols = prepare_inference_window(df, window_size)
    print(f"        Features ({len(feat_cols)}): {feat_cols[:5]}...")
    res = run_inference(model, X, base_price)
    print_terminal_report(res, horizon)

    # 5. Visualização
    print("  [5/5] Gerando visualização Plotly...")
    build_projection_chart(df, res, horizon, n_history=args.n_history, save_path=args.output)

    print(f"\n  ✅ Projeção concluída com sucesso!")
    print(f"  📄 Abra '{args.output}' no navegador para visualização.\n")


if __name__ == '__main__':
    main()
