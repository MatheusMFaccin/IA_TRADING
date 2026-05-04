# -*- coding: utf-8 -*-
"""
===============================================================================
MÓDULO: MoE GATING NETWORK — PROJEÇÃO GEOMÉTRICA DE SÉRIES TEMPORAIS
===============================================================================
Arquitetura de 4 especialistas neurais com Gating Network aprendível,
treinado end-to-end com Loss Morfológica (Soft-DTW + Curvature Penalty).

┌─────────────────────────────────────────────────────────────────────────┐
│                   MoE FORECASTER (End-to-End)                           │
│                                                                         │
│  ┌──────────────┐ ┌──────────────┐ ┌───────────────┐ ┌──────────────┐  │
│  │ Inception    │ │ LSTM Seq2Seq │ │  Transformer  │ │ Residual MLP │  │
│  │ Time (E1)    │ │ (E2)        │ │  Decoder (E3) │ │ N-BEATS (E4) │  │
│  │ 3 blocos     │ │ Enc-Dec     │ │  Causal Attn  │ │ Skip-conn.   │  │
│  │ filtros:     │ │ 128→64      │ │  4 heads      │ │ 256→512→256 │  │
│  │ 10, 20, 40   │ │             │ │               │ │              │  │
│  └──────┬───────┘ └──────┬──────┘ └───────┬───────┘ └──────┬───────┘  │
│         │                │                │                │           │
│         └────────────────┴────────────────┴────────────────┘           │
│                                    │ preds (4 × H × 4)                 │
│                                    │                                    │
│  RegimeGatingNetwork               │                                    │
│  EMA(10)[Hurst, GK_Vol] → ─────► Weighted Sum → Output(H, 4)          │
│  Dense(32)→LN→Dense(16)→Softmax(4)                                     │
│                                                                         │
│  Loss: α·Soft-DTW + β·Curvature + γ·Huber(Close)   α=0.5 β=0.2 γ=1.0  │
└─────────────────────────────────────────────────────────────────────────┘

Referências:
    - Fawaz et al. (2020): "InceptionTime: Finding AlexNet for Time Series"
    - Oreshkin et al. (2020): "N-BEATS: Neural Basis Expansion for TSF"
    - Cuturi & Blondel (2017): "Soft-DTW: a Differentiable Loss for TSF"
    - Jacobs et al. (1991): "Adaptive Mixtures of Local Experts"
===============================================================================
"""

from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
import warnings
import logging
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field, asdict

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Layer, RepeatVector,
    LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D,
    BatchNormalization, GaussianNoise, TimeDistributed, Reshape,
    Conv1D, MaxPooling1D, Concatenate, Add, Flatten, Lambda
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

logger = logging.getLogger('MoE_Gating')
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    ))
    logger.addHandler(handler)


# =============================================================================
# CONFIGURAÇÃO CENTRAL (Hyperparâmetros + Caminhos de Checkpoint)
# =============================================================================

@dataclass
class MoEConfig:
    """
    Configuração centralizada do pipeline MoE.

    Encapsula hiperparâmetros de treino e caminhos de persistência
    (checkpointing) em um único objeto serializável para JSON.

    Uso:
        config = MoEConfig(epochs=80, n_splits=5)
        config.to_json('moe_config.json')
        config = MoEConfig.from_json('moe_config.json')
    """
    # ── Dados ────────────────────────────────────────────────────────────
    input_file: str = '../data/final/dataset_final.h5'
    input_key: str = 'features'

    # ── Arquitetura ─────────────────────────────────────────────────────
    window_size: int = 60
    horizon: int = 15
    n_outputs: int = 4
    dropout: float = 0.2
    ema_periods: int = 10

    # ── Treino ──────────────────────────────────────────────────────────
    purge_bars: int = 200
    n_splits: int = 5
    epochs: int = 60
    batch_size: int = 64
    learning_rate: float = 1e-3
    loss_alpha: float = 0.5
    loss_beta: float = 0.2
    loss_gamma: float = 1.0
    huber_delta: float = 1.0
    confidence_level: float = 0.90

    # ── Persistência (Checkpointing) ────────────────────────────────────
    model_save_path: str = '../models/USDJPY/moe_model.keras'
    results_save_path: str = '../exports/inference/results_inference.parquet'
    config_save_path: str = '../models/USDJPY/moe_config.json'

    def to_json(self, path: Optional[str] = None) -> str:
        """Serializa config para JSON. Salva em disco se path fornecido."""
        data = asdict(self)
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        if path:
            tmp = path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                f.write(json_str)
            os.replace(tmp, path)
        return json_str

    @classmethod
    def from_json(cls, path: str) -> 'MoEConfig':
        """Carrega config de um arquivo JSON."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Filtrar apenas campos válidos do dataclass
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


# =============================================================================
# ÍNDICES DOS ALPHAS DE REGIME NO TENSOR DE FEATURES
# =============================================================================
# Serão detectados automaticamente via nome das colunas — ver _find_regime_cols()
HURST_COL_CANDIDATES = ['Alpha_Hurst_norm', 'Alpha_Hurst']
GK_VOL_COL_CANDIDATES = ['Alpha_GK_Vol_norm', 'Alpha_GK_Vol']


def _find_regime_col_indices(feature_cols: List[str]) -> Tuple[int, int]:
    """
    Encontra os índices de Hurst e GK_Vol na lista de features.
    Retorna (hurst_idx, gk_idx). Usa fallback para índices 0 e 1.
    """
    hurst_idx, gk_idx = 0, 1

    for candidate in HURST_COL_CANDIDATES:
        if candidate in feature_cols:
            hurst_idx = feature_cols.index(candidate)
            break

    for candidate in GK_VOL_COL_CANDIDATES:
        if candidate in feature_cols:
            gk_idx = feature_cols.index(candidate)
            break

    return hurst_idx, gk_idx


# =============================================================================
# ESPECIALISTA 1: CNN-1D INCEPTION TIME
# =============================================================================

class InceptionBlock(Layer):
    """
    Inception Block para séries temporais.

    4 branches paralelas:
        · Bottleneck 1×1 → Conv1D(kernel=10)
        · Bottleneck 1×1 → Conv1D(kernel=20)
        · Bottleneck 1×1 → Conv1D(kernel=40)
        · MaxPool(3) → Conv1D(1×1)  [preserva info de pico]

    Saídas concatenadas → LayerNorm → Residual

    A Bottleneck (kernel=1) reduz dimensionalidade antes das convoluções
    longas, controlando parâmetros e preservando gradiente.
    """

    def __init__(self, n_filters: int = 32, bottleneck_size: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.n_filters = n_filters
        self.bottleneck_size = bottleneck_size

        # Bottlenecks (redução de dimensionalidade)
        self.bn_10 = Conv1D(bottleneck_size, 1, padding='same', use_bias=False)
        self.bn_20 = Conv1D(bottleneck_size, 1, padding='same', use_bias=False)
        self.bn_40 = Conv1D(bottleneck_size, 1, padding='same', use_bias=False)

        # Convoluções multi-escala
        self.conv_10 = Conv1D(n_filters, 10, padding='same', activation='relu', use_bias=False)
        self.conv_20 = Conv1D(n_filters, 20, padding='same', activation='relu', use_bias=False)
        self.conv_40 = Conv1D(n_filters, 40, padding='same', activation='relu', use_bias=False)

        # Branch de MaxPool (captura picos)
        self.mp = MaxPooling1D(pool_size=3, strides=1, padding='same')
        self.conv_pool = Conv1D(n_filters, 1, padding='same', activation='relu', use_bias=False)

        # Fusão e normalização
        self.concat = Concatenate(axis=-1)
        self.norm = LayerNormalization()

        # Residual: projeta input para mesma dimensão da saída
        self.residual_proj = Conv1D(n_filters * 4, 1, padding='same', use_bias=False)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        residual = self.residual_proj(x)

        # Branch 1: kernel 10
        b1 = self.conv_10(self.bn_10(x))
        # Branch 2: kernel 20
        b2 = self.conv_20(self.bn_20(x))
        # Branch 3: kernel 40
        b3 = self.conv_40(self.bn_40(x))
        # Branch 4: MaxPool
        b4 = self.conv_pool(self.mp(x))

        out = self.concat([b1, b2, b3, b4])  # (B, T, 4*n_filters)
        out = self.norm(out + residual)        # Residual connection
        return out

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({'n_filters': self.n_filters, 'bottleneck_size': self.bottleneck_size})
        return cfg


def build_inception_forecaster(
    window_size: int,
    n_features: int,
    horizon: int,
    n_outputs: int = 4,
    n_blocks: int = 3,
    n_filters: int = 32,
    bottleneck_size: int = 16,
    dropout: float = 0.25
) -> Model:
    """
    InceptionTime para projeção OHLC multi-step.

    Arquitetura:
        Input(W, F) → GaussianNoise →
        3 × InceptionBlock(32 filtros × 4 branches) →
        GlobalAvgPool → Dropout → Dense(H×4) → Reshape(H,4)

    3 escalas temporais simultâneas (10, 20, 40 barras) permitem ao modelo
    capturar micro-padrões (10), padrões médios (20) e tendências (40)
    em uma única passagem, sem hierarquia forçada.

    LayerNorm (não BatchNorm) para estabilidade em dados financeiros
    não-estacionários onde a distribuição dos batches varia muito.
    """
    inp = Input(shape=(window_size, n_features), name='inception_input')
    x = GaussianNoise(0.005)(inp)

    for i in range(n_blocks):
        x = InceptionBlock(n_filters=n_filters, bottleneck_size=bottleneck_size,
                            name=f'inception_block_{i}')(x)

    x = GlobalAveragePooling1D()(x)
    x = LayerNormalization()(x)
    x = Dropout(dropout)(x)
    x = Dense(128, activation='gelu')(x)
    x = Dropout(dropout)(x)
    x = Dense(horizon * n_outputs)(x)
    out = Reshape((horizon, n_outputs), name='inception_output')(x)

    model = Model(inp, out, name='InceptionTime_Forecaster')
    return model


# =============================================================================
# ESPECIALISTA 2: LSTM SEQ2SEQ (mantido do llm.py — reimplementado aqui)
# =============================================================================

def build_lstm_seq2seq_moe(
    window_size: int,
    n_features: int,
    horizon: int,
    n_outputs: int = 4,
    encoder_units: int = 128,
    decoder_units: int = 64,
    dropout: float = 0.2
) -> Model:
    """
    LSTM Encoder-Decoder para captura de tendência macro.
    Especialidade: dependência temporal de longo prazo no Close.
    """
    enc_inp = Input(shape=(window_size, n_features), name='lstm_input')
    x = GaussianNoise(0.005)(enc_inp)
    x = LSTM(encoder_units, return_sequences=True)(x)
    x = Dropout(dropout)(x)
    context = LSTM(encoder_units // 2, return_sequences=False)(x)
    context = LayerNormalization()(context)

    x = RepeatVector(horizon)(context)
    x = LSTM(decoder_units, return_sequences=True)(x)
    x = Dropout(dropout)(x)
    x = LSTM(decoder_units // 2, return_sequences=True)(x)
    out = TimeDistributed(Dense(n_outputs), name='lstm_output')(x)

    return Model(enc_inp, out, name='LSTM_Seq2Seq')


# =============================================================================
# ESPECIALISTA 3: TRANSFORMER DECODER COM CAUSAL ATTENTION
# =============================================================================

class CausalSelfAttentionMoE(Layer):
    """Multi-Head Self-Attention com máscara causal (autoregressive)."""

    def __init__(self, num_heads: int, d_model: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.dropout_rate = dropout
        self.mha = MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout
        )
        self.norm = LayerNormalization()

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        seq_len = tf.shape(x)[1]
        mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        mask = tf.cast(mask, tf.bool)
        attn = self.mha(query=x, key=x, value=x,
                        attention_mask=mask, training=training)
        return self.norm(x + attn)

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({'num_heads': self.num_heads, 'd_model': self.d_model,
                    'dropout_rate': self.dropout_rate})
        return cfg


def build_transformer_decoder_moe(
    window_size: int,
    n_features: int,
    horizon: int,
    n_outputs: int = 4,
    num_heads: int = 4,
    d_model: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2
) -> Model:
    """
    Transformer Decoder: especialidade em reversões de regime e padrões globais.

    Extrai o ÚLTIMO estado oculto (last hidden state) ao invés de
    GlobalAveragePooling1D. Isso preserva a autocorrelação máxima do
    tick mais recente (t-0), eliminando o phase lag causado pela
    diluição temporal do pooling global.

    Justificativa Técnica:
        O GlobalAveragePooling misturava informação de barras distantes
        (t-59) com o tick corrente (t-0), produzindo um vetor de contexto
        "borrado" que subestimava sistematicamente o Close em regimes de
        quebra estrutural. O index slice x[:, -1, :] ancora a predição
        no estado causal mais recente da atenção.
    """
    inp = Input(shape=(window_size, n_features), name='transformer_input')
    x = Dense(d_model)(inp)
    x = LayerNormalization()(x)
    x = GaussianNoise(0.005)(x)

    for i in range(num_layers):
        x = CausalSelfAttentionMoE(num_heads=num_heads, d_model=d_model,
                                   dropout=dropout, name=f'causal_attn_{i}')(x)
        ffn = Dense(d_model * 2, activation='gelu')(x)
        ffn = Dropout(dropout)(ffn)
        ffn = Dense(d_model)(ffn)
        x = LayerNormalization()(x + ffn)

    # ── Last Hidden State (substitui GlobalAveragePooling1D) ─────────
    # Extrai apenas o último timestep da sequência causal.
    # Shape: (Batch, window_size, d_model) → (Batch, d_model)
    x = Lambda(lambda t: t[:, -1, :], name='last_hidden_state')(x)

    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(horizon * n_outputs)(x)
    out = Reshape((horizon, n_outputs), name='transformer_output')(x)

    return Model(inp, out, name='Transformer_Decoder')


# =============================================================================
# ESPECIALISTA 4: RESIDUAL MLP (N-BEATS STYLE)
# =============================================================================

class ResidualMLPBlock(Layer):
    """
    Bloco residual do N-BEATS: FC → ReLU → FC → ReLU + skip-connection.

    A skip-connection permite ao modelo aprender *incrementos* sobre
    a predição base, ao invés de aprender a predição absoluta — crucial
    para séries financeiras não-estacionárias onde a magnitude absoluta
    varia muito entre regimes.
    """

    def __init__(self, units: int, dropout: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.fc1 = Dense(units, activation='gelu')
        self.fc2 = Dense(units, activation='gelu')
        self.proj = Dense(units)  # Projeção para residual
        self.norm = LayerNormalization()
        self.drop = Dropout(dropout)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        residual = self.proj(x)
        x = self.fc1(x)
        x = self.drop(x, training=training)
        x = self.fc2(x)
        x = self.drop(x, training=training)
        return self.norm(x + residual)

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({'units': self.fc1.units, 'dropout': self.drop.rate})
        return cfg


def build_residual_mlp_forecaster(
    window_size: int,
    n_features: int,
    horizon: int,
    n_outputs: int = 4,
    hidden_units: int = 512,
    n_blocks: int = 4,
    dropout: float = 0.25
) -> Model:
    """
    Residual MLP (N-BEATS style) para projeção direta multi-step.

    Arquitetura:
        Flatten(W×F) → Dense(512) →
        4 × ResidualMLPBlock(512) →
        Dense(256) → Dense(H×4) → Reshape(H,4)

    Por que N-BEATS style para forex?
        - Séries de 60 barras × ~15 features = ~900 inputs: MLP puro
          converge rápido para esse regime de dimensionalidade.
        - Blocos residuais evitam que o modelo "esqueça" a escala do input.
        - Sem induction bias temporal (ao contrário do LSTM/Transformer):
          aprende correlações diretas entre features e H passos futuros.
        - Complementa os outros especialistas que têm forte prior temporal.
    """
    inp = Input(shape=(window_size, n_features), name='mlp_input')
    x = Flatten()(inp)
    x = GaussianNoise(0.005)(x)
    x = Dense(hidden_units, activation='gelu')(x)
    x = LayerNormalization()(x)

    for i in range(n_blocks):
        x = ResidualMLPBlock(hidden_units, dropout=dropout,
                              name=f'res_block_{i}')(x)

    x = Dense(256, activation='gelu')(x)
    x = Dropout(dropout)(x)
    x = Dense(horizon * n_outputs)(x)
    out = Reshape((horizon, n_outputs), name='mlp_output')(x)

    return Model(inp, out, name='ResidualMLP_NBeats')


# =============================================================================
# REGIME GATING NETWORK
# =============================================================================

class RegimeGatingNetwork(Layer):
    """
    Rede de Gating baseada em regime de mercado.

    Input:
        x: tensor 3D (Batch, window_size, n_features)
        Extrai EMA(10 barras) de [Alpha_Hurst, Alpha_GK_Vol] do tensor.

    EMA(10) via decay exponencial:
        α = 2 / (10 + 1) ≈ 0.182
        EMA_t = α·x_t + (1-α)·EMA_{t-1}

    Implementado como convolução 1D com pesos fixos (não treináveis)
    para manter compatibilidade com tf.function e batching eficiente.

    Architecture:
        [ema_hurst, ema_gkv] → Dense(32,ReLU) → LayerNorm →
        Dense(16, ReLU) → Dense(n_experts, Softmax)

    Interpretação dos pesos:
        · w_inception alto: regime de alta frequência / quebra de padrão
        · w_lstm alto: tendência estável de longo prazo (Hurst > 0.6)
        · w_transformer alto: reversão / regime de memória longa
        · w_mlp alto: regime de baixa volatilidade / sideways
    """

    def __init__(self, n_experts: int = 4,
                 hurst_idx: int = 0, gk_idx: int = 1,
                 ema_periods: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.n_experts = n_experts
        self.hurst_idx = hurst_idx
        self.gk_idx = gk_idx
        self.ema_periods = ema_periods
        self.alpha_ema = 2.0 / (ema_periods + 1)

        self.fc1 = Dense(32, activation='relu')
        self.norm = LayerNormalization()
        self.fc2 = Dense(16, activation='relu')
        self.out = Dense(n_experts, activation='softmax')

    def _compute_ema(self, series: tf.Tensor) -> tf.Tensor:
        """
        Computa EMA da série usando tf.scan (diferenciável).
        series: (Batch, window_size)
        Retorna EMA final: (Batch,)
        """
        alpha = self.alpha_ema
        one_minus = 1.0 - alpha

        # Transpor para (window_size, Batch) para usar tf.scan ao longo do tempo
        series_t = tf.transpose(series)  # (W, B)

        def ema_step(prev_ema, x_t):
            return alpha * x_t + one_minus * prev_ema

        # Inicializar com o primeiro valor da série
        init = series_t[0]
        ema_final = tf.scan(ema_step, series_t, initializer=init)
        # Retornar o último valor da EMA: (Batch,)
        return ema_final[-1]

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        x: (Batch, window_size, n_features)
        Retorna: (Batch, n_experts) — pesos softmax
        """
        # Pad feature dimension so indices hurst_idx/gk_idx are always valid.
        # When the dataset has fewer features than expected (e.g. only 'Returns'),
        # the fallback indices (0, 1) may exceed dim 2. Padding with zeros
        # ensures safe access — the gating simply receives zeros for missing features.
        required = max(self.hurst_idx, self.gk_idx) + 1
        n_feat   = tf.shape(x)[2]
        pad_n    = tf.maximum(0, required - n_feat)
        x_safe   = tf.pad(x, [[0, 0], [0, 0], [0, pad_n]])

        hurst_series = x_safe[:, :, self.hurst_idx]   # (B, W)
        gk_series    = x_safe[:, :, self.gk_idx]       # (B, W)

        # EMA(10) de cada série
        ema_hurst = self._compute_ema(hurst_series)  # (B,)
        ema_gk    = self._compute_ema(gk_series)      # (B,)

        # Concatenar como vetor de regime
        regime_vec = tf.stack([ema_hurst, ema_gk], axis=-1)  # (B, 2)

        # Gating MLP
        g = self.fc1(regime_vec)
        g = self.norm(g)
        g = self.fc2(g)
        weights = self.out(g)  # (B, n_experts)
        return weights

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({
            'n_experts': self.n_experts,
            'hurst_idx': self.hurst_idx,
            'gk_idx': self.gk_idx,
            'ema_periods': self.ema_periods,
        })
        return cfg


# =============================================================================
# LOSS MORFOLÓGICA: SOFT-DTW + CURVATURE PENALTY
# =============================================================================

def soft_dtw_loss(y_true: tf.Tensor, y_pred: tf.Tensor,
                  gamma: float = 1.0) -> tf.Tensor:
    """
    Soft-DTW diferenciável para garantia de similaridade de formato.

    Diferente do MSE que penaliza erros ponto-a-ponto, o Soft-DTW permite
    pequenas dilações temporais — um pico previsto 1 barra atrasado
    não é penalizado tanto quanto no MSE puro.

    Implementação: DP com softmin diferenciável usando Python loops.
    H=15 é estático → loops Python compilam correctamente em tf.function.

        softmin(a,b,c) = -γ · log(exp(-a/γ) + exp(-b/γ) + exp(-c/γ))

    Complexidade: O(H²) = O(225) por batch — negligível.

    y_true, y_pred: (Batch, H, 4) — opera sobre Close (dim 3)
    """
    y_t = y_true[:, :, 3]   # (B, H)
    y_p = y_pred[:, :, 3]   # (B, H)

    # Matriz de custo local D[b,i,j] = (y_t[b,i] - y_p[b,j])²
    y_t_exp = tf.expand_dims(y_t, 2)   # (B, H, 1)
    y_p_exp = tf.expand_dims(y_p, 1)   # (B, 1, H)
    D = tf.square(y_t_exp - y_p_exp)   # (B, H, H)

    H_static = y_t.shape[1] or 15      # valor estático para os loops Python
    INF = tf.constant(1e9, dtype=tf.float32)

    def softmin3(a, b, c):
        """Softmin diferenciável de 3 escalares (batch)."""
        stk = tf.stack([-a / gamma, -b / gamma, -c / gamma], axis=-1)
        return -gamma * tf.reduce_logsumexp(stk, axis=-1)

    # Matriz de acumulação R: lista de listas para indexação Python pura
    # R[i][j] = tensor (B,) — acumulação soft-DTW até (i,j)
    # i,j ∈ [0, H] (com bordas de guarda em 0)
    R = [[INF * tf.ones(tf.shape(y_t)[0]) for _ in range(H_static + 1)]
         for _ in range(H_static + 1)]
    R[0][0] = tf.zeros(tf.shape(y_t)[0])

    for i in range(1, H_static + 1):
        for j in range(1, H_static + 1):
            cost = D[:, i - 1, j - 1]          # (B,)
            r_min = softmin3(R[i-1][j-1], R[i-1][j], R[i][j-1])
            R[i][j] = cost + r_min

    return tf.reduce_mean(R[H_static][H_static])


def curvature_penalty(y_pred: tf.Tensor) -> tf.Tensor:
    """
    Penalidade na segunda derivada (curvatura) do Close projetado.

    Evita que o modelo projete linhas retas (flat lines), que representam
    convergência para a média estatística — o principal problema a resolver.

    Segunda derivada discreta:
        Δ²C_k = C_{k+2} - 2·C_{k+1} + C_k

    Penalidade: mean(Δ²C²)
    Quanto MAIOR a curvatura (picos/vales), MENOR a penalidade.
    Quanto mais flat, MAIOR a penalidade → força morfologia não-trivial.

    NOTA: esta penalidade é subtraída da loss (incentiva curvatura),
    ou melhor: usamos -curvature como regularização positiva.
    Na prática: adicionamos (1/curvature) ou usamos -mean(|Δ²|).
    """
    close = y_pred[:, :, 3]  # (B, H)

    # Primeira diferença: (B, H-1)
    d1 = close[:, 1:] - close[:, :-1]
    # Segunda diferença: (B, H-2)
    d2 = d1[:, 1:] - d1[:, :-1]

    # Queremos MAXIMIZAR curvatura → penalizar falta dela
    # Penalidade = 1 / (mean(|Δ²|) + ε) — mas não é convexa
    # Melhor: usar negative curvature como adicional à loss
    # penalty = - mean(|Δ²|) → somando à loss, minimizamos e maximizamos curvatura

    mean_curvature = tf.reduce_mean(tf.abs(d2))  # escalar

    # Retornamos a FALTA de curvatura: quanto mais flat, maior o valor
    # Usamos exp(-curvature) como proxy de "flatness"
    flatness = tf.exp(-10.0 * mean_curvature)
    return flatness


def morphological_loss(y_true: tf.Tensor, y_pred: tf.Tensor,
                        alpha: float = 0.5, beta: float = 0.2) -> tf.Tensor:
    """
    Loss Morfológica legada (mantida para compatibilidade).

    Para novos treinos, utilize hybrid_financial_loss() que adiciona
    ancoragem nominal via Huber Loss no Close.

        L = α·Soft-DTW(ŷ, y) + β·Curvature_Penalty(ŷ)
    """
    dtw = soft_dtw_loss(y_true, y_pred)
    curve = curvature_penalty(y_pred)

    H = tf.cast(tf.shape(y_pred)[1], tf.float32)
    dtw_norm = dtw / (H * H)

    return alpha * dtw_norm + beta * curve


def hybrid_financial_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    alpha: float = 0.5,
    beta: float = 0.2,
    gamma: float = 1.0,
    huber_delta: float = 1.0,
) -> tf.Tensor:
    """
    Loss Híbrida Financeira: Morfologia + Ancoragem Nominal.

    Combina três componentes complementares para eliminar o bias de
    subestimação do Close sem sacrificar a acurácia direcional:

        L_total = α · Soft-DTW(ŷ, y)
                + β · Curvature_Penalty(ŷ)
                + γ · Huber(y_true_close, y_pred_close)

    Componentes:
    ────────────
    · Soft-DTW (α=0.5): Similaridade de formato com tolerância a dilações
      temporais. Opera sobre o Close (dim 3) com DP diferenciável.
      Garante que a FORMA da trajetória projetada siga a real.

    · Curvature Penalty (β=0.2): Punição exponencial à flatness.
      Previne convergência para linhas retas (média estatística).
      Usa exp(-10·|Δ²C|) como proxy de "falta de curvatura".

    · Huber Loss (γ=1.0): Âncora nominal rígida no Close de TODOS
      os steps do horizonte. Age como contrapeso ao shrinkage do
      Soft-DTW, forçando alinhamento de magnitude ponto-a-ponto.
      O delta=1.0 garante robustez contra outliers de volatilidade:
        - |erro| ≤ δ → penalidade quadrática (precisa para erros pequenos)
        - |erro| > δ → penalidade linear (robusta contra spikes)

    Justificativa Financeira:
    ─────────────────────────
    O Soft-DTW sozinho sofre de "convexidade excessiva": o modelo
    sacrifica a precisão nominal do Close para minimizar o custo de
    alinhamento temporal. O Huber inserido em TODOS os H steps
    funciona como uma "âncora de preço" que impede o encolhimento
    de magnitude, sem introduzir sensibilidade a outliers que o MSE
    puro causaria em dados heteroscedásticos.

    Parameters
    ----------
    y_true : tf.Tensor
        Targets reais, shape (Batch, H, 4) — OHLC como retornos relativos.
    y_pred : tf.Tensor
        Predições do MoE, shape (Batch, H, 4).
    alpha : float
        Peso do Soft-DTW (similaridade de formato). Default: 0.5.
    beta : float
        Peso da Curvature Penalty (anti-flatline). Default: 0.2.
    gamma : float
        Peso do Huber Loss (ancoragem nominal do Close). Default: 1.0.
    huber_delta : float
        Threshold do Huber Loss para transição quadrática→linear.
        Default: 1.0 (robusto contra outliers de volatilidade).

    Returns
    -------
    tf.Tensor
        Escalar — loss total diferenciável para backpropagation.
    """
    # ── Componente 1: Soft-DTW (formato) ─────────────────────────────
    dtw = soft_dtw_loss(y_true, y_pred)
    H = tf.cast(tf.shape(y_pred)[1], tf.float32)
    dtw_norm = dtw / (H * H)

    # ── Componente 2: Curvature Penalty (anti-flatline) ──────────────
    curve = curvature_penalty(y_pred)

    # ── Componente 3: Huber Loss no Close — âncora nominal ──────────
    # Opera sobre TODOS os H steps do Close (dim 3), não apenas o
    # último, para forçar alinhamento de magnitude ao longo de todo
    # o horizonte de projeção.
    close_true = y_true[:, :, 3]   # (B, H)
    close_pred = y_pred[:, :, 3]   # (B, H)
    huber_fn = tf.keras.losses.Huber(delta=huber_delta, reduction='none')
    # huber_fn retorna loss por amostra: (B, H)
    huber_per_step = huber_fn(close_true, close_pred)  # (B, H)
    huber = tf.reduce_mean(huber_per_step)

    # ── Composição final ─────────────────────────────────────────────
    total = alpha * dtw_norm + beta * curve + gamma * huber
    return total


# =============================================================================
# MOE FORECASTER (MODELO PRINCIPAL — END-TO-END)
# =============================================================================

class MoEForecaster(tf.keras.Model):
    """
    Mixture of Experts treinável end-to-end.

    4 especialistas neurais + RegimeGatingNetwork + Loss Morfológica.

    O treinamento end-to-end permite que o Gating aprenda:
        · Dar peso à Inception em regimes de quebra de volatilidade (GK alto)
        · Dar peso ao LSTM em tendências longas (Hurst > 0.6)
        · Dar peso ao Transformer em reversões de regime
        · Dar peso ao MLP em regimes sideways (baixa volatilidade)

    Parâmetros:
    -----------
    window_size : int
        Janela de observação (W=60)
    n_features : int
        Número de features de entrada (N)
    horizon : int
        Passos futuros a projetar (H=15)
    n_outputs : int
        Número de outputs OHLC (4)
    hurst_idx : int
        Índice da feature Hurst no tensor de input
    gk_idx : int
        Índice da feature GK_Vol no tensor de input
    """

    def __init__(
        self,
        window_size: int,
        n_features: int,
        horizon: int,
        n_outputs: int = 4,
        hurst_idx: int = 0,
        gk_idx: int = 1,
        ema_periods: int = 10,
        dropout: float = 0.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.n_features = n_features
        self.horizon = horizon
        self.n_outputs = n_outputs
        self.hurst_idx = hurst_idx
        self.gk_idx = gk_idx

        # Especialistas
        self.expert_inception = build_inception_forecaster(
            window_size, n_features, horizon, n_outputs, dropout=dropout
        )
        self.expert_lstm = build_lstm_seq2seq_moe(
            window_size, n_features, horizon, n_outputs, dropout=dropout
        )
        self.expert_transformer = build_transformer_decoder_moe(
            window_size, n_features, horizon, n_outputs, dropout=dropout
        )
        self.expert_mlp = build_residual_mlp_forecaster(
            window_size, n_features, horizon, n_outputs, dropout=dropout
        )

        # Gating Network
        self.gating = RegimeGatingNetwork(
            n_experts=4,
            hurst_idx=hurst_idx,
            gk_idx=gk_idx,
            ema_periods=ema_periods,
            name='regime_gating'
        )

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass do MoE.

        x: (Batch, W, F)
        Retorna: (Batch, H, 4) — predição ponderada pelo Gating
        """
        # Predições dos 4 especialistas: cada (B, H, 4)
        p_inception   = self.expert_inception(x, training=training)
        p_lstm        = self.expert_lstm(x, training=training)
        p_transformer = self.expert_transformer(x, training=training)
        p_mlp         = self.expert_mlp(x, training=training)

        # Pesos do Gating: (B, 4)
        weights = self.gating(x, training=training)

        # Expandir pesos para broadcasting: (B, 1, 1, 4)
        w = tf.reshape(weights, [-1, 1, 1, 4])

        # Stack predições: (B, H, 4, 4)
        preds = tf.stack(
            [p_inception, p_lstm, p_transformer, p_mlp],
            axis=-1
        )  # (B, H, n_outputs, n_experts)

        # Soma ponderada: (B, H, 4)
        output = tf.reduce_sum(preds * w, axis=-1)
        return output

    def get_individual_predictions(
        self, x: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """Retorna predições individuais de cada especialista (sem gradiente)."""
        return {
            'inception':   self.expert_inception(x, training=False),
            'lstm':        self.expert_lstm(x, training=False),
            'transformer': self.expert_transformer(x, training=False),
            'mlp':         self.expert_mlp(x, training=False),
        }

    def get_gating_weights(self, x: tf.Tensor) -> np.ndarray:
        """
        Retorna pesos do Gating para análise de regime.
        Shape: (Batch, 4) → nomes: [inception, lstm, transformer, mlp]
        """
        return self.gating(x, training=False).numpy()

    def get_config(self) -> dict:
        return {
            'window_size': self.window_size,
            'n_features': self.n_features,
            'horizon': self.horizon,
            'n_outputs': self.n_outputs,
            'hurst_idx': self.hurst_idx,
            'gk_idx': self.gk_idx,
        }


# =============================================================================
# PREPARAÇÃO DE DADOS (compatível com llm.py)
# =============================================================================

def prepare_moe_data(
    df: pd.DataFrame,
    window_size: int = 60,
    horizon: int = 15,
    feature_cols: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Prepara sequências 3D para o MoEForecaster.

    Targets são retornos relativos ao último Close da janela:
        Δ_{t+k} = (Price_{t+k} - Close_t) / Close_t

    Compatível com o protocolo de `prepare_projection_data()` do llm.py.
    """
    if feature_cols is None:
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
        feature_cols = [c for c in candidates if c in df.columns]

    if not feature_cols:
        df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        feature_cols = ['Returns']

    for col in ['Open', 'High', 'Low', 'Close']:
        if col not in df.columns:
            raise ValueError(f"Coluna OHLC ausente: '{col}'")

    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    X_list, y_list, bp_list = [], [], []
    max_idx = len(df) - window_size - horizon
    features_matrix = df[feature_cols].values.astype('float32')
    for i in range(max_idx):
        window = features_matrix[i:i + window_size]
        base_close = float(df['Close'].iloc[i + window_size - 1])

        future = np.zeros((horizon, 4))
        for k in range(horizon):
            idx = i + window_size + k
            future[k, 0] = (df['Open'].iloc[idx]  - base_close) / base_close
            future[k, 1] = (df['High'].iloc[idx]  - base_close) / base_close
            future[k, 2] = (df['Low'].iloc[idx]   - base_close) / base_close
            future[k, 3] = (df['Close'].iloc[idx] - base_close) / base_close

        X_list.append(window)
        y_list.append(future)
        bp_list.append(base_close)

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
        np.array(bp_list, dtype=np.float64),
        feature_cols
    )


# =============================================================================
# UTILITÁRIOS DE CARREGAMENTO DE DADOS
# =============================================================================

def load_data_moe(
    input_file: str = 'dataset_final.h5',
    fallback_keys: Optional[List[Tuple[str, str]]] = None,
) -> Optional[pd.DataFrame]:
    """
    Carrega dados com fallback para múltiplos arquivos HDF5.

    Tenta, em ordem:
        1. (input_file, 'features')
        2. ('dataset_clean.h5', 'data')
        3. ('dataset_final.h5', 'features')

    Retorna None se nenhum arquivo for encontrado.
    """
    if fallback_keys is None:
        fallback_keys = [
            (input_file, 'features'),
            ('../data/processed/dataset_clean.h5', 'data'),
            ('../data/final/dataset_final.h5', 'features'),
        ]

    for filepath, key in fallback_keys:
        try:
            df = pd.read_hdf(filepath, key=key)
            print(f"  📂 Dados: {len(df)} barras de '{filepath}'")
            return df
        except (FileNotFoundError, KeyError):
            continue

    print("  ❌ Nenhum arquivo encontrado. Execute calcula_alphas.py primeiro.")
    return None
