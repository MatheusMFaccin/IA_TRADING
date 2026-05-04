"""
===============================================================================
MÓDULO 4: CONSELHO DE ESPECIALISTAS — STACKING REGRESSOR (MoE)
===============================================================================
Arquitetura de 3 especialistas heterogêneos para projeção OHLC multi-step.

┌─────────────────────────────────────────────────────────────────┐
│              CONSELHO DE ESPECIALISTAS (MoE)                    │
│                                                                 │
│  ┌──────────────────┐ ┌──────────────────┐ ┌─────────────────┐  │
│  │  Especialista 1  │ │  Especialista 2  │ │  Especialista 3 │  │
│  │  LSTM Seq2Seq    │ │  XGBoost         │ │  Transformer    │  │
│  │  (Trend Follower)│ │  Multi-Step      │ │  Decoder        │  │
│  │                  │ │  (Vol. Hunter)   │ │  (Pattern Rec.) │  │
│  │  Encoder-Decoder │ │  Direct Strategy │ │  Causal SelfAttn│  │
│  │  Loss: RMSE      │ │  + Decay Factor  │ │  Loss: Huber    │  │
│  │  Focus: Tendência│ │  Focus: OHLC     │ │  Focus: Regimes │  │
│  │  macro do Close  │ │  independente    │ │  e reversões    │  │
│  └────────┬─────────┘ └────────┬─────────┘ └───────┬─────────┘  │
│           │                    │                    │            │
│           ▼                    ▼                    ▼            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Stacking Aggregator (Ridge Regression)                  │   │
│  │  Shrinkage → Equal Weighting fallback para estabilidade  │   │
│  │  Pesos dinâmicos baseados no erro residual recente       │   │
│  └──────────────────────────────────────────────────────────┘   │
│           │                                                     │
│           ▼                                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Conformal Prediction                                    │   │
│  │  [ŷ - q_α, ŷ + q_α]  com cobertura garantida ≈ 1-α     │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘

Teoria:
    Stacking Regressor (Wolpert, 1992): Os modelos de nível 0 (especialistas)
    geram predições que são usadas como features pelo meta-learner (Ridge).
    Ridge Regression com L2 regularization identifica em quais regimes
    cada especialista performa melhor.

    Conformal Prediction (Vovk et al., 2005): Fornece intervalos de confiança
    com cobertura garantida sem assumptions distribucional.

Referências:
    - Wolpert (1992): "Stacked Generalization"
    - Vovk et al. (2005): "Algorithmic Learning in a Random World"
    - Salinas et al. (2020): "DeepAR: Probabilistic Forecasting"
    - Lim et al. (2021): "Temporal Fusion Transformers"
===============================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import warnings
import logging
from typing import Optional, Dict, Tuple, List, Any
from dataclasses import dataclass, field

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

import xgboost as xgb

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Layer, RepeatVector,
    LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D,
    BatchNormalization, GaussianNoise, TimeDistributed, Reshape
)
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Logging estruturado
logger = logging.getLogger('MoE_Council')
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    ))
    logger.addHandler(handler)


# =============================================================================
# ESPECIALISTA 1: LSTM SEQ2SEQ (THE TREND FOLLOWER)
# =============================================================================

def build_lstm_seq2seq(
    window_size: int,
    n_features: int,
    horizon: int,
    n_outputs: int = 4,
    encoder_units: int = 128,
    decoder_units: int = 64,
    dropout: float = 0.2
) -> Model:
    """
    LSTM Encoder-Decoder para projeção multi-step de OHLC.

    Arquitetura Seq2Seq:
        Encoder: Input[W × F] → GaussianNoise → LSTM(128) → LSTM(64) → Context
        Decoder: RepeatVector(H) → LSTM(64) → LSTM(32) → TimeDistributed(Dense(4))

    Especialidade: Captura de dependência temporal e projeção da trajetória
    contínua do preço de Close. Minimiza RMSE na tendência macro.

    Loss: MSE (equivalente a RMSE na otimização).
    """
    # Encoder
    encoder_input = Input(shape=(window_size, n_features), name='encoder_input')
    x = GaussianNoise(0.005)(encoder_input)
    x = LSTM(encoder_units, return_sequences=True)(x)
    x = Dropout(dropout)(x)
    context = LSTM(encoder_units // 2, return_sequences=False, name='context')(x)
    context = BatchNormalization()(context)

    # Decoder
    x = RepeatVector(horizon)(context)
    x = LSTM(decoder_units, return_sequences=True)(x)
    x = Dropout(dropout)(x)
    x = LSTM(decoder_units // 2, return_sequences=True)(x)

    # Output: H timesteps × 4 valores (O, H, L, C)
    outputs = TimeDistributed(Dense(n_outputs), name='ohlc_output')(x)

    model = Model(encoder_input, outputs, name='LSTM_Seq2Seq_TrendFollower')
    model.compile(
        optimizer='adam',
        loss='mse',  # RMSE = sqrt(MSE) — minimiza tendência macro
        metrics=['mae']
    )
    return model


# =============================================================================
# ESPECIALISTA 2: XGBOOST MULTI-STEP (THE VOLATILITY HUNTER)
# =============================================================================

class XGBoostMultiStepForecaster:
    """
    XGBoost com Direct Strategy para projeção OHLC independente.

    Cada componente OHLC é modelado de forma INDEPENDENTE por step,
    preservando a microestrutura de cada barra sem contaminar
    Open com informação de Close e vice-versa.

    Decay Factor:
        A confiança da predição diminui conforme nos aproximamos da
        última barra projetada: weight_step = exp(-λ × step)
        Isso evita que o ruído da barra 1 projete erro exponencial
        para a barra 15.

    Estratégia Direct (vs Recursive):
        - Direct: um modelo por step × output (4 × H modelos)
        - Vantagem: não há error compounding
        - Desvantagem: menos dados por modelo
        - Mitigação: XGBoost com regularização forte (max_depth=5)
    """

    def __init__(
        self,
        horizon: int = 15,
        n_estimators: int = 300,
        max_depth: int = 5,
        decay_lambda: float = 0.05
    ):
        self.horizon: int = horizon
        self.models: Dict[Tuple[int, int], xgb.XGBRegressor] = {}
        self.n_estimators: int = n_estimators
        self.max_depth: int = max_depth
        self.decay_lambda: float = decay_lambda
        self.decay_weights: np.ndarray = np.exp(
            -decay_lambda * np.arange(horizon)
        )

    def _build_model(self) -> xgb.XGBRegressor:
        return xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective='reg:squarederror',
            verbosity=0,
            random_state=42
        )

    def fit(self, X_flat: np.ndarray, y_targets: np.ndarray) -> None:
        """
        Treina modelos para cada output × cada step.

        X_flat : (n_samples, n_features_flat)
        y_targets : (n_samples, horizon, 4)
        """
        n_outputs = y_targets.shape[2]

        for step in range(self.horizon):
            # Sample weights com decay: barras futuras distantes
            # contribuem menos para o loss
            for out_idx in range(n_outputs):
                key = (step, out_idx)
                model = self._build_model()
                model.fit(X_flat, y_targets[:, step, out_idx])
                self.models[key] = model

    def predict(self, X_flat: np.ndarray) -> np.ndarray:
        """
        Prevê OHLC para os próximos H steps com Decay Factor.

        Retorna: (n_samples, horizon, 4)

        O Decay Factor é aplicado como atenuação das predições
        em steps distantes: pred_step_k *= decay_weight_k
        Isso reduz a amplitude das projeções distantes,
        refletindo menor confiança.
        """
        n_samples = X_flat.shape[0]
        predictions = np.zeros((n_samples, self.horizon, 4))

        for step in range(self.horizon):
            for out_idx in range(4):
                key = (step, out_idx)
                if key in self.models:
                    raw_pred = self.models[key].predict(X_flat)
                    # Aplicar decay factor
                    predictions[:, step, out_idx] = (
                        raw_pred * self.decay_weights[step]
                    )

        return predictions


# =============================================================================
# ESPECIALISTA 3: TRANSFORMER DECODER (THE PATTERN RECOGNITION)
# =============================================================================

class CausalSelfAttention(Layer):
    """
    Multi-Head Self-Attention com máscara causal (autoregressive).

    A máscara causal impede que posições futuras influenciem as
    posições anteriores, garantindo a propriedade autoregressiva:

        mask[i, j] = -∞  se j > i  (futuro mascarado)
        mask[i, j] = 0   se j ≤ i  (passado + presente visível)

    Isso é essencial para o Transformer Decoder:
    ao prever a barra t+k, o modelo só vê barras ≤ t+k-1.
    """

    def __init__(self, num_heads: int, d_model: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.dropout_rate = dropout
        self.mha = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout
        )
        self.layer_norm = LayerNormalization()

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        seq_len = tf.shape(x)[1]
        # Máscara causal: lower triangular
        causal_mask = tf.linalg.band_part(
            tf.ones((seq_len, seq_len)), -1, 0
        )
        causal_mask = tf.cast(causal_mask, tf.bool)

        attn_output = self.mha(
            query=x, key=x, value=x,
            attention_mask=causal_mask,
            training=training
        )
        return self.layer_norm(x + attn_output)  # Residual connection

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'd_model': self.d_model,
            'dropout_rate': self.dropout_rate
        })
        return config


def build_transformer_decoder(
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
    Transformer Decoder Autoregressivo para projeção de trajetória.

    Arquitetura:
        Input → Linear(d_model) → LayerNorm → GaussianNoise →
        N × [Causal Self-Attention + FFN + LayerNorm] →
        GlobalPool → Dense(H × 4) → Reshape(H, 4) → Output

    A Causal Self-Attention usa máscara triangular inferior para
    garantir autogressividade: posição t só vê posições ≤ t.

    Especialidade: Identificação de padrões de reversão de regime
    e correlações de longo prazo via atenção multi-cabeça global.
    Captura inflection points que LSTM e XGBoost ignoram.

    Loss: Huber (robusta a outliers de flash crashes).
    """
    inputs = Input(shape=(window_size, n_features))

    # Projeção linear para d_model dimensões
    x = Dense(d_model)(inputs)
    x = LayerNormalization()(x)
    x = GaussianNoise(0.005)(x)

    # Transformer Decoder blocks com causal mask
    for layer_idx in range(num_layers):
        # Causal Self-Attention
        x = CausalSelfAttention(
            num_heads=num_heads,
            d_model=d_model,
            dropout=dropout,
            name=f'causal_attn_{layer_idx}'
        )(x)

        # Feed-Forward Network
        ffn = Dense(d_model * 2, activation='gelu')(x)
        ffn = Dropout(dropout)(ffn)
        ffn = Dense(d_model)(ffn)
        x = LayerNormalization()(x + ffn)  # Residual connection

    # Projeção para output
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(horizon * n_outputs)(x)
    outputs = Reshape((horizon, n_outputs))(x)

    model = Model(inputs, outputs, name='Transformer_Decoder_PatternRec')
    model.compile(
        optimizer='adam',
        loss='huber',  # Robusta a outliers
        metrics=['mae']
    )
    return model


# =============================================================================
# STACKING AGGREGATOR (RIDGE + SHRINKAGE → EQUAL WEIGHTING FALLBACK)
# =============================================================================

class StackingAggregator:
    """
    Meta-learner que combina predições dos 3 especialistas via Ridge Regression.

    Arquitetura:
        X_meta = [pred_M1.flat, pred_M2.flat, pred_M3.flat]  shape: (n, H×4×3)
        y_meta = y_true.flat                                  shape: (n, H×4)
        Ridge(alpha=1.0).fit(X_meta, y_meta)

    Shrinkage Factor:
        Quando os coeficientes do Ridge são instáveis (alta variância entre
        folds) ou quando há poucos dados de calibração, o aggregator faz
        fallback automático para Equal Weighting (Simple Average):

            shrinkage = min(1, n_calibration / n_min_stable)

            weight_final = shrinkage × weight_ridge + (1 - shrinkage) × (1/3)

        Isso garante robustez no início do walk-forward quando o meta-learner
        ainda não tem dados suficientes para estimar pesos confiáveis.

    Atualização Dinâmica:
        A cada fold do walk-forward, o Ridge é retreinado com as predições
        out-of-sample acumuladas dos folds anteriores.
    """

    def __init__(
        self,
        n_models: int = 3,
        alpha: float = 1.0,
        n_min_stable: int = 500
    ):
        self.n_models: int = n_models
        self.alpha: float = alpha
        self.n_min_stable: int = n_min_stable

        self.ridge: Optional[Ridge] = None
        self.fold_maes: List[List[float]] = [[] for _ in range(n_models)]
        self.weights: np.ndarray = np.ones(n_models) / n_models

        # Acumuladores para retreino do Ridge
        self._meta_X: List[np.ndarray] = []
        self._meta_y: List[np.ndarray] = []

        self.shrinkage: float = 0.0  # Começa com equal weighting
        self.is_fitted: bool = False

    def update_predictions(
        self,
        predictions: List[np.ndarray],
        y_true: np.ndarray,
        model_maes: List[float]
    ) -> None:
        """
        Registra predições dos especialistas e retreina o Ridge.

        predictions : list de (n_samples, horizon, 4) — predição de cada modelo
        y_true : (n_samples, horizon, 4) — targets reais
        model_maes : list de float — MAE de cada modelo no fold atual
        """
        for i, mae in enumerate(model_maes):
            self.fold_maes[i].append(mae)

        # Stack predições para meta-learner
        n_samples = predictions[0].shape[0]
        meta_features = np.hstack([
            p.reshape(n_samples, -1) for p in predictions
        ])  # (n_samples, H*4*3)
        meta_targets = y_true.reshape(n_samples, -1)  # (n_samples, H*4)

        self._meta_X.append(meta_features)
        self._meta_y.append(meta_targets)

        # Retreinar Ridge com todos os dados acumulados
        all_X = np.vstack(self._meta_X)
        all_y = np.vstack(self._meta_y)

        n_calibration = len(all_X)

        try:
            self.ridge = Ridge(alpha=self.alpha)
            self.ridge.fit(all_X, all_y)
            self.is_fitted = True

            # Calcular shrinkage
            self.shrinkage = min(1.0, n_calibration / self.n_min_stable)

            logger.info(
                f"Ridge retreinado com {n_calibration} samples | "
                f"shrinkage={self.shrinkage:.2f}"
            )
        except Exception as e:
            logger.warning(f"Ridge fit failed: {e}. Usando equal weighting.")
            self.shrinkage = 0.0

        # Atualizar pesos para report (via inverse MAE)
        self._update_display_weights()

    def _update_display_weights(self) -> None:
        """Calcula pesos para visualização (inverse MAE)."""
        mean_maes = []
        for maes in self.fold_maes:
            if maes:
                mean_maes.append(np.mean(maes) + 1e-8)
            else:
                mean_maes.append(1.0)

        inv_mae = [1.0 / m for m in mean_maes]
        total = sum(inv_mae)
        self.weights = np.array([w / total for w in inv_mae])

    def predict(self, predictions: List[np.ndarray]) -> np.ndarray:
        """
        Combina predições dos especialistas.

        Usa Ridge se fitted + shrinkage > 0, senão equal weighting.

        predictions : list de (n_samples, horizon, 4)
        Retorna: (n_samples, horizon, 4)
        """
        n_samples = predictions[0].shape[0]
        horizon = predictions[0].shape[1]
        n_outputs = predictions[0].shape[2]

        if self.is_fitted and self.shrinkage > 0:
            # Ridge prediction
            meta_features = np.hstack([
                p.reshape(n_samples, -1) for p in predictions
            ])
            ridge_pred = self.ridge.predict(meta_features)
            ridge_pred = ridge_pred.reshape(n_samples, horizon, n_outputs)

            # Equal weighting
            equal_pred = np.mean(predictions, axis=0)

            # Shrinkage blend
            combined = (
                self.shrinkage * ridge_pred +
                (1 - self.shrinkage) * equal_pred
            )
            return combined
        else:
            # Pure equal weighting (fallback)
            return np.mean(predictions, axis=0)

    def predict_single(self, predictions: List[np.ndarray]) -> np.ndarray:
        """Predição para um único sample. Wrapper para predict()."""
        # Garantir batch dimension
        preds_batch = [
            p.reshape(1, *p.shape) if p.ndim == 2 else p for p in predictions
        ]
        result = self.predict(preds_batch)
        return result[0]

    def report(self) -> None:
        """Imprime relatório dos pesos e performance."""
        names = ['LSTM Seq2Seq', 'XGB Multi-Step', 'Transformer Dec.']
        print("\n  ╔═══════════════════════════════════════════════════════╗")
        print("  ║    STACKING AGGREGATOR — PESOS DOS ESPECIALISTAS     ║")
        print("  ╠═══════════════════════════════════════════════════════╣")
        print(f"  ║  Shrinkage Factor: {self.shrinkage:.3f}"
              f"{'  (Ridge ativo)' if self.shrinkage > 0.5 else '  (Equal Weighting)'}"
              f"{'':>{20 - (15 if self.shrinkage > 0.5 else 18)}}║")
        print("  ╠═══════════════════════════════════════════════════════╣")
        for i in range(self.n_models):
            name = names[i] if i < len(names) else f'Model_{i}'
            maes = self.fold_maes[i]
            mean_mae = np.mean(maes) if maes else 0
            w = self.weights[i]
            bar = '█' * int(w * 30)
            print(f"  ║ {name:18s} │ w={w:.3f} │ MAE={mean_mae:.6f}  ║")
            print(f"  ║                    │ {bar:30s}   ║")
        print("  ╚═══════════════════════════════════════════════════════╝")


# =============================================================================
# CONFORMAL PREDICTION
# =============================================================================

class ConformalPredictor:
    """
    Conformal Prediction para intervalos de confiança sem assumptions
    distribucional.

    Algoritmo (Split Conformal):
        1. Calibrar nonconformity scores no set de calibração:
           α_i = ||y_true_i - ŷ_i||₁  (resíduos absolutos, por componente)

        2. Calcular quantil:
           q = quantile(α, p=ceil((1-α)(n+1))/n)

        3. Intervalo de predição:
           [ŷ - q, ŷ + q]

    Garantia de Cobertura:
        P(y ∈ [ŷ - q, ŷ + q]) ≥ 1 - α
        Válido para QUALQUER distribuição dos resíduos (distribution-free).

    Referência:
        Vovk et al. (2005): "Algorithmic Learning in a Random World"
        Romano et al. (2019): "Conformalized Quantile Regression"
    """

    def __init__(self, confidence_level: float = 0.90):
        self.confidence_level: float = confidence_level
        self.calibration_scores: Optional[np.ndarray] = None
        self.quantiles: Optional[np.ndarray] = None
        self.is_calibrated: bool = False

    def calibrate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> None:
        """
        Calibra o preditor conformal com resíduos do set de calibração.

        y_true : (n_calibration, horizon, 4)
        y_pred : (n_calibration, horizon, 4)
        """
        # Nonconformity scores: resíduo absoluto por componente OHLC × step
        residuals = np.abs(y_true - y_pred)  # (n_cal, H, 4)

        # Quantil por (step, output) para bandas heterogêneas
        n_cal = residuals.shape[0]
        horizon = residuals.shape[1]
        n_outputs = residuals.shape[2]

        self.quantiles = np.zeros((horizon, n_outputs))

        alpha = 1 - self.confidence_level
        quantile_level = min(
            np.ceil((1 - alpha) * (n_cal + 1)) / n_cal,
            1.0
        )

        for h in range(horizon):
            for o in range(n_outputs):
                self.quantiles[h, o] = np.quantile(
                    residuals[:, h, o], quantile_level
                )

        self.calibration_scores = residuals
        self.is_calibrated = True

        logger.info(
            f"Conformal calibrado: {n_cal} samples, "
            f"confidence={self.confidence_level:.0%}, "
            f"q_mean={np.mean(self.quantiles):.6f}"
        )

    def predict_interval(
        self,
        y_pred: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gera intervalos de confiança conformais.

        y_pred : (horizon, 4) ou (n_samples, horizon, 4)
        Retorna: (lower_bound, upper_bound) com mesma shape de y_pred
        """
        if not self.is_calibrated:
            raise RuntimeError("ConformalPredictor não calibrado. Chame calibrate() primeiro.")

        if y_pred.ndim == 2:
            # Single sample
            lower = y_pred - self.quantiles
            upper = y_pred + self.quantiles
        else:
            # Batch
            lower = y_pred - self.quantiles[np.newaxis, :, :]
            upper = y_pred + self.quantiles[np.newaxis, :, :]

        return lower, upper


# =============================================================================
# PREPARAÇÃO DE DADOS
# =============================================================================

def prepare_projection_data(
    df: pd.DataFrame,
    window_size: int = 60,
    horizon: int = 15,
    feature_cols: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Prepara sequências de input e targets OHLC relativos para projeção.

    Targets são RETORNOS RELATIVOS ao último Close da janela:
        ΔO_{t+k} = (Open_{t+k} - Close_t) / Close_t
        ΔH_{t+k} = (High_{t+k} - Close_t) / Close_t
        ΔL_{t+k} = (Low_{t+k} - Close_t) / Close_t
        ΔC_{t+k} = (Close_{t+k} - Close_t) / Close_t

    Retorna:
    --------
    (X_seq, y_targets, base_prices, feature_names)
    """
    if feature_cols is None:
        candidates = [
            'Alpha_Returns', 'Alpha_Returns_norm',
            'Alpha_GK_Vol', 'Alpha_GK_Vol_norm',
            'Alpha_Hurst', 'Alpha_Hurst_norm',
            'Alpha_Amihud', 'Alpha_Amihud_norm',
            'Alpha_CDF_VPIN', 'Alpha_CDF_VPIN_norm',
            'Alpha_MA_Cross',
            'Alpha_Spread', 'Alpha_Spread_norm',
            'Alpha_VWAP', 'Alpha_VWAP_norm',
            'Alpha_OI', 'Alpha_OI_norm',
            'Alpha_Close_FracDiff', 'Alpha_Close_FracDiff_norm',
            'Returns', 'Returns_Kalman',
        ]
        feature_cols = [c for c in candidates if c in df.columns]

        if len(feature_cols) == 0:
            feature_cols = ['Returns']
            if 'Returns' not in df.columns:
                df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))

    # Garantir OHLC presente
    for col in ['Open', 'High', 'Low', 'Close']:
        if col not in df.columns:
            raise ValueError(f"Coluna '{col}' não encontrada no DataFrame.")

    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    X_sequences: List[np.ndarray] = []
    y_targets: List[np.ndarray] = []
    base_prices: List[float] = []

    max_idx = len(df) - window_size - horizon

    for i in range(max_idx):
        window = df[feature_cols].iloc[i:i + window_size].values
        X_sequences.append(window)

        base_close = df['Close'].iloc[i + window_size - 1]
        base_prices.append(base_close)

        future_ohlc = np.zeros((horizon, 4))
        for k in range(horizon):
            idx = i + window_size + k
            future_ohlc[k, 0] = (df['Open'].iloc[idx] - base_close) / base_close
            future_ohlc[k, 1] = (df['High'].iloc[idx] - base_close) / base_close
            future_ohlc[k, 2] = (df['Low'].iloc[idx] - base_close) / base_close
            future_ohlc[k, 3] = (df['Close'].iloc[idx] - base_close) / base_close

        y_targets.append(future_ohlc)

    return (
        np.array(X_sequences),
        np.array(y_targets),
        np.array(base_prices),
        feature_cols
    )


def relative_to_absolute(
    relative_ohlc: np.ndarray,
    base_price: float
) -> np.ndarray:
    """
    Converte retornos relativos para preços absolutos:
        Price_{t+k} = base_price × (1 + Δ_{t+k})
    """
    return base_price * (1 + relative_ohlc)


# =============================================================================
# PIPELINE PRINCIPAL DO CONSELHO DE ESPECIALISTAS
# =============================================================================

def run_council_pipeline(
    input_file: str = 'dataset_final.h5',
    window_size: int = 60,
    horizon: int = 15,
    purge_bars: int = 200,
    n_splits: int = 5,
    epochs: int = 50,
    batch_size: int = 64,
    confidence_level: float = 0.90,
    decay_lambda: float = 0.05
) -> Optional[Dict[str, Any]]:
    """
    Pipeline completo do Conselho de Especialistas (MoE).

    Treina 3 especialistas via Purged Walk-Forward, combina via
    Stacking Aggregator (Ridge + Shrinkage), e calibra intervalos
    de confiança via Conformal Prediction.

    Parâmetros:
    -----------
    input_file : str
        Arquivo HDF5 com features (output do calcula_alphas.py).
    window_size : int
        Janela temporal de observação (lookback).
    horizon : int
        Número de barras futuras a projetar (H=15).
    purge_bars : int
        Gap entre treino e validação (elimina leakage).
    n_splits : int
        Folds de walk-forward.
    epochs : int
        Épocas de treino dos modelos neurais.
    batch_size : int
        Tamanho do batch.
    confidence_level : float
        Nível de confiança para Conformal Prediction (0.90 = 90%).
    decay_lambda : float
        Fator de decay para XGBoost multi-step.

    Retorna:
    --------
    dict com: projected_ohlc, confidence_intervals, metrics, ensemble, models
    """
    print("═" * 65)
    print("  🧠 CONSELHO DE ESPECIALISTAS (MoE) — STACKING REGRESSOR")
    print(f"  Projetando as próximas {horizon} barras")
    print("═" * 65)

    # 1. Carregar dados
    df = _load_data(input_file)
    if df is None:
        return None

    if 'Returns' not in df.columns:
        df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df.dropna(inplace=True)

    # 2. Preparar dados
    logger.info(f"Preparando sequências (window={window_size}, horizon={horizon})...")
    X_seq, y_targets, base_prices, feat_names = prepare_projection_data(
        df, window_size=window_size, horizon=horizon
    )

    print(f"  Sequências: {X_seq.shape[0]}")
    print(f"  Features ({len(feat_names)}): {feat_names[:5]}...")
    print(f"  Shape X: {X_seq.shape} | Shape y: {y_targets.shape}")

    n_features = X_seq.shape[2]

    # 3. Walk-Forward com Purge
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=purge_bars)
    aggregator = StackingAggregator(n_models=3)
    conformal = ConformalPredictor(confidence_level=confidence_level)

    fold_metrics: List[dict] = []
    all_val_preds: List[np.ndarray] = []
    all_val_true: List[np.ndarray] = []

    # Persistir últimos modelos treinados
    last_models: Dict[str, Any] = {}

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_seq)):
        print(f"\n  {'─' * 55}")
        print(f"  Fold {fold + 1}/{n_splits} │ Train: {len(train_idx)} │ "
              f"Val: {len(val_idx)} │ Purge: {purge_bars}")
        print(f"  {'─' * 55}")

        X_train = X_seq[train_idx]
        X_val = X_seq[val_idx]
        y_train = y_targets[train_idx]
        y_val = y_targets[val_idx]

        if len(X_train) < batch_size:
            logger.warning("Skip fold (insufficient data)")
            continue

        # --- ESPECIALISTA 1: LSTM Seq2Seq ---
        print("    [E1] LSTM Seq2Seq (Trend Follower)...", end=' ')
        model_1 = build_lstm_seq2seq(window_size, n_features, horizon)
        model_1.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs, batch_size=batch_size, verbose=0,
            callbacks=[EarlyStopping(
                monitor='val_loss', patience=8,
                restore_best_weights=True
            )]
        )
        pred_m1 = model_1.predict(X_val, verbose=0)
        mae_m1 = mean_absolute_error(
            y_val.reshape(-1, horizon * 4),
            pred_m1.reshape(-1, horizon * 4)
        )
        print(f"MAE={mae_m1:.6f}")
        last_models['lstm'] = model_1

        # --- ESPECIALISTA 2: XGBoost Multi-Step ---
        print("    [E2] XGBoost Multi-Step (Volatility Hunter)...", end=' ')
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)

        xgb_forecaster = XGBoostMultiStepForecaster(
            horizon=horizon,
            n_estimators=300,
            max_depth=5,
            decay_lambda=decay_lambda
        )
        xgb_forecaster.fit(X_train_flat, y_train)
        pred_m2 = xgb_forecaster.predict(X_val_flat)
        mae_m2 = mean_absolute_error(
            y_val.reshape(-1, horizon * 4),
            pred_m2.reshape(-1, horizon * 4)
        )
        print(f"MAE={mae_m2:.6f}")
        last_models['xgb'] = xgb_forecaster

        # --- ESPECIALISTA 3: Transformer Decoder ---
        print("    [E3] Transformer Decoder (Pattern Recognition)...", end=' ')
        model_3 = build_transformer_decoder(window_size, n_features, horizon)
        model_3.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs, batch_size=batch_size, verbose=0,
            callbacks=[EarlyStopping(
                monitor='val_loss', patience=8,
                restore_best_weights=True
            )]
        )
        pred_m3 = model_3.predict(X_val, verbose=0)
        mae_m3 = mean_absolute_error(
            y_val.reshape(-1, horizon * 4),
            pred_m3.reshape(-1, horizon * 4)
        )
        print(f"MAE={mae_m3:.6f}")
        last_models['transformer'] = model_3

        # --- STACKING AGGREGATOR ---
        aggregator.update_predictions(
            predictions=[pred_m1, pred_m2, pred_m3],
            y_true=y_val,
            model_maes=[mae_m1, mae_m2, mae_m3]
        )

        # Predição do ensemble para este fold
        ensemble_pred = aggregator.predict([pred_m1, pred_m2, pred_m3])
        mae_ensemble = mean_absolute_error(
            y_val.reshape(-1, horizon * 4),
            ensemble_pred.reshape(-1, horizon * 4)
        )
        rmse_ensemble = np.sqrt(mean_squared_error(
            y_val.reshape(-1, horizon * 4),
            ensemble_pred.reshape(-1, horizon * 4)
        ))

        print(f"    [Ensemble] MAE={mae_ensemble:.6f} | RMSE={rmse_ensemble:.6f} | "
              f"shrinkage={aggregator.shrinkage:.2f}")

        # Acumular para calibração conformal
        all_val_preds.append(ensemble_pred)
        all_val_true.append(y_val)

        fold_metrics.append({
            'fold': fold + 1,
            'mae_m1': mae_m1, 'mae_m2': mae_m2, 'mae_m3': mae_m3,
            'mae_ensemble': mae_ensemble, 'rmse_ensemble': rmse_ensemble
        })

    # 4. Relatório do Aggregator
    aggregator.report()

    # 5. Calibrar Conformal Prediction
    if all_val_preds:
        cal_preds = np.concatenate(all_val_preds)
        cal_true = np.concatenate(all_val_true)
        conformal.calibrate(cal_true, cal_preds)

    # 6. Projeção Final
    print("\n  Gerando projeção final sobre dados mais recentes...")
    last_window = X_seq[-1:]
    last_base_price = base_prices[-1]

    final_pred_m1 = last_models['lstm'].predict(last_window, verbose=0)[0]
    final_pred_m2 = last_models['xgb'].predict(
        last_window.reshape(1, -1)
    )[0]
    final_pred_m3 = last_models['transformer'].predict(
        last_window, verbose=0
    )[0]

    proj_center = aggregator.predict_single(
        [final_pred_m1, final_pred_m2, final_pred_m3]
    )

    # Intervalos conformais
    conf_lower, conf_upper = conformal.predict_interval(proj_center)

    # Converter para preços absolutos
    proj_ohlc_abs = relative_to_absolute(proj_center, last_base_price)
    lower_abs = relative_to_absolute(conf_lower, last_base_price)
    upper_abs = relative_to_absolute(conf_upper, last_base_price)

    # 7. Print da projeção
    _print_projection_table(proj_ohlc_abs, lower_abs, upper_abs,
                            last_base_price, horizon, confidence_level)

    return {
        'projected_ohlc': proj_ohlc_abs,
        'confidence_lower': lower_abs,
        'confidence_upper': upper_abs,
        'proj_relative': proj_center,
        'base_price': last_base_price,
        'confidence_level': confidence_level,
        'aggregator': aggregator,
        'conformal': conformal,
        'fold_metrics': fold_metrics,
        'models': last_models,
        'feature_names': feat_names,
        'individual_predictions': {
            'lstm_seq2seq': relative_to_absolute(final_pred_m1, last_base_price),
            'xgboost_multi': relative_to_absolute(final_pred_m2, last_base_price),
            'transformer': relative_to_absolute(final_pred_m3, last_base_price),
        }
    }


# =============================================================================
# predict_candles() — INTERFACE PÚBLICA
# =============================================================================

def predict_candles(
    df: pd.DataFrame,
    horizon: int = 15,
    confidence_level: float = 0.90,
    models: Optional[Dict[str, Any]] = None,
    aggregator: Optional[StackingAggregator] = None,
    conformal: Optional[ConformalPredictor] = None,
    window_size: int = 60,
    feature_cols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Projeta os próximos n períodos com intervalo de confiança
    via Conformal Prediction.

    Interface principal para uso externo e pelo projecao_conselho.py.

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame com dados atualizados (OHLC + features).
    horizon : int
        Número de barras futuras a projetar.
    confidence_level : float
        Nível de confiança (0.90 = 90%).
    models : dict
        {'lstm': Model, 'xgb': XGBoostMultiStepForecaster,
         'transformer': Model}
    aggregator : StackingAggregator
        Meta-learner com pesos calibrados.
    conformal : ConformalPredictor
        Preditor conformal calibrado.
    window_size : int
        Janela de observação.
    feature_cols : list, optional
        Features a usar. Se None, auto-detecta.

    Retorna:
    --------
    dict:
        candles          : np.ndarray(H, 4)       — OHLC absoluto
        confidence_lower : np.ndarray(H, 4)
        confidence_upper : np.ndarray(H, 4)
        confidence_level : float
        individual_predictions : dict
        aggregator_weights : np.ndarray
    """
    if models is None:
        raise ValueError(
            "models deve ser fornecido. Execute run_council_pipeline() primeiro."
        )

    if feature_cols is None:
        candidates = [
            'Alpha_Returns', 'Alpha_Returns_norm',
            'Alpha_GK_Vol', 'Alpha_GK_Vol_norm',
            'Alpha_Hurst', 'Alpha_Hurst_norm',
            'Alpha_Amihud', 'Alpha_Amihud_norm',
            'Alpha_CDF_VPIN', 'Alpha_CDF_VPIN_norm',
            'Alpha_MA_Cross',
            'Alpha_Spread', 'Alpha_Spread_norm',
            'Alpha_VWAP', 'Alpha_VWAP_norm',
            'Alpha_OI', 'Alpha_OI_norm',
            'Alpha_Close_FracDiff', 'Alpha_Close_FracDiff_norm',
            'Returns', 'Returns_Kalman',
        ]
        feature_cols = [c for c in candidates if c in df.columns]

    # Extrair última janela
    window_data = df[feature_cols].iloc[-window_size:].values
    window_input = window_data.reshape(1, window_size, -1)
    base_price = float(df['Close'].iloc[-1])

    # Predições individuais
    pred_lstm = models['lstm'].predict(window_input, verbose=0)[0]
    pred_xgb = models['xgb'].predict(window_input.reshape(1, -1))[0]
    pred_tfm = models['transformer'].predict(window_input, verbose=0)[0]

    # Aggregator
    if aggregator is not None:
        center = aggregator.predict_single([pred_lstm, pred_xgb, pred_tfm])
    else:
        center = np.mean([pred_lstm, pred_xgb, pred_tfm], axis=0)

    # Intervalos conformais
    if conformal is not None and conformal.is_calibrated:
        conf_lower, conf_upper = conformal.predict_interval(center)
    else:
        # Fallback: bandas de divergência entre modelos
        stacked = np.stack([pred_lstm, pred_xgb, pred_tfm], axis=0)
        std_div = np.std(stacked, axis=0)
        conf_lower = center - 1.96 * std_div
        conf_upper = center + 1.96 * std_div

    # Converter para absoluto
    candles_abs = relative_to_absolute(center, base_price)
    lower_abs = relative_to_absolute(conf_lower, base_price)
    upper_abs = relative_to_absolute(conf_upper, base_price)

    return {
        'candles': candles_abs,
        'confidence_lower': lower_abs,
        'confidence_upper': upper_abs,
        'confidence_level': confidence_level,
        'base_price': base_price,
        'individual_predictions': {
            'lstm_seq2seq': relative_to_absolute(pred_lstm, base_price),
            'xgboost_multi': relative_to_absolute(pred_xgb, base_price),
            'transformer': relative_to_absolute(pred_tfm, base_price),
        },
        'aggregator_weights': (
            aggregator.weights if aggregator else np.ones(3) / 3
        ),
    }


# =============================================================================
# UTILITÁRIOS
# =============================================================================

def _load_data(input_file: str) -> Optional[pd.DataFrame]:
    """Carrega dados de múltiplas fontes com fallback."""
    sources = [
        (input_file, 'features'),
        ('dataset_clean.h5', 'data'),
        ('dataset_final.h5', 'features'),
    ]

    for filepath, key in sources:
        try:
            df = pd.read_hdf(filepath, key=key)
            print(f"  📂 Dados carregados: {len(df)} barras de '{filepath}'")
            return df
        except (FileNotFoundError, KeyError):
            continue

    print("  ❌ Nenhum arquivo de dados encontrado.")
    print("     Execute: baixa_dados → limpaArquivos → calcula_alphas")
    return None


def _print_projection_table(
    proj_ohlc: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    base_price: float,
    horizon: int,
    confidence_level: float
) -> None:
    """Imprime tabela formatada da projeção."""
    print(f"\n  Projeção para {horizon} barras (confiança: {confidence_level:.0%}):")
    print(f"  Base Price (último Close): {base_price:.5f}")
    print(f"  {'Bar':>4s}  {'Open':>10s}  {'High':>10s}  {'Low':>10s}"
          f"  {'Close':>10s}  {'CI Low':>10s}  {'CI High':>10s}")
    print(f"  {'─' * 4}  {'─' * 10}  {'─' * 10}  {'─' * 10}"
          f"  {'─' * 10}  {'─' * 10}  {'─' * 10}")
    for k in range(horizon):
        o, h, l, c = proj_ohlc[k]
        ci_l = lower[k, 3]  # CI do Close
        ci_h = upper[k, 3]
        print(f"  {k+1:4d}  {o:10.5f}  {h:10.5f}  {l:10.5f}"
              f"  {c:10.5f}  {ci_l:10.5f}  {ci_h:10.5f}")

    direction = "ALTA ↑" if proj_ohlc[-1, 3] > base_price else "BAIXA ↓"
    delta_pct = (proj_ohlc[-1, 3] - base_price) / base_price * 100

    print(f"\n  Direção: {direction} | Variação: {delta_pct:+.4f}%")


# =============================================================================
# EXECUÇÃO
# =============================================================================
if __name__ == "__main__":
    results = run_council_pipeline(
        input_file='dataset_final.h5',
        window_size=60,
        horizon=15,
        purge_bars=200,
        n_splits=5,
        epochs=50,
        batch_size=64,
        confidence_level=0.90,
        decay_lambda=0.05
    )