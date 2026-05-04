# -*- coding: utf-8 -*-
"""
===============================================================================
MÓDULO: MoE TRAINING + TICK-TO-DAILY AGGREGATION + CHECKPOINTING
===============================================================================
Script PRINCIPAL de execução do pipeline MoE.

Responsabilidades:
    1. Carregar e processar os dados (HDF5 → features)
    2. Instanciar e treinar o MoEForecaster (Walk-Forward + Purge)
    3. Salvar checkpoints:
       · moe_model.keras          — modelo treinado (TF Keras)
       · results_inference.parquet — predições vs targets
       · moe_config.json          — hiperparâmetros e métricas
    4. (Opcional) Consolidar tick bars → daily candles + comparação visual

Hierarquia de chamadas:
    moe_gating.py  (definições de modelo)  ←  importa classes
    moe_to_daily.py  ★ ESTE ARQUIVO ★      ←  treino + persistência
    moe_visualization.py                    ←  lê .parquet (standalone)

Convenções de mercado Forex (USD/JPY):
    - O "dia" de trading começa às 17:00 ET (NY Close) do dia anterior
      e termina às 17:00 ET do dia corrente.

Referências:
    - López de Prado (2018): "Advances in Financial Machine Learning" — Cap. 2
    - Dacorogna et al. (2001): "An Introduction to High-Frequency Finance"
===============================================================================
"""

from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
import warnings
import logging
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field


warnings.filterwarnings("ignore")

logger = logging.getLogger("MoE_Daily")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(_handler)





# =============================================================================
# CLASSE: TimeSeriesAggregator — Data Wrangling
# =============================================================================


@dataclass
class TimeSeriesAggregator:
    """
    Consolida tick bars em candles de timeframe superior via resample OHLC.

    O "dia" Forex é delimitado pelo NY Close (17:00 ET). Para isso,
    os timestamps são convertidos para US/Eastern e deslocados
    (-17h) de forma que o corte de ``pd.Grouper(freq='1D')`` coincida
    com 17:00 ET.

    Attributes
    ----------
    ny_close_hour : int
        Hora de corte do dia Forex no fuso US/Eastern (default=17).
    min_ticks_per_candle : int
        Mínimo de tick bars necessário para formar um candle diário
        válido. Dias com menos ticks são descartados (filtro de
        liquidez insuficiente — fins de semana, feriados parciais).
    """

    ny_close_hour: int = 17
    min_ticks_per_candle: int = 10

    # ── Aggregate real tick bars → Daily OHLCV ──────────────────────────────

    def aggregate_ticks_to_daily(
        self,
        df_ticks: pd.DataFrame,
        timestamp_col: str = "Date",
    ) -> pd.DataFrame:
        """
        Resample tick bars em candles diários usando OHLC padrão.

        Parameters
        ----------
        df_ticks : pd.DataFrame
            DataFrame com tick bars. Colunas obrigatórias: ``timestamp_col``,
            ``Open``, ``High``, ``Low``, ``Close``.
            Opcionais: ``Volume``, ``Tick_Count``.
        timestamp_col : str
            Nome da coluna de timestamp (default ``'Date'``).

        Returns
        -------
        pd.DataFrame
            Candles diários com índice de data (truncado ao dia), colunas
            ``Open``, ``High``, ``Low``, ``Close``, ``Volume``, ``Tick_Count``.

        Raises
        ------
        ValueError
            Se o DataFrame não contiver colunas OHLC obrigatórias ou se
            não houver ticks suficientes para compor ao menos 1 candle.
        """
        self._validate_ohlc_columns(df_ticks)

        df = df_ticks.copy()

        # ── Timestamp → DatetimeIndex ────────────────────────────────────
        if timestamp_col in df.columns:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df = df.set_index(timestamp_col)
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                f"Coluna '{timestamp_col}' não encontrada e o índice não é "
                "DatetimeIndex. Forneça timestamps válidos."
            )

        # ── Converter para NY (US/Eastern) e aplicar offset ─────────────
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert("US/Eastern")

        # Desloca -17h p/ que o corte de Grouper('1D') caia em 17:00 ET
        df.index = df.index - pd.Timedelta(hours=self.ny_close_hour)

        # ── Resample ────────────────────────────────────────────────────
        ohlc_agg: Dict[str, Any] = {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
        }
        if "Volume" in df.columns:
            ohlc_agg["Volume"] = "sum"
        if "Tick_Count" in df.columns:
            ohlc_agg["Tick_Count"] = "sum"

        daily = df.resample("1D").agg(ohlc_agg)

        # ── Reverter offset do índice para data de NY Close ─────────────
        daily.index = daily.index + pd.Timedelta(hours=self.ny_close_hour)
        daily.index = daily.index.normalize()  # truncar para data pura
        daily.index.name = "Date"

        # ── Filtrar dias incompletos ────────────────────────────────────
        # Recomputa contagem de ticks por dia para filtragem
        tick_counts_day = df.resample("1D").size()
        valid_mask = tick_counts_day >= self.min_ticks_per_candle
        # Alinhar máscara com daily
        valid_dates = tick_counts_day[valid_mask].index + pd.Timedelta(
            hours=self.ny_close_hour
        )
        valid_dates = valid_dates.normalize()
        daily = daily[daily.index.isin(valid_dates)]

        daily = daily.dropna(subset=["Open", "Close"])

        if len(daily) == 0:
            raise ValueError(
                f"Nenhum candle diário formado. Total de tick bars: "
                f"{len(df_ticks)} (mín. requerido por candle: "
                f"{self.min_ticks_per_candle})."
            )

        logger.info(
            f"Aggregated {len(df_ticks)} tick bars → {len(daily)} daily candles"
        )
        return daily

    # ── Aggregate MoE projection ticks → Daily shadow candles ───────────────

    def aggregate_projection_to_daily(
        self,
        proj_ohlc: np.ndarray,
        timestamps: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        Mapeia a projeção MoE (H tick-bars) p/ candles diários projetados.

        A projeção original tem H passos no espaço de tick bars. Para
        convertê-los em candles diários, agrupamos os passos que caem
        no mesmo dia calendário (usando NY Close como delimitador) e
        agregamos com OHLC padrão sobre o array de projeção.

        Parameters
        ----------
        proj_ohlc : np.ndarray
            Array (H, 4) com [Open, High, Low, Close] projetados por barra.
        timestamps : pd.DatetimeIndex
            Timestamps estimados para cada tick-bar projetada. Deve ter
            comprimento H, alinhado 1:1 com ``proj_ohlc``.

        Returns
        -------
        pd.DataFrame
            Candles diários da projeção MoE.
        """
        if len(proj_ohlc) != len(timestamps):
            raise ValueError(
                f"proj_ohlc ({len(proj_ohlc)}) e timestamps ({len(timestamps)}) "
                "devem ter o mesmo comprimento."
            )

        df_proj = pd.DataFrame(
            proj_ohlc,
            columns=["Open", "High", "Low", "Close"],
            index=timestamps,
        )
        df_proj.index.name = "Date"

        # Aplicar mesma lógica de offset NY Close
        if df_proj.index.tz is None:
            df_proj.index = df_proj.index.tz_localize("UTC")
        df_proj.index = df_proj.index.tz_convert("US/Eastern")
        df_proj.index = df_proj.index - pd.Timedelta(hours=self.ny_close_hour)

        daily_proj = df_proj.resample("1D").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
        )

        daily_proj.index = daily_proj.index + pd.Timedelta(
            hours=self.ny_close_hour
        )
        daily_proj.index = daily_proj.index.normalize()
        daily_proj.index.name = "Date"

        daily_proj = daily_proj.dropna(subset=["Open", "Close"])

        logger.info(
            f"Projection: {len(proj_ohlc)} tick bars → "
            f"{len(daily_proj)} daily candles"
        )
        return daily_proj

    # ── Estimador de timestamps para projeções sem timestamp ────────────────

    @staticmethod
    def estimate_projection_timestamps(
        last_timestamp: pd.Timestamp,
        horizon: int,
        avg_bar_duration: pd.Timedelta,
    ) -> pd.DatetimeIndex:
        """
        Estima timestamps futuros para a projeção MoE.

        Usa a duração média observada entre tick bars para extrapolar
        os próximos H timestamps a partir do último bar histórico.

        Parameters
        ----------
        last_timestamp : pd.Timestamp
            Timestamp do último tick bar histórico.
        horizon : int
            Número de barras projetadas (H).
        avg_bar_duration : pd.Timedelta
            Duração média entre tick bars observada nos dados históricos.

        Returns
        -------
        pd.DatetimeIndex
            H timestamps estimados para as barras projetadas.
        """
        # Garantir os tipos explícitos de Timestamp e Timedelta para evitar TypeError
        ts = pd.Timestamp(last_timestamp)
        delta = pd.Timedelta(avg_bar_duration)
        return pd.DatetimeIndex(
            [ts + delta * (k + 1) for k in range(horizon)]
        )

    # ── Validação ──────────────────────────────────────────────────────────

    @staticmethod
    def _validate_ohlc_columns(df: pd.DataFrame) -> None:
        """Verifica presença das colunas OHLC obrigatórias."""
        required = {"Open", "High", "Low", "Close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Colunas OHLC ausentes: {missing}. "
                f"Colunas disponíveis: {list(df.columns)}"
            )


# =============================================================================
# PERSISTÊNCIA: CHECKPOINTING ATÔMICO
# =============================================================================


def _safe_save_parquet(df: pd.DataFrame, path: str) -> None:
    """
    Salva DataFrame como Parquet com escrita atômica.

    Escreve primeiro em um .tmp e depois faz rename atômico,
    evitando corrupção se o processo for interrompido durante a escrita.
    """
    tmp_path = path + '.tmp'
    try:
        df.to_parquet(tmp_path, engine='pyarrow')
        os.replace(tmp_path, path)
        logger.info(f"Parquet salvo: {path}")
    except Exception as e:
        # Limpar tmp corrompido
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise IOError(f"Falha ao salvar {path}: {e}") from e


def save_checkpoint(
    results: Dict[str, Any],
    config: 'MoEConfig',
    fold_metrics: List[dict],
) -> None:
    """
    Salva todos os artefatos do treino de forma segura.

    Artefatos salvos:
        1. moe_model.keras         — modelo TF Keras
        2. results_inference.parquet — predições + targets + gating
        3. moe_config.json         — hiperparâmetros + fold_metrics

    Parameters
    ----------
    results : dict
        Output do run_moe_pipeline() com projected_ohlc, confidence,
        individual_predictions, gating_weights_final, etc.
    config : MoEConfig
        Configuração do pipeline.
    fold_metrics : list[dict]
        Métricas por fold (MAE, gating weights, epochs).
    """
    from moe_gating import MoEConfig  # type guard

    print(f"\n{'─' * 65}")
    print("  💾 SALVANDO CHECKPOINTS...")
    print(f"{'─' * 65}")

    # Garantir que os diretórios de destino existam
    for p in [config.model_save_path, config.results_save_path, config.config_save_path]:
        parent = os.path.dirname(p)
        if parent:
            os.makedirs(parent, exist_ok=True)

    # ── 1. Modelo .keras ─────────────────────────────────────────────────
    model = results.get('model')
    if model is not None:
        try:
            model.save(config.model_save_path)
            print(f"  ✅ Modelo: {config.model_save_path}")
        except Exception as e:
            print(f"  ⚠️ Falha ao salvar modelo: {e}")
            # Fallback: salvar pesos
            try:
                weights_path = config.model_save_path.replace('.keras', '_weights.h5')
                model.save_weights(weights_path)
                print(f"  ✅ Pesos (fallback): {weights_path}")
            except Exception as e2:
                print(f"  ❌ Falha ao salvar pesos: {e2}")

    # ── 2. Results .parquet ──────────────────────────────────────────────
    try:
        proj_ohlc = results['projected_ohlc']           # (H, 4)
        lower_abs = results.get('confidence_lower')       # (H, 4)
        upper_abs = results.get('confidence_upper')       # (H, 4)
        ind_preds = results.get('individual_predictions', {})
        gating_final = results.get('gating_weights_final', np.ones(4) / 4)
        base_price = results.get('base_price', proj_ohlc[0, 0])
        horizon = len(proj_ohlc)

        rows = []
        for h in range(horizon):
            row = {
                'step': h + 1,
                'Open_proj': proj_ohlc[h, 0],
                'High_proj': proj_ohlc[h, 1],
                'Low_proj': proj_ohlc[h, 2],
                'Close_proj': proj_ohlc[h, 3],
            }

            if lower_abs is not None:
                row['Open_lower'] = lower_abs[h, 0]
                row['High_lower'] = lower_abs[h, 1]
                row['Low_lower'] = lower_abs[h, 2]
                row['Close_lower'] = lower_abs[h, 3]

            if upper_abs is not None:
                row['Open_upper'] = upper_abs[h, 0]
                row['High_upper'] = upper_abs[h, 1]
                row['Low_upper'] = upper_abs[h, 2]
                row['Close_upper'] = upper_abs[h, 3]

            # Predições individuais dos experts (Close)
            for key in ['inception', 'lstm', 'transformer', 'mlp']:
                if key in ind_preds:
                    arr = ind_preds[key]
                    row[f'Close_{key}'] = arr[h, 3] if arr.ndim == 2 else arr[h]

            # Gating weights (repetidos por step)
            gating_labels = ['inception', 'lstm', 'transformer', 'mlp']
            for i, gl in enumerate(gating_labels):
                row[f'gating_{gl}'] = float(gating_final[i])

            row['base_price'] = base_price
            rows.append(row)

        df_results = pd.DataFrame(rows)

        # Metadados como atributos do DataFrame
        df_results.attrs['confidence_level'] = results.get('confidence_level', 0.90)
        df_results.attrs['gating_labels'] = json.dumps(
            results.get('gating_labels', ['InceptionTime', 'LSTM', 'Transformer', 'ResidualMLP'])
        )

        _safe_save_parquet(df_results, config.results_save_path)
        print(f"  ✅ Resultados: {config.results_save_path}")

    except Exception as e:
        print(f"  ❌ Falha ao salvar resultados: {e}")

    # ── 3. Config .json ──────────────────────────────────────────────────
    try:
        # Adicionar fold_metrics ao JSON de config
        config_data = json.loads(config.to_json())
        # Converter gating_weights (ndarray) para lista para serialização
        serializable_metrics = []
        for m in fold_metrics:
            m_copy = dict(m)
            if 'gating_weights' in m_copy:
                m_copy['gating_weights'] = m_copy['gating_weights'].tolist() \
                    if hasattr(m_copy['gating_weights'], 'tolist') \
                    else list(m_copy['gating_weights'])
            serializable_metrics.append(m_copy)
        config_data['fold_metrics'] = serializable_metrics
        config_data['gating_weights_final'] = gating_final.tolist() \
            if hasattr(gating_final, 'tolist') else list(gating_final)

        tmp = config.config_save_path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, config.config_save_path)
        print(f"  ✅ Config: {config.config_save_path}")

    except Exception as e:
        print(f"  ❌ Falha ao salvar config: {e}")

    print(f"{'─' * 65}")


# =============================================================================
# PIPELINE PRINCIPAL: TREINO + PERSISTÊNCIA (absorvido de moe_gating.py)
# =============================================================================


def run_moe_pipeline(
    config: Optional['MoEConfig'] = None,
    **kwargs,
) -> Optional[Dict[str, Any]]:
    """
    Pipeline completo do MoE Gating Network com checkpointing.

    1. Carrega dados do HDF5
    2. Prepara sequências (Batch, 60, N_features)
    3. Walk-Forward com Purge (protocolo rigoroso anti-leakage)
    4. Treina MoEForecaster end-to-end com Loss Morfológica
    5. Gera projeção final + análise de pesos do Gating
    6. ★ Salva checkpoints (.keras + .parquet + .json)

    Parameters
    ----------
    config : MoEConfig, optional
        Configuração centralizada. Se None, usa defaults + kwargs.
    **kwargs
        Overrides para MoEConfig (ex: epochs=80, n_splits=3).

    Returns
    -------
    dict ou None
        Resultados completos compatíveis com projecao_conselho.py.
    """
    # Lazy imports para evitar carga desnecessária
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error
    from moe_gating import (
        MoEForecaster, MoEConfig, prepare_moe_data,
        hybrid_financial_loss, _find_regime_col_indices, load_data_moe,
    )
    from llm import ConformalPredictor, relative_to_absolute

    # ── Config ───────────────────────────────────────────────────────────
    if config is None:
        config = MoEConfig(**kwargs)

    print("═" * 65)
    print("  🧠 MoE GATING NETWORK — PROJEÇÃO GEOMÉTRICA + CHECKPOINT")
    print(f"  Especialistas: InceptionTime | LSTM | Transformer | ResidualMLP")
    print(f"  Loss: Soft-DTW (α={config.loss_alpha}) + Curvature (β={config.loss_beta}) + Huber (γ={config.loss_gamma}, δ={config.huber_delta})")
    print(f"  Gating: EMA({config.ema_periods})[Hurst, GK_Vol] → Softmax(4)")
    print(f"  Persistência: {config.model_save_path} + {config.results_save_path}")
    print("═" * 65)

    # 1. Carregar dados
    df = load_data_moe(config.input_file)
    if df is None:
        return None

    # Validar que as colunas essenciais existem
    required_cols = ['Open', 'High', 'Low', 'Close']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"  ❌ Colunas OHLC ausentes no dataset: {missing}")
        print(f"  Verifique se o arquivo '{config.input_file}' foi gerado corretamente.")
        return None

    if 'Returns' not in df.columns:
        df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df.dropna(inplace=True)

    # 2. Preparar sequências
    logger.info(f"Preparando dados (window={config.window_size}, horizon={config.horizon})...")
    X_seq, y_targets, base_prices, feat_names = prepare_moe_data(
        df, window_size=config.window_size, horizon=config.horizon
    )

    n_features = X_seq.shape[2]
    hurst_idx, gk_idx = _find_regime_col_indices(feat_names)

    print(f"  Sequências: {X_seq.shape[0]} | Features: {n_features}")
    
    hurst_name = feat_names[hurst_idx] if hurst_idx < len(feat_names) else 'Not Found'
    gk_name = feat_names[gk_idx] if gk_idx < len(feat_names) else 'Not Found'
    
    print(f"  Gating inputs → Hurst[{hurst_idx}]='{hurst_name}' | "
          f"GK[{gk_idx}]='{gk_name}'")

    # 3. Walk-Forward com Purge
    tscv = TimeSeriesSplit(n_splits=config.n_splits, gap=config.purge_bars)
    fold_metrics: List[dict] = []
    all_val_preds: List[np.ndarray] = []
    all_val_true: List[np.ndarray] = []
    last_moe: Optional[MoEForecaster] = None

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_seq)):
        print(f"\n  {'─' * 55}")
        print(f"  Fold {fold + 1}/{config.n_splits} │ "
              f"Train: {len(train_idx)} │ Val: {len(val_idx)} │ Purge: {config.purge_bars}")
        print(f"  {'─' * 55}")

        X_train, X_val = X_seq[train_idx], X_seq[val_idx]
        y_train, y_val = y_targets[train_idx], y_targets[val_idx]

        if len(X_train) < config.batch_size * 2:
            logger.warning(f"Fold {fold+1}: dados insuficientes — skipping")
            continue

        # Instanciar MoE fresco por fold
        moe = MoEForecaster(
            window_size=config.window_size,
            n_features=n_features,
            horizon=config.horizon,
            n_outputs=config.n_outputs,
            hurst_idx=hurst_idx,
            gk_idx=gk_idx,
            ema_periods=config.ema_periods,
        )

        # Compilar com loss híbrida financeira (Soft-DTW + Curvature + Huber)
        alpha_val, beta_val = config.loss_alpha, config.loss_beta
        gamma_val, delta_val = config.loss_gamma, config.huber_delta
        moe.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
            loss=lambda y_t, y_p: hybrid_financial_loss(
                y_t, y_p, alpha_val, beta_val, gamma_val, delta_val
            ),
        )

        # Build explícito
        moe(X_train[:1])

        print(f"  Parâmetros totais: "
              f"{sum(np.prod(v.shape) for v in moe.trainable_variables):,}")

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10,
                          restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=5, min_lr=1e-5, verbose=0),
        ]

        history = moe.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config.epochs,
            batch_size=config.batch_size,
            callbacks=callbacks,
            verbose=0
        )

        # Avaliação do fold
        val_pred = moe(X_val, training=False).numpy()
        mae_moe = mean_absolute_error(
            y_val.reshape(-1, config.horizon * 4),
            val_pred.reshape(-1, config.horizon * 4)
        )

        # Predições individuais
        ind = moe.get_individual_predictions(X_val)
        mae_inc = mean_absolute_error(y_val.reshape(-1), ind['inception'].numpy().reshape(-1))
        mae_lst = mean_absolute_error(y_val.reshape(-1), ind['lstm'].numpy().reshape(-1))
        mae_tfm = mean_absolute_error(y_val.reshape(-1), ind['transformer'].numpy().reshape(-1))
        mae_mlp = mean_absolute_error(y_val.reshape(-1), ind['mlp'].numpy().reshape(-1))

        # Pesos médios do Gating neste fold
        gating_w = moe.get_gating_weights(X_val).mean(axis=0)

        print(f"  MAE → MoE={mae_moe:.6f} | "
              f"Inc={mae_inc:.6f} | LSTM={mae_lst:.6f} | "
              f"TFM={mae_tfm:.6f} | MLP={mae_mlp:.6f}")
        print(f"  Gating médio → "
              f"Inc={gating_w[0]:.3f} | LSTM={gating_w[1]:.3f} | "
              f"TFM={gating_w[2]:.3f} | MLP={gating_w[3]:.3f}")

        all_val_preds.append(val_pred)
        all_val_true.append(y_val)

        fold_metrics.append({
            'fold': fold + 1,
            'mae_moe': mae_moe,
            'mae_inception': mae_inc,
            'mae_lstm': mae_lst,
            'mae_transformer': mae_tfm,
            'mae_mlp': mae_mlp,
            'gating_weights': gating_w,
            'epochs_trained': len(history.history['loss']),
        })

        last_moe = moe

    if last_moe is None:
        logger.error("Nenhum fold completou o treino.")
        return None

    # 4. Calibração Conformal
    conformal = ConformalPredictor(confidence_level=config.confidence_level)
    if all_val_preds:
        cal_preds = np.concatenate(all_val_preds)
        cal_true = np.concatenate(all_val_true)
        conformal.calibrate(cal_true, cal_preds)

    # 5. Projeção Final
    print("\n  Gerando projeção final sobre dados mais recentes...")
    last_window = X_seq[-1:]
    last_base = float(base_prices[-1])

    proj_rel = last_moe(last_window, training=False).numpy()[0]
    ind_final = last_moe.get_individual_predictions(last_window)
    gating_final = last_moe.get_gating_weights(last_window)[0]

    conf_lower_rel, conf_upper_rel = conformal.predict_interval(proj_rel)

    proj_abs = relative_to_absolute(proj_rel, last_base)
    lower_abs = relative_to_absolute(conf_lower_rel, last_base)
    upper_abs = relative_to_absolute(conf_upper_rel, last_base)

    # Relatório
    _print_moe_report(fold_metrics, proj_abs, last_base,
                      config.horizon, config.confidence_level, gating_final)

    results = {
        'projected_ohlc':       proj_abs,
        'confidence_lower':     lower_abs,
        'confidence_upper':     upper_abs,
        'proj_relative':        proj_rel,
        'base_price':           last_base,
        'confidence_level':     config.confidence_level,
        'fold_metrics':         fold_metrics,
        'model':                last_moe,
        'feature_names':        feat_names,
        'gating_weights_final': gating_final,
        'gating_labels':        ['InceptionTime', 'LSTM Seq2Seq',
                                 'Transformer', 'ResidualMLP'],
        'individual_predictions': {
            'inception':   relative_to_absolute(
                ind_final['inception'].numpy()[0], last_base),
            'lstm':        relative_to_absolute(
                ind_final['lstm'].numpy()[0], last_base),
            'transformer': relative_to_absolute(
                ind_final['transformer'].numpy()[0], last_base),
            'mlp':         relative_to_absolute(
                ind_final['mlp'].numpy()[0], last_base),
        },
    }

    # ★ 6. CHECKPOINT — Salvar artefatos de forma segura
    save_checkpoint(results, config, fold_metrics)

    return results


# =============================================================================
# RELATÓRIO MoE (permanece aqui — não depende de visualização)
# =============================================================================


def _print_moe_report(
    fold_metrics: List[dict],
    proj_abs: np.ndarray,
    base_price: float,
    horizon: int,
    confidence_level: float,
    gating_final: np.ndarray,
) -> None:
    """Imprime relatório estruturado do MoE."""
    print(f"\n{'═' * 65}")
    print("  MoE GATING — RELATÓRIO FINAL")
    print(f"{'═' * 65}")

    if fold_metrics:
        avg_mae = np.mean([m['mae_moe'] for m in fold_metrics])
        print(f"  MAE médio (MoE, {len(fold_metrics)} folds): {avg_mae:.6f}")

        labels = ['InceptionTime', 'LSTM', 'Transformer', 'ResidualMLP']
        avg_gating = np.mean([m['gating_weights'] for m in fold_metrics], axis=0)
        print(f"\n  Pesos médios do Gating (todos os folds):")
        for label, w in zip(labels, avg_gating):
            bar = '█' * int(w * 40)
            print(f"    {label:15s} │ {w:.3f} │ {bar}")

    print(f"\n  Pesos do Gating na projeção final:")
    labels = ['InceptionTime', 'LSTM', 'Transformer', 'ResidualMLP']
    for label, w in zip(labels, gating_final):
        bar = '█' * int(w * 40)
        print(f"    {label:15s} │ {w:.3f} │ {bar}")

    direction = "ALTA ↑" if proj_abs[-1, 3] > base_price else "BAIXA ↓"
    delta_pct = (proj_abs[-1, 3] - base_price) / base_price * 100
    print(f"\n  Direção projetada: {direction} ({delta_pct:+.4f}%)")
    print(f"  Base price:        {base_price:.5f}")
    print(f"  Confiança:         {confidence_level:.0%} (Conformal)")
    print(f"{'═' * 65}")


# =============================================================================
# EXECUÇÃO DIRETA
# =============================================================================

if __name__ == "__main__":
    from moe_visualization import run_daily_comparison

    print("=" * 65)
    print("  MoE PIPELINE — TREINO + CHECKPOINT + DAILY COMPARISON")
    print("=" * 65)

    try:
        from moe_gating import MoEConfig

        config = MoEConfig(
            input_file='../data/final/dataset_final.h5',
            input_key='features',
            window_size=60,
            horizon=15,
            purge_bars=200,
            n_splits=5,
            epochs=60,
            batch_size=64,
            confidence_level=0.90,
            loss_alpha=0.5,
            loss_beta=0.2,
            loss_gamma=1.0,
            huber_delta=1.0,
            model_save_path='../models/EURUSD/moe_model.keras',
            results_save_path='../exports/inference/results_inference.parquet',
            config_save_path='../models/EURUSD/moe_config.json',
        )

        # ─── ETAPA 1: TREINAR E SALVAR ──────────────────────────────────
        results = run_moe_pipeline(config=config)

        if results is None:
            print("  ❌ Pipeline falhou. Ver logs acima.")
            exit(1)

        print(f"\n  ✅ Treino completo! Checkpoints salvos:")
        print(f"     · Modelo:     {config.model_save_path}")
        print(f"     · Resultados: {config.results_save_path}")
        print(f"     · Config:     {config.config_save_path}")

        # ─── ETAPA 2: DAILY COMPARISON (via moe_visualization) ──────────
        df_ticks = pd.read_hdf(config.input_file, key=config.input_key)
        if "Date" not in df_ticks.columns:
            if isinstance(df_ticks.index, pd.DatetimeIndex):
                df_ticks["Date"] = df_ticks.index
            else:
                print("  ⚠️ Sem timestamps — daily comparison não disponível.")
                exit(0)

        run_daily_comparison(df_ticks, results)

    except FileNotFoundError:
        print("  ❌ dataset não encontrado.")
        print("  Execute baixa_dados.py e calcula_alphas.py primeiro.")
    except Exception as e:
        print(f"  ❌ Erro: {e}")
        import traceback
        traceback.print_exc()

