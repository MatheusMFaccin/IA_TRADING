"""
===============================================================================
MÓDULO 3: ENGENHARIA DE ALPHA — MICROESTRUTURA + FRACDIFF + TRIPLE BARRIER
===============================================================================
Gera features de microestrutura de mercado com correlação cruzada < 0.40,
aplica Fractional Differentiation para manter memória sem perder
estacionaridade, e implementa Triple Barrier Method para labeling.

Categorias de Alpha:
    1. Direção       (Returns, MA Crossover Signal)
    2. Volatilidade  (Garman-Klass, Yang-Zhang, Parkinson)
    3. Liquidez      (Amihud Illiquidity, Kyle's Lambda)
    4. Informação    (CDF-VPIN, Tick Run Lengths)
    5. Regime        (Hurst Exponent)
    6. Microestrutura (Spread Bid-Ask, VWAP, Order Imbalance) [NOVO]
    7. Memória       (Fractional Differentiation de Close e Volume) [NOVO]

Labeling:
    Triple Barrier Method substitui o retorno fixo de 1 barra.
    Define take-profit, stop-loss e timeout como barreiras simétricas
    baseadas na volatilidade local (Garman-Klass).

Normalização: Z-Score Robusto via MAD (Median Absolute Deviation)
    z_robust = (x - median) / (1.4826 · MAD)

Referências:
    - Garman & Klass (1980): Volatility Estimation
    - Yang & Zhang (2000): Drift-Independent Volatility
    - Amihud (2002): Illiquidity and Stock Returns
    - Kyle (1985): Continuous Auctions and Insider Trading
    - Easley et al. (2012): Flow Toxicity (VPIN)
    - López de Prado (2018): AFML — Cap. 2, 3, 5, 19
    - Hosking (1981): Fractional Differencing
===============================================================================
"""

from __future__ import annotations

import os
import shutil
import pandas as pd
import numpy as np
from scipy.stats import norm
from typing import Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# 1. ALPHAS DE DIREÇÃO
# =============================================================================

def alpha_log_returns(df: pd.DataFrame) -> pd.Series:
    """
    Log Returns: r_t = ln(C_t / C_{t-1})

    Superior a pct_change() porque:
    - Aditivamente composto (r_{0→T} = Σ r_t)
    - Aproximadamente normal para intervalos curtos
    - Estacionário (diferenciação de primeira ordem do log-preço)
    """
    return np.log(df['Close'] / df['Close'].shift(1))


def alpha_ma_crossover(
    df: pd.DataFrame,
    fast: int = 9,
    slow: int = 21
) -> pd.Series:
    """
    MA Crossover Signal: sign(EMA_fast - EMA_slow)

    Retorna sinal discreto {-1, 0, +1}:
        +1 = EMA rápida ACIMA da lenta (momentum bullish)
        -1 = EMA rápida ABAIXO da lenta (momentum bearish)

    Usamos EMA para dar mais peso a observações recentes — crítico
    em tick bars onde a informação decai rapidamente.
    """
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()

    diff = ema_fast - ema_slow
    atr = (df['High'] - df['Low']).rolling(slow).mean()
    atr = atr.replace(0, np.nan)

    normalized_diff = diff / atr
    return np.sign(normalized_diff)


# =============================================================================
# 2. ALPHAS DE VOLATILIDADE
# =============================================================================

def alpha_garman_klass(df: pd.DataFrame) -> pd.Series:
    """
    Garman-Klass Volatility Estimator:
        σ²_GK = 0.5 · ln(H/L)² - (2ln2 - 1) · ln(C/O)²

    5x mais eficiente que close-to-close por usar informação OHLC completa.
    """
    log_hl = np.log(df['High'] / df['Low'])
    log_co = np.log(df['Close'] / df['Open'])
    return 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2


def alpha_yang_zhang(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Yang-Zhang Volatility Estimator (rolling):
        σ²_YZ = σ²_overnight + k · σ²_close-to-close + (1-k) · σ²_Rogers-Satchell

    O estimador mais eficiente para dados OHLC com drift.
    """
    log_oc = np.log(df['Open'] / df['Close'].shift(1))
    log_co = np.log(df['Close'] / df['Open'])
    log_ho = np.log(df['High'] / df['Open'])
    log_lo = np.log(df['Low'] / df['Open'])

    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    n = window
    k = 0.34 / (1.34 + (n + 1) / (n - 1))

    overnight_var = log_oc.rolling(window).var()
    close_var = log_co.rolling(window).var()
    rs_var = rs.rolling(window).mean()

    return overnight_var + k * close_var + (1 - k) * rs_var


def alpha_parkinson(df: pd.DataFrame) -> pd.Series:
    """
    Parkinson Volatility: σ²_P = ln(H/L)² / (4 · ln2)
    Usa apenas High/Low, enviesado para cima em mercados com mean-reversion.
    """
    log_hl = np.log(df['High'] / df['Low'])
    return log_hl**2 / (4 * np.log(2))


# =============================================================================
# 3. ALPHAS DE LIQUIDEZ
# =============================================================================

def alpha_amihud_illiquidity(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Amihud Illiquidity Ratio: ILLIQ_t = |r_t| / V_t
    ILLIQ alto = mercado ilíquido; ILLIQ baixo = mercado líquido.

    Obs: Em Forex tick bars (MT5), Volume pode ser 0 ou inexistente.
    Nesse caso, faz fallback para Tick_Count ou retorna |returns| puro.
    """
    returns = np.log(df['Close'] / df['Close'].shift(1)).abs()

    # Tentar Volume real, depois Tick_Count, depois fallback
    volume = None

    if 'Volume' in df.columns:
        vol_candidate = df['Volume'].replace(0, np.nan)
        if vol_candidate.notna().sum() > 0:
            volume = vol_candidate

    if volume is None and 'Tick_Count' in df.columns:
        vol_candidate = df['Tick_Count'].replace(0, np.nan)
        if vol_candidate.notna().sum() > 0:
            volume = vol_candidate

    if volume is None and 'tick_volume' in df.columns:
        vol_candidate = df['tick_volume'].replace(0, np.nan)
        if vol_candidate.notna().sum() > 0:
            volume = vol_candidate

    if volume is None and 'real_volume' in df.columns:
        vol_candidate = df['real_volume'].replace(0, np.nan)
        if vol_candidate.notna().sum() > 0:
            volume = vol_candidate

    if volume is None:
        # Sem dados de volume válidos — retorna |returns| rolling como proxy
        return returns.rolling(window).mean()

    illiq = returns / volume
    return illiq.rolling(window).mean()


def alpha_kyles_lambda(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Kyle's Lambda: λ = Cov(ΔP, signed_V) / Var(signed_V)
    Mede fragilidade da liquidez — quanto o market maker ajusta o preço
    em resposta ao order flow.
    """
    delta_price = df['Close'].diff()
    returns = np.log(df['Close'] / df['Close'].shift(1))

    if 'Volume' in df.columns:
        volume = df['Volume'].replace(0, 1)
    elif 'Tick_Count' in df.columns:
        volume = df['Tick_Count'].replace(0, 1)
    else:
        volume = pd.Series(1, index=df.index)

    signed_volume = np.sign(returns) * volume

    cov_rolling = delta_price.rolling(window).cov(signed_volume)
    var_rolling = signed_volume.rolling(window).var()
    var_rolling = var_rolling.replace(0, np.nan)

    kyles_lambda = cov_rolling / var_rolling
    return kyles_lambda.abs()


# =============================================================================
# 4. ALPHAS DE INFORMAÇÃO ASSIMÉTRICA
# =============================================================================

def alpha_cdf_vpin(df: pd.DataFrame, window: int = 50) -> pd.Series:
    """
    CDF-VPIN: Probabilidade de informed trading normalizada em [0, 1].

    1. Classificar ticks via Lee-Ready adaptado
    2. VPIN_raw = |V_buy - V_sell| / V_total
    3. CDF-VPIN = Φ((VPIN - μ) / σ)

    CDF-VPIN ≈ 1 → Alta toxicidade
    CDF-VPIN ≈ 0 → Fluxo balanceado
    """
    returns = np.log(df['Close'] / df['Close'].shift(1))

    if 'Volume' in df.columns:
        volume = df['Volume'].replace(0, 1)
    elif 'Tick_Count' in df.columns:
        volume = df['Tick_Count'].replace(0, 1)
    else:
        volume = pd.Series(1, index=df.index)

    sigma_r = returns.rolling(window).std().replace(0, np.nan)
    z_score = returns / sigma_r
    prob_buy = pd.Series(norm.cdf(z_score), index=df.index)

    v_buy = prob_buy * volume
    v_sell = (1 - prob_buy) * volume

    vpin_raw = (v_buy - v_sell).abs() / volume
    vpin_rolling = vpin_raw.rolling(window).mean()

    vpin_mean = vpin_rolling.rolling(window * 4).mean()
    vpin_std = vpin_rolling.rolling(window * 4).std().replace(0, np.nan)

    return pd.Series(
        norm.cdf((vpin_rolling - vpin_mean) / vpin_std),
        index=df.index
    )


def alpha_tick_run_length(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Tick Run Length: comprimento médio de sequências de ticks na mesma direção.
    Run Length alto → provável informed trading.
    """
    direction = np.sign(df['Close'].diff())
    direction_change = (direction != direction.shift(1)).astype(int)
    run_groups = direction_change.cumsum()
    run_lengths = direction.groupby(run_groups).transform('count')
    return run_lengths.rolling(window).mean()


# =============================================================================
# 5. ALPHAS DE REGIME
# =============================================================================

def alpha_hurst_exponent(series: pd.Series, window: int = 100) -> pd.Series:
    """
    Expoente de Hurst (Análise R/S rolling):
        H > 0.5 → trending (momentum)
        H = 0.5 → random walk
        H < 0.5 → mean-reverting
    """
    def _hurst(ts: np.ndarray) -> float:
        if len(ts) < 20:
            return np.nan

        lags = range(2, min(20, len(ts) // 2))
        tau: list[float] = []
        valid_lags: list[int] = []

        for lag in lags:
            diff = ts[lag:] - ts[:-lag]
            std_val = np.std(diff)
            if std_val > 0:
                tau.append(np.sqrt(std_val))
                valid_lags.append(lag)

        if len(valid_lags) < 3:
            return np.nan

        poly = np.polyfit(np.log(valid_lags), np.log(tau), 1)
        return poly[0] * 2.0

    return series.rolling(window).apply(_hurst, raw=True)


# =============================================================================
# 6. ALPHAS DE MICROESTRUTURA [NOVO]
# =============================================================================

def alpha_spread_bid_ask(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Spread Bid-Ask rolling normalizado pelo midprice.

        Spread_norm_t = Spread_Mean_t / Midprice_Close_t

    Quantifica o custo de transação implícito.
    Spread alto → menor liquidez, maior custo de execução.
    Spread baixo → mercado líquido, tight quotes.

    Se Spread_Mean não disponível, calcula via (Ask_Close - Close) proxy.
    """
    if 'Spread_Mean' in df.columns and 'Midprice_Close' in df.columns:
        raw_spread = df['Spread_Mean'] / df['Midprice_Close']
    elif 'Ask_Close' in df.columns:
        raw_spread = (df['Ask_Close'] - df['Close']) / df['Close']
    else:
        # Fallback: usar High-Low como proxy de spread
        raw_spread = (df['High'] - df['Low']) / df['Close']

    return raw_spread.rolling(window).mean()


def alpha_vwap(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Volume Weighted Average Price — desvio relativo do Close.

        VWAP_w = Σ(TypicalPrice_i × Vol_i) / Σ(Vol_i)
        Alpha = (Close - VWAP) / VWAP

    TypicalPrice = (High + Low + Close) / 3

    Desvio positivo → preço ACIMA do VWAP (pressão de compra)
    Desvio negativo → preço ABAIXO do VWAP (pressão de venda)
    """
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3.0

    if 'Volume' in df.columns:
        volume = df['Volume'].replace(0, 1)
    elif 'Tick_Count' in df.columns:
        volume = df['Tick_Count'].replace(0, 1)
    else:
        volume = pd.Series(1, index=df.index)

    pv = typical_price * volume
    vwap = pv.rolling(window).sum() / volume.rolling(window).sum()

    # Desvio relativo
    vwap = vwap.replace(0, np.nan)
    return (df['Close'] - vwap) / vwap


def alpha_order_imbalance(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Order Imbalance via classificação Buy/Sell (Lee-Ready simplificado).

        OI = (V_buy - V_sell) / (V_buy + V_sell)

    Positivo → pressão de compra dominante
    Negativo → pressão de venda dominante

    Classificação: sign(r_t) determina se o tick é buy (+) ou sell (-).
    """
    returns = np.log(df['Close'] / df['Close'].shift(1))

    if 'Volume' in df.columns:
        volume = df['Volume'].replace(0, 1)
    elif 'Tick_Count' in df.columns:
        volume = df['Tick_Count'].replace(0, 1)
    else:
        volume = pd.Series(1, index=df.index)

    # Classificar volume por direção
    v_buy = volume.where(returns > 0, 0)
    v_sell = volume.where(returns < 0, 0)

    v_buy_sum = v_buy.rolling(window).sum()
    v_sell_sum = v_sell.rolling(window).sum()

    total = v_buy_sum + v_sell_sum
    total = total.replace(0, np.nan)

    return (v_buy_sum - v_sell_sum) / total


# =============================================================================
# 7. FRACTIONAL DIFFERENTIATION [NOVO]
# =============================================================================

def _get_fracdiff_weights(d: float, threshold: float = 1e-5) -> np.ndarray:
    """
    Calcula os pesos da diferenciação fracionária.

    w_k = -w_{k-1} × (d - k + 1) / k

    Os pesos decaem geometricamente. Truncamos quando |w_k| < threshold
    para eficiência computacional.

    Parâmetros:
    -----------
    d : float
        Ordem de diferenciação fracionária. d ∈ (0, 1).
        d → 0: Muita memória, pouca estacionaridade
        d → 1: Estacionário, pouca memória
        Ótimo: d ∈ [0.3, 0.5] para séries financeiras
    threshold : float
        Peso mínimo para truncamento da série.

    Retorna:
    --------
    np.ndarray com os pesos w_0, w_1, ..., w_K
    """
    weights: list[float] = [1.0]
    k: int = 1
    while True:
        w_k = -weights[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        weights.append(w_k)
        k += 1
        if k > 10000:
            break
    return np.array(weights)


def fractional_differentiation(
    series: pd.Series,
    d: float,
    threshold: float = 1e-5
) -> pd.Series:
    """
    Aplica Fractional Differentiation (FracDiff) sobre uma série temporal.

    FracDiff mantém a memória da série (correlação de longo prazo) enquanto
    torna a série estacionária — essencial para evitar overfitting em modelos
    de ML aplicados a séries financeiras.

    Fórmula (expanding window):
        X̃_t = Σ_{k=0}^{K} w_k · X_{t-k}

    onde w_k são os pesos fracionários.

    Parâmetros:
    -----------
    series : pd.Series
        Série temporal (tipicamente log-preço ou volume).
    d : float
        Ordem fracionária. d=0 → original, d=1 → diff completa.
    threshold : float
        Truncamento dos pesos.

    Retorna:
    --------
    pd.Series estacionária com memória preservada.

    Referência:
    -----------
    Hosking (1981), López de Prado (2018) AFML Cap. 5
    """
    weights = _get_fracdiff_weights(d, threshold)
    K = len(weights)

    values = series.values
    n = len(values)
    result = np.full(n, np.nan)

    for t in range(K - 1, n):
        window = values[t - K + 1: t + 1][::-1]  # Mais recente primeiro
        result[t] = np.dot(weights[:len(window)], window)

    return pd.Series(result, index=series.index, name=f'{series.name}_fracdiff')


def find_optimal_d(
    series: pd.Series,
    max_d: float = 1.0,
    p_value_threshold: float = 0.05,
    step: float = 0.05
) -> float:
    """
    Encontra o menor d que torna a série estacionária via ADF test.

    Busca iterativa: d = 0.05, 0.10, ..., max_d
    Para em d quando ADF p-value < p_value_threshold.

    Garante d ∈ [0.3, 0.7] para manter memória suficiente.

    Retorna:
    --------
    float: d ótimo
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        print("  ⚠️ statsmodels não instalado. Usando d=0.4 como fallback.")
        return 0.4

    d_values = np.arange(step, max_d + step, step)
    optimal_d: float = 0.4  # Fallback

    for d in d_values:
        fd_series = fractional_differentiation(series, d)
        fd_clean = fd_series.dropna()

        if len(fd_clean) < 100:
            continue

        try:
            adf_result = adfuller(fd_clean, maxlag=20, autolag='AIC')
            adf_pvalue = adf_result[1]

            if adf_pvalue < p_value_threshold:
                optimal_d = max(d, 0.3)  # Mínimo 0.3 para memória
                optimal_d = min(optimal_d, 0.7)  # Máximo 0.7
                break
        except Exception:
            continue

    return optimal_d


# =============================================================================
# 8. TRIPLE BARRIER METHOD [NOVO]
# =============================================================================

def triple_barrier_labels(
    df: pd.DataFrame,
    vol_window: int = 20,
    k_up: float = 2.0,
    k_down: float = 2.0,
    max_holding: int = 10,
    vol_estimator: str = 'garman_klass'
) -> Tuple[pd.Series, pd.Series]:
    """
    Triple Barrier Method para labeling de treino.

    Define 3 barreiras ao redor de cada observação t:

        1. Upper Barrier (Take-Profit):
           Close_t × (1 + σ_t × k_up)

        2. Lower Barrier (Stop-Loss):
           Close_t × (1 - σ_t × k_down)

        3. Vertical Barrier (Timeout):
           t + max_holding barras

    σ_t é a volatilidade local estimada via vol_estimator.

    Labels:
        +1 = Upper barrier tocada primeiro (movimento de alta)
        -1 = Lower barrier tocada primeiro (movimento de baixa)
         0 = Vertical barrier atingida (timeout / indecisão)

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame com OHLC.
    vol_window : int
        Janela para estimativa de volatilidade.
    k_up, k_down : float
        Multiplicadores de volatilidade para as barreiras.
        k = 2.0 → barreira a 2 desvios padrão.
    max_holding : int
        Número máximo de barras antes do timeout (vertical barrier).
    vol_estimator : str
        'garman_klass' ou 'atr'. Métrica de volatilidade.

    Retorna:
    --------
    (labels, returns_at_touch) : Tuple[pd.Series, pd.Series]
        labels: {-1, 0, +1} conforme qual barreira foi tocada primeiro.
        returns_at_touch: retorno contínuo no ponto de toque.

    Referência:
    -----------
    López de Prado (2018) AFML — Cap. 3: Meta-Labeling
    """
    n = len(df)
    labels = np.full(n, np.nan)
    returns_at_touch = np.full(n, np.nan)

    # Estimar volatilidade local
    if vol_estimator == 'garman_klass':
        vol = alpha_garman_klass(df).rolling(vol_window).mean()
        vol = np.sqrt(np.abs(vol))  # Desvio padrão
    else:
        # ATR como alternativa
        tr = pd.concat([
            df['High'] - df['Low'],
            (df['High'] - df['Close'].shift(1)).abs(),
            (df['Low'] - df['Close'].shift(1)).abs()
        ], axis=1).max(axis=1)
        vol = tr.rolling(vol_window).mean() / df['Close']

    vol = vol.bfill().fillna(0.001)

    closes = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values
    vol_values = vol.values

    for t in range(n):
        if np.isnan(vol_values[t]) or vol_values[t] <= 0:
            continue

        entry_price = closes[t]
        sigma = vol_values[t]

        upper = entry_price * (1.0 + sigma * k_up)
        lower = entry_price * (1.0 - sigma * k_down)

        # Percorrer barras futuras
        label: int = 0
        ret: float = 0.0

        end = min(t + max_holding, n - 1)
        for h in range(t + 1, end + 1):
            # Verificar se o High tocou upper barrier
            if highs[h] >= upper:
                label = 1
                ret = (upper - entry_price) / entry_price
                break
            # Verificar se o Low tocou lower barrier
            if lows[h] <= lower:
                label = -1
                ret = (lower - entry_price) / entry_price
                break

            # Se chegou ao timeout
            if h == end:
                label = 0
                ret = (closes[h] - entry_price) / entry_price

        labels[t] = label
        returns_at_touch[t] = ret

    return (
        pd.Series(labels, index=df.index, name='TB_Label'),
        pd.Series(returns_at_touch, index=df.index, name='TB_Return')
    )


# =============================================================================
# NORMALIZAÇÃO ROBUSTA
# =============================================================================

def robust_zscore(series: pd.Series, window: int = 252) -> pd.Series:
    """
    Z-Score Robusto via MAD:
        z_robust = (x - median) / (1.4826 · MAD)

    Resistente a outliers (breakdown point = 50%).
    """
    rolling_median = series.rolling(window, min_periods=1).median()
    rolling_mad = (series - rolling_median).abs().rolling(window, min_periods=1).median()

    denominator = 1.4826 * rolling_mad + 1e-10
    return (series - rolling_median) / denominator


def minmax_adaptive(series: pd.Series, window: int = 252) -> pd.Series:
    """Min-Max Adaptativo (rolling window), mapeia para [0, 1]."""
    rolling_min = series.rolling(window, min_periods=1).min()
    rolling_max = series.rolling(window, min_periods=1).max()

    range_val = (rolling_max - rolling_min).replace(0, np.nan)
    return (series - rolling_min) / range_val


# =============================================================================
# PIPELINE PRINCIPAL DE GERAÇÃO DE ALPHAS
# =============================================================================

def generate_alpha_features(
    input_file: str = '../data/processed/dataset_clean.h5',
    output_file: str = '../data/final/dataset_final.h5',
    normalize: str = 'robust_zscore',
    window_norm: int = 252,
    fracdiff_d: Optional[float] = None,
    use_triple_barrier: bool = True,
    tb_k_up: float = 2.0,
    tb_k_down: float = 2.0,
    tb_max_holding: int = 10
) -> Tuple[Optional[pd.DataFrame], Optional[list]]:
    """
    Pipeline completo de geração de alphas com FracDiff e Triple Barrier.

    Parâmetros:
    -----------
    input_file : str
        Arquivo HDF5 de entrada (output do limpaArquivos.py).
    output_file : str
        Arquivo HDF5 de saída com features prontas.
    normalize : str
        'robust_zscore', 'minmax', ou 'none'.
    window_norm : int
        Janela para normalização rolling.
    fracdiff_d : float, optional
        Ordem de diferenciação fracionária. Se None, auto-detecta.
    use_triple_barrier : bool
        Se True, usa Triple Barrier Method para labeling.
    tb_k_up, tb_k_down : float
        Multiplicadores de volatilidade para as barreiras.
    tb_max_holding : int
        Timeout em barras para o Triple Barrier.

    Retorna:
    --------
    (pd.DataFrame, list) : DataFrame com features e lista de nomes.
    """
    print("═" * 65)
    print("  ENGENHARIA DE ALPHA — MICROESTRUTURA + FRACDIFF + TRIPLE BARRIER")
    print("═" * 65)

    # 1. Carregar dados denoised (com fallback de keys)
    df = None
    for key_try in ['data', 'tick_bars', 'features']:
        try:
            df = pd.read_hdf(input_file, key=key_try)
            print(f"  📂 Dados carregados: {len(df):,} barras (key='{key_try}')")
            print(f"     Colunas: {len(df.columns)} | {list(df.columns[:8])}...")
            break
        except (KeyError, Exception):
            continue

    if df is None:
        # Fallback: tentar ler qualquer key
        try:
            with pd.HDFStore(input_file, mode='r') as store:
                available_keys = store.keys()
                print(f"  Keys disponíveis em '{input_file}': {available_keys}")
                if available_keys:
                    df = store[available_keys[0]]
                    print(f"  📂 Dados carregados via fallback: {len(df):,} barras")
        except Exception as e:
            pass

    if df is None:
        print(f"  ❌ Erro ao carregar '{input_file}': nenhuma key válida encontrada")
        print("     Execute limpaArquivos.py primeiro.")
        return None, None

    # Verificação antecipada: dataset mínimo para as janelas rolantes
    min_required = 400  # Hurst(100) + Normalização(252) + margem
    if len(df) < min_required:
        print(f"  ⚠️ Dataset pequeno: {len(df):,} barras (recomendado: ≥{min_required})")
        print(f"     Ajustando janelas para acomodar dataset reduzido...")

    # 2. Gerar Alphas Clássicos
    print("\n  Calculando Alphas...")

    # Calcular janelas adaptativas baseadas no tamanho do dataset
    n_bars = len(df)
    w20 = min(20, max(5, n_bars // 20))     # Janela curta (default 20)
    w50 = min(50, max(10, n_bars // 10))     # Janela média (default 50)
    w100 = min(100, max(20, n_bars // 5))    # Janela longa (default 100)

    if n_bars < 400:
        print(f"  ⚠️ Janelas adaptadas: w20={w20}, w50={w50}, w100={w100}")

    # --- Direção ---
    print("    [ 1/13] Log Returns")
    df['Alpha_Returns'] = alpha_log_returns(df)

    print("    [ 2/13] MA Crossover Signal")
    df['Alpha_MA_Cross'] = alpha_ma_crossover(df, fast=9, slow=21)

    # --- Volatilidade ---
    print("    [ 3/13] Garman-Klass Volatility")
    df['Alpha_GK_Vol'] = alpha_garman_klass(df)

    print("    [ 4/13] Yang-Zhang Volatility")
    df['Alpha_YZ_Vol'] = alpha_yang_zhang(df, window=w20)

    print("    [ 5/13] Parkinson Volatility")
    df['Alpha_Parkinson'] = alpha_parkinson(df)

    # --- Liquidez ---
    print("    [ 6/13] Amihud Illiquidity")
    df['Alpha_Amihud'] = alpha_amihud_illiquidity(df, window=w20)

    print("    [ 7/13] Kyle's Lambda")
    df['Alpha_Kyle_Lambda'] = alpha_kyles_lambda(df, window=w20)

    # --- Informação ---
    print("    [ 8/13] CDF-VPIN")
    df['Alpha_CDF_VPIN'] = alpha_cdf_vpin(df, window=w50)

    print("    [ 9/13] Tick Run Length")
    df['Alpha_Tick_Run'] = alpha_tick_run_length(df, window=w20)

    # --- Regime ---
    print("    [10/13] Hurst Exponent")
    df['Alpha_Hurst'] = alpha_hurst_exponent(df['Close'], window=w100)

    # --- Microestrutura [NOVO] ---
    print("    [11/13] Spread Bid-Ask")
    df['Alpha_Spread'] = alpha_spread_bid_ask(df, window=w20)

    print("    [12/13] VWAP Deviation")
    df['Alpha_VWAP'] = alpha_vwap(df, window=w20)

    print("    [13/13] Order Imbalance")
    df['Alpha_OI'] = alpha_order_imbalance(df, window=w20)

    # 3. Lista de features alpha
    alpha_cols = [
        'Alpha_Returns', 'Alpha_MA_Cross',
        'Alpha_GK_Vol', 'Alpha_YZ_Vol', 'Alpha_Parkinson',
        'Alpha_Amihud', 'Alpha_Kyle_Lambda',
        'Alpha_CDF_VPIN', 'Alpha_Tick_Run',
        'Alpha_Hurst',
        'Alpha_Spread', 'Alpha_VWAP', 'Alpha_OI',
    ]

    # 4. Fractional Differentiation [NOVO]
    print("\n  Aplicando Fractional Differentiation...")

    if fracdiff_d is None:
        print("    Auto-detectando d ótimo via ADF test...")
        fracdiff_d = find_optimal_d(df['Close'], max_d=0.7, step=0.05)
    print(f"    d = {fracdiff_d:.2f}")

    df['Alpha_Close_FracDiff'] = fractional_differentiation(
        df['Close'], d=fracdiff_d
    )
    alpha_cols.append('Alpha_Close_FracDiff')

    if 'Volume' in df.columns:
        df['Alpha_Volume_FracDiff'] = fractional_differentiation(
            df['Volume'], d=min(fracdiff_d + 0.1, 0.8)
        )
        alpha_cols.append('Alpha_Volume_FracDiff')

    # 5. Labeling
    if use_triple_barrier:
        print("\n  Aplicando Triple Barrier Method...")
        tb_labels, tb_returns = triple_barrier_labels(
            df, vol_window=20,
            k_up=tb_k_up, k_down=tb_k_down,
            max_holding=tb_max_holding
        )
        df['Target_TB_Label'] = tb_labels
        df['Target_TB_Return'] = tb_returns

        # Estatísticas do labeling
        valid_labels = df['Target_TB_Label'].dropna()
        n_up = (valid_labels == 1).sum()
        n_down = (valid_labels == -1).sum()
        n_timeout = (valid_labels == 0).sum()
        total = len(valid_labels)
        print(f"    Labels: +1={n_up} ({n_up/total*100:.1f}%) | "
              f"-1={n_down} ({n_down/total*100:.1f}%) | "
              f"0={n_timeout} ({n_timeout/total*100:.1f}%)")

        if n_timeout / total > 0.05:
            print("    ⚠️ Timeout > 5% — considere ajustar k_up/k_down ou max_holding")

    # Legacy target (para compatibilidade)
    df['Target_Direction'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # 6. Remover NaN — apenas nas colunas de alpha (não em todas)
    # Diagnóstico: quantas linhas cada alpha invalida
    n_before = len(df)
    print(f"\n  Diagnóstico de NaN (antes de limpar): {n_before} barras")

    all_target_cols = ['Target_Direction']
    if use_triple_barrier:
        all_target_cols += ['Target_TB_Label', 'Target_TB_Return']

    nan_diagnostic_cols = alpha_cols + all_target_cols
    for col in nan_diagnostic_cols:
        if col in df.columns:
            n_nan = df[col].isna().sum()
            if n_nan > 0:
                print(f"    {col}: {n_nan} NaN ({n_nan / n_before * 100:.1f}%)")

    # Drop apenas nas colunas de alpha + targets (preserva colunas OHLC/features extras)
    drop_cols = [c for c in nan_diagnostic_cols if c in df.columns]
    df.dropna(subset=drop_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)

    n_after_alphas = len(df)
    print(f"  Barras após limpeza de alphas: {n_after_alphas} (removidas: {n_before - n_after_alphas})")

    if n_after_alphas == 0:
        print("  ❌ Todas as barras foram eliminadas pelos NaN dos alphas!")
        print("  Causa provável: dataset muito pequeno para as janelas rolantes.")
        print(f"  Janela máxima usada: Hurst(100), Normalização({window_norm})")
        print(f"  Mínimo necessário: ~{max(252, window_norm) + 100} barras")
        raise ValueError(
            f"DataFrame vazio após processamento — "
            f"dataset de {n_before} barras é insuficiente para "
            f"as janelas rolantes (mínimo: ~{max(252, window_norm) + 100})."
        )

    # 7. Normalização
    # Ajustar window_norm se o dataset remanescente for menor
    effective_window_norm = min(window_norm, max(20, n_after_alphas // 3))
    if effective_window_norm != window_norm:
        print(f"  ⚠️ window_norm ajustado: {window_norm} → {effective_window_norm} "
              f"(dataset de {n_after_alphas} barras)")

    print(f"\n  Normalizando alphas ({normalize}, window={effective_window_norm})...")

    discrete_cols = {'Alpha_MA_Cross'}
    if normalize == 'robust_zscore':
        for col in alpha_cols:
            if col not in discrete_cols:
                df[f'{col}_norm'] = robust_zscore(df[col], window=effective_window_norm)
        norm_cols = [
            f'{c}_norm' if c not in discrete_cols else c for c in alpha_cols
        ]
    elif normalize == 'minmax':
        for col in alpha_cols:
            if col not in discrete_cols:
                df[f'{col}_norm'] = minmax_adaptive(df[col], window=effective_window_norm)
        norm_cols = [
            f'{c}_norm' if c not in discrete_cols else c for c in alpha_cols
        ]
    else:
        norm_cols = alpha_cols

    # Limpar NaN pós-normalização (apenas nas colunas normalizadas)
    df.dropna(subset=norm_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  Barras após normalização: {len(df)} (removidas pela norm: {n_after_alphas - len(df)})")

    # 8. Verificação de correlação
    print("\n  📊 Matriz de Correlação dos Alphas:")
    corr_matrix = df[alpha_cols].corr()
    _print_correlation_report(corr_matrix, alpha_cols)

    # 9. Salvar — Protocolo de Escrita Robusta
    # a) Validar que o DataFrame não ficou vazio após processamento
    if len(df) == 0:
        raise ValueError(
            "DataFrame vazio após processamento — "
            "nenhum alpha válido gerado. Verifique os dados de entrada."
        )

    # b) Remover colunas 100% NaN (podem corromper o HDFStore)
    nan_cols = [c for c in df.columns if df[c].isna().all()]
    if nan_cols:
        print(f"  ⚠️ Removendo {len(nan_cols)} colunas 100% NaN: {nan_cols}")
        df.drop(columns=nan_cols, inplace=True)

    # c) Garantir que o diretório de destino existe
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # d) Escrita via context manager para garantir flush/close
    tmp_file = output_file + '.tmp'
    try:
        with pd.HDFStore(tmp_file, mode='w', complevel=6, complib='blosc:zstd') as store:
            store.put('features', df, format='table')
    except Exception as e:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
        raise IOError(f"Falha ao escrever HDF5: {e}") from e

    # d) Validar integridade pós-escrita (tamanho mínimo)
    file_size = os.path.getsize(tmp_file)
    if file_size < 1024:
        os.remove(tmp_file)
        raise IOError(
            f"Arquivo corrompido ({file_size} bytes). "
            f"Esperado > 1KB para {len(df):,} barras."
        )

    # e) Atomic rename — sobrescreve apenas se válido
    shutil.move(tmp_file, output_file)

    # f) Round-trip test — verificar que a leitura retorna o mesmo shape
    df_verify = pd.read_hdf(output_file, key='features')
    if len(df_verify) != len(df):
        raise IOError(
            f"Round-trip falhou: escreveu {len(df):,}, "
            f"leu {len(df_verify):,} barras."
        )

    file_size_mb = file_size / (1024 * 1024)

    print(f"\n{'═' * 65}")
    print(f"  ✅ ALPHAS GERADOS")
    print(f"  Features: {len(norm_cols)} ({len(alpha_cols)} raw + normalizados)")
    print(f"  FracDiff d: {fracdiff_d:.2f}")
    print(f"  Labeling: {'Triple Barrier' if use_triple_barrier else 'Fixed Return'}")
    print(f"  Barras finais: {len(df):,}")
    print(f"  Arquivo: {output_file} ({file_size_mb:.1f} MB)")
    print(f"  Integridade: ✅ Round-trip OK ({len(df_verify):,} barras lidas)")
    print(f"{'═' * 65}")

    return df, norm_cols


def _print_correlation_report(corr: pd.DataFrame, cols: list) -> None:
    """Imprime diagnóstico de correlação entre alphas."""
    high_corr_pairs: list[tuple] = []

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c = abs(corr.iloc[i, j])
            if c > 0.40:
                high_corr_pairs.append((cols[i], cols[j], c))

    if len(high_corr_pairs) == 0:
        print("    ✅ Todas as correlações < 0.40 — Diversidade confirmada!")
    else:
        print(f"    ⚠️ {len(high_corr_pairs)} pares com |ρ| > 0.40:")
        for a, b, c in sorted(high_corr_pairs, key=lambda x: -x[2]):
            short_a = a.replace('Alpha_', '')
            short_b = b.replace('Alpha_', '')
            print(f"       {short_a} ↔ {short_b}: ρ = {c:.3f}")


# =============================================================================
# EXECUÇÃO
# =============================================================================
if __name__ == "__main__":
    from pathlib import Path
    Path('../data/final').mkdir(parents=True, exist_ok=True)
    df_features, feature_names = generate_alpha_features(
        input_file='../data/processed/dataset_clean.h5',
        output_file='../data/final/dataset_final.h5',
        normalize='robust_zscore',
        window_norm=252,
        fracdiff_d=None,       # Auto-detect via ADF
        use_triple_barrier=True,
        tb_k_up=2.0,
        tb_k_down=2.0,
        tb_max_holding=10
    )

    if df_features is not None:
        print(f"\nShape da matriz de features: {df_features[feature_names].shape}")
        print(f"\nÚltimas 5 linhas:")
        target_cols = ['Target_TB_Label', 'Target_TB_Return']
        available_targets = [c for c in target_cols if c in df_features.columns]
        print(df_features[feature_names + available_targets].tail())