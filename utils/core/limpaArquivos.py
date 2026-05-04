"""
===============================================================================
MÓDULO 2: DENOISING MULTI-CAMADA
===============================================================================
Pipeline de Separação Sinal/Ruído em 3 Camadas:

    Dados Brutos → [Wavelet DWT] → [Kalman Adaptativo] → [Marchenko-Pastur] → Sinal

Camada 1 — Wavelet Transform (Donoho-Johnstone):
    Decompõe o sinal em componentes de frequência via DWT.
    Aplica soft thresholding com limiar universal λ = σ√(2·ln(n)).
    Remove ruído de microestrutura (d1-d2) preservando tendência (a4).

Camada 2 — Kalman Adaptativo:
    Filtro de estado com observation_covariance dinâmica.
    Adapta-se automaticamente a regimes de alta/baixa volatilidade.
    Q_t = base_Q · (σ_local / σ_global)²

Camada 3 — Marchenko-Pastur (Denoising de Matriz de Correlação):
    Remove eigenvalues indistinguíveis de ruído aleatório.
    λ± = σ²(1 ± √(N/T))² — limites do bulk teórico.
    Eigenvalues acima de λ+ carregam sinal informacional real.

Referências:
    - Donoho & Johnstone (1994): "Ideal Spatial Adaptation by Wavelet Shrinkage"
    - Kalman (1960): "A New Approach to Linear Filtering and Prediction Problems"
    - Marchenko & Pastur (1967): "Distribution of Eigenvalues for Some Sets
      of Random Matrices"
    - López de Prado (2020): "Machine Learning for Asset Managers" — Cap. 2
===============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
from pykalman import KalmanFilter
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# CAMADA 1: WAVELET DENOISING (Donoho-Johnstone)
# =============================================================================

def wavelet_denoise(
    signal: np.ndarray,
    wavelet: str = 'db8',
    level: int = 4,
    mode: str = 'soft'
) -> np.ndarray:
    """
    Decomposição Wavelet Discreta (DWT) com thresholding universal.
    
    Parâmetros:
    -----------
    signal : np.ndarray
        Série temporal a ser denoised.
    wavelet : str
        Família wavelet. 'db8' (Daubechies-8) é ótima para séries financeiras
        por ter suporte compacto largo (captura oscilações suaves).
    level : int
        Nível de decomposição. Level 4 separa:
            d1: Ruído de microestrutura (altíssima freq.)
            d2: Ruído de tick-to-tick (alta freq.)
            d3: Componentes intradiários (média freq.)
            d4: Padrões de sessão (baixa freq.)
            a4: Tendência de fundo (muito baixa freq.)
    mode : str
        'soft' para soft thresholding (shrinkage — preserva continuidade).
        'hard' para hard thresholding (zera coeficientes abaixo do limiar).
    
    Retorna:
    --------
    np.ndarray : Sinal denoised.
    
    Lógica Matemática:
    ------------------
    O limiar universal de Donoho-Johnstone é:
    
        λ = σ̂ · √(2 · ln(n))
    
    onde σ̂ é estimado via MAD (Median Absolute Deviation) dos coeficientes
    de detalhe do nível mais fino:
    
        σ̂ = median(|d1|) / 0.6745
    
    O fator 0.6745 é o quantil 75% da distribuição normal padrão.
    Soft thresholding: ŵ = sign(w) · max(|w| - λ, 0)
    """
    # Decomposição
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # Estimativa robusta do ruído via MAD nos coeficientes de maior frequência
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    
    # Limiar universal de Donoho-Johnstone
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    
    # Aplicar thresholding apenas nos coeficientes de detalhe (não na aproximação)
    denoised_coeffs = [coeffs[0]]  # Aproximação (tendência) intacta
    for detail_coeff in coeffs[1:]:
        denoised_coeffs.append(
            pywt.threshold(detail_coeff, value=threshold, mode=mode)
        )
    
    # Reconstrução
    reconstructed = pywt.waverec(denoised_coeffs, wavelet)
    
    # Ajustar tamanho (waverec pode adicionar 1 sample por padding)
    return reconstructed[:len(signal)]


# =============================================================================
# CAMADA 2: FILTRO DE KALMAN ADAPTATIVO
# =============================================================================

def adaptive_kalman_denoise(
    df: pd.DataFrame,
    column: str = 'Close_Wavelet',
    vol_window: int = 50,
    base_obs_cov: float = 1.0,
    base_trans_cov: float = 0.01
) -> pd.DataFrame:
    """
    Filtro de Kalman com observation_covariance adaptativa ao regime de volatilidade.
    
    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame com os dados.
    column : str
        Coluna a ser filtrada.
    vol_window : int
        Janela para cálculo da volatilidade local.
    base_obs_cov : float
        Covariância de observação base. Quanto MAIOR, mais suavização.
    base_trans_cov : float
        Covariância de transição base. Quanto MENOR, sinal mais suave.
    
    Lógica Matemática:
    ------------------
    O Kalman padrão usa covariâncias fixas. Aqui adaptamos:
    
        R_t = R_base · (σ_local(t) / σ_global)²
    
    Em regime de ALTA volatilidade → R_t GRANDE → filtra mais agressivamente.
    Em regime de BAIXA volatilidade → R_t PEQUENO → segue o preço de perto.
    
    Isso é superior ao Kalman de parâmetros fixos porque o mercado alterna
    entre regimes (Hurst > 0.5 trending vs Hurst < 0.5 mean-reverting).
    """
    print(f"  🔬 Kalman Adaptativo em {len(df)} barras...")
    
    series = df[column].values
    
    # Calcular volatilidade local e global
    returns = np.log(series[1:] / series[:-1])
    returns = np.insert(returns, 0, 0)  # Pad para manter tamanho
    
    # Volatilidade local (rolling std)
    local_vol = pd.Series(returns).rolling(vol_window, min_periods=1).std().values
    global_vol = np.std(returns)
    
    if global_vol == 0:
        global_vol = 1e-8
    
    # Razão de volatilidade (regime indicator)
    vol_ratio = (local_vol / global_vol) ** 2
    vol_ratio = np.clip(vol_ratio, 0.1, 10.0)  # Limitar extremos
    
    # --- Kalman com covariância adaptativa ---
    # Primeira passada: Kalman padrão para obter estimativas iniciais
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=series[0],
        initial_state_covariance=1,
        observation_covariance=base_obs_cov,
        transition_covariance=base_trans_cov
    )
    
    # Segunda passada: Aplicar filtro iterativamente com R_t adaptativo
    n = len(series)
    filtered_state = np.zeros(n)
    filtered_state[0] = series[0]
    state_mean = series[0]
    state_cov = 1.0
    
    for t in range(1, n):
        # Prediction step
        pred_mean = state_mean  # x_{t|t-1} = x_{t-1|t-1} (random walk)
        pred_cov = state_cov + base_trans_cov  # P_{t|t-1}
        
        # Adaptive observation covariance
        R_t = base_obs_cov * vol_ratio[t]
        
        # Update step (correção de Kalman)
        innovation = series[t] - pred_mean  # y_t - H·x_{t|t-1}
        innovation_cov = pred_cov + R_t  # S_t = P_{t|t-1} + R_t
        kalman_gain = pred_cov / innovation_cov  # K_t = P_{t|t-1} / S_t
        
        state_mean = pred_mean + kalman_gain * innovation
        state_cov = (1 - kalman_gain) * pred_cov
        
        filtered_state[t] = state_mean
    
    df['Close_Kalman'] = filtered_state
    df['Returns_Kalman'] = np.log(
        df['Close_Kalman'] / df['Close_Kalman'].shift(1)
    )
    
    return df


# =============================================================================
# CAMADA 3: DENOISING DE MATRIZ DE CORRELAÇÃO (Marchenko-Pastur)
# =============================================================================

def denoise_correlation_matrix(
    returns_matrix: np.ndarray,
    bandwidth: float = 0.01
) -> np.ndarray:
    """
    Remove eigenvalues ruidosos da matriz de correlação usando
    a distribuição de Marchenko-Pastur como benchmark teórico.
    
    Parâmetros:
    -----------
    returns_matrix : np.ndarray
        Matriz T×N (T amostras, N features). Cada coluna é uma feature.
    bandwidth : float
        Parâmetro de suavização do kernel para estimação da densidade.
    
    Retorna:
    --------
    np.ndarray : Matriz de correlação denoised (N×N).
    
    Lógica Matemática:
    ------------------
    Para uma matriz aleatória T×N com entradas i.i.d. de média 0 e variância σ²,
    os eigenvalues da matriz de correlação amostral seguem a distribuição
    de Marchenko-Pastur:
    
        f(λ) = (T/N) · √((λ+ - λ)(λ - λ-)) / (2πσ²λ)
    
    com limites:
        λ± = σ²(1 ± √(N/T))²
    
    Eigenvalues ACIMA de λ+ → Sinal informacional real.
    Eigenvalues DENTRO de [λ-, λ+] → Indistinguíveis de ruído.
    
    Procedimento:
    1. Calcular eigendecomposition da correlação amostral
    2. Identificar λ+ (limiar do bulk)
    3. Substituir eigenvalues ruidosos pela média deles
    4. Reconstruir a matriz de correlação
    """
    T, N = returns_matrix.shape
    q = T / N  # Razão amostras/features
    
    if q <= 1:
        print("  ⚠️ T/N <= 1: Mais features que amostras. Marchenko-Pastur não aplicável.")
        return np.corrcoef(returns_matrix.T)
    
    # Matriz de correlação amostral
    corr_matrix = np.corrcoef(returns_matrix.T)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
    
    # Limites de Marchenko-Pastur
    sigma_sq = 1.0  # Para correlação padronizada
    lambda_plus = sigma_sq * (1 + np.sqrt(N / T)) ** 2
    lambda_minus = sigma_sq * (1 - np.sqrt(N / T)) ** 2
    
    # Separar sinal e ruído
    noise_mask = eigenvalues <= lambda_plus
    signal_mask = eigenvalues > lambda_plus
    
    n_signal = np.sum(signal_mask)
    n_noise = np.sum(noise_mask)
    
    print(f"  📊 Marchenko-Pastur: λ+ = {lambda_plus:.4f}")
    print(f"     Eigenvalues de sinal: {n_signal}/{N}")
    print(f"     Eigenvalues de ruído: {n_noise}/{N}")
    
    # Substituir eigenvalues ruidosos pela média (preserva traço)
    if n_noise > 0:
        noise_mean = np.mean(eigenvalues[noise_mask])
        eigenvalues_denoised = eigenvalues.copy()
        eigenvalues_denoised[noise_mask] = noise_mean
    else:
        eigenvalues_denoised = eigenvalues.copy()
    
    # Reconstruir matriz
    corr_denoised = eigenvectors @ np.diag(eigenvalues_denoised) @ eigenvectors.T
    
    # Normalizar para garantir diagonal = 1
    d = np.sqrt(np.diag(corr_denoised))
    d[d == 0] = 1e-10
    corr_denoised = corr_denoised / np.outer(d, d)
    np.fill_diagonal(corr_denoised, 1.0)
    
    return corr_denoised


# =============================================================================
# PIPELINE COMPLETO DE DENOISING
# =============================================================================

def run_denoising_pipeline(
    input_file: str = '../data/raw/dataset_raw.h5',
    output_file: str = '../data/processed/dataset_clean.h5',
    wavelet: str = 'db8',
    wavelet_level: int = 4,
    kalman_vol_window: int = 50,
    show_plots: bool = True
) -> pd.DataFrame:
    """
    Executa o pipeline completo de denoising em 3 camadas.
    
    Retorna o DataFrame com colunas adicionais:
        - Close_Wavelet: Preço após wavelet denoising
        - Close_Kalman: Preço após Kalman adaptativo
        - Returns_Kalman: Log-returns do sinal limpo
    """
    print("═" * 60)
    print("  PIPELINE DE DENOISING MULTI-CAMADA")
    print("═" * 60)
    
    # 1. Carregar dados
    try:
        df = pd.read_hdf(input_file, key='tick_bars')
        print(f"  📂 Dados carregados: {len(df)} barras de '{input_file}'")
    except (FileNotFoundError, KeyError):
        print(f"  ❌ Arquivo '{input_file}' não encontrado.")
        print("     Execute baixa_dados.py primeiro.")
        return None
    
    # 2. CAMADA 1: Wavelet Denoising
    print(f"\n  [1/3] Wavelet Transform ({wavelet}, level={wavelet_level})...")
    df['Close_Wavelet'] = wavelet_denoise(
        df['Close'].values,
        wavelet=wavelet,
        level=wavelet_level
    )
    print(f"     ✅ SNR improvement estimado: "
          f"{_estimate_snr_improvement(df['Close'].values, df['Close_Wavelet'].values):.1f} dB")
    
    # 3. CAMADA 2: Kalman Adaptativo
    print(f"\n  [2/3] Kalman Adaptativo (vol_window={kalman_vol_window})...")
    df = adaptive_kalman_denoise(
        df,
        column='Close_Wavelet',
        vol_window=kalman_vol_window
    )
    print(f"     ✅ Suavização completa.")
    
    # 4. CAMADA 3: Marchenko-Pastur (na matriz de features)
    print(f"\n  [3/3] Marchenko-Pastur Denoising...")
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Returns']
    available_cols = [c for c in numeric_cols if c in df.columns]
    
    if len(available_cols) >= 2:
        returns_for_mp = df[available_cols].dropna().values
        if returns_for_mp.shape[0] > returns_for_mp.shape[1]:
            denoised_corr = denoise_correlation_matrix(returns_for_mp)
            # Salvar a matriz denoised para uso posterior no calcula_alphas
            df.attrs['denoised_correlation'] = denoised_corr.tolist()
            print(f"     ✅ Matriz de correlação {denoised_corr.shape} denoised.")
        else:
            print(f"     ⚠️ Amostras insuficientes para Marchenko-Pastur.")
    
    # 5. Limpar NaN residuais
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # 6. Visualização
    if show_plots:
        _plot_denoising_comparison(df)
    
    # 7. Salvar dataset denoised
    import os
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_hdf(output_file, key='data', mode='w',
              format='table', complevel=6, complib='blosc:zstd')
    
    print(f"\n{'═' * 60}")
    print(f"  ✅ DENOISING COMPLETO")
    print(f"  Barras finais: {len(df):,}")
    print(f"  Arquivo salvo: {output_file}")
    print(f"{'═' * 60}")
    
    return df


def _estimate_snr_improvement(original: np.ndarray, denoised: np.ndarray) -> float:
    """Estima a melhoria de SNR em decibéis."""
    noise = original - denoised
    signal_power = np.var(denoised)
    noise_power = np.var(noise)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)


def _plot_denoising_comparison(df: pd.DataFrame, n_points: int = 500):
    """Visualização comparativa das 3 camadas de denoising."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    
    idx = slice(0, min(n_points, len(df)))
    
    # Painel 1: Preços
    axes[0].plot(df['Close'].iloc[idx], label='Original (Ruidoso)',
                 alpha=0.4, color='gray', linewidth=0.8)
    axes[0].plot(df['Close_Wavelet'].iloc[idx], label='Wavelet (DWT)',
                 alpha=0.7, color='blue', linewidth=1.0)
    axes[0].plot(df['Close_Kalman'].iloc[idx], label='Kalman Adaptativo',
                 color='red', linewidth=1.5)
    axes[0].set_title('Denoising Multi-Camada: Preço USDJPY (Tick Bars)', fontsize=14)
    axes[0].legend(loc='upper left', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylabel('Preço')
    
    # Painel 2: Retornos
    if 'Returns' in df.columns and 'Returns_Kalman' in df.columns:
        axes[1].plot(df['Returns'].iloc[idx], label='Returns Original',
                     alpha=0.3, color='gray', linewidth=0.5)
        axes[1].plot(df['Returns_Kalman'].iloc[idx], label='Returns Kalman',
                     color='red', linewidth=1.0)
        axes[1].set_title('Log-Returns: Original vs. Denoised', fontsize=14)
        axes[1].legend(loc='upper left', fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylabel('Log Return')
        axes[1].set_xlabel('Tick Bar Index')
    
    plt.tight_layout()
    plt.savefig('exports/plots/denoising_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# EXECUÇÃO
# =============================================================================
if __name__ == "__main__":
    from pathlib import Path
    Path('../data/processed').mkdir(parents=True, exist_ok=True)
    Path('../exports/plots').mkdir(parents=True, exist_ok=True)
    df_denoised = run_denoising_pipeline(
        input_file='../data/raw/dataset_raw.h5',
        output_file='../data/processed/dataset_clean.h5',
        wavelet='db8',
        wavelet_level=4,
        kalman_vol_window=50,
        show_plots=True
    )