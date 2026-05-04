"""
===============================================================================
MÓDULO 5: BACKTESTING ENGINE COM RIGOR ESTATÍSTICO
===============================================================================
Motor de backtest com Purged Walk-Forward Cross-Validation.

Componentes:
    1. PurgedWalkForwardCV — Cross-validation com Purge & Embargo
    2. PerformanceMetrics — Suite completa de métricas quantitativas
    3. BacktestEngine — Orquestrador que integra tudo

Purge & Embargo (López de Prado, 2018):
    |---Train---|---PURGE---|---Val---|---EMBARGO---|---Next Window---|

    PURGE (200 barras): Elimina autocorrelação serial residual entre
    o dataset de treino e validação.

    EMBARGO (50 barras): Período após a validação excluído de
    QUALQUER uso futuro. Previne leakage reverso.

Métricas de Performance:
    - Sharpe Ratio (anualizado para tick bars)
    - Sortino Ratio (penaliza apenas downside vol)
    - Information Coefficient — IC = ρ_Spearman(ŷ, y) [NOVO]
    - Max Drawdown (percentual do pico)
    - Information Ratio (vs. benchmark)
    - Hit Rate (acurácia direcional)
    - Calmar Ratio (return / MDD)
    - Profit Factor (gross_profit / gross_loss)
    - Directional Accuracy by Regime [NOVO]

Referências:
    - López de Prado (2018): "Advances in Financial Machine Learning" — Cap. 7
    - Bailey & López de Prado (2012): "The Sharpe Ratio Efficient Frontier"
    - Grinold & Kahn (2000): "Active Portfolio Management" — IC
===============================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Generator, Dict
from scipy.stats import spearmanr
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# 1. PURGED WALK-FORWARD CROSS-VALIDATION
# =============================================================================

class PurgedWalkForwardCV:
    """
    Walk-Forward Cross-Validation com Purge & Embargo.

    Parâmetros:
    -----------
    n_splits : int
        Número de folds de validação.
    purge_bars : int
        Gap (barras) entre treino e validação. Remove autocorrelação serial.
    embargo_bars : int
        Período (barras) após validação excluído. Previne leakage reverso.
    expanding : bool
        Se True, treino EXPANDE a cada fold (mais dados).
        Se False, tamanho fixo (sliding window).

    Lógica Matemática:
    ------------------
    Para cada fold i:
        train_end_i = base_size + i × step_size
        val_start_i = train_end_i + purge_bars
        val_end_i = val_start_i + val_size
        embargo_end_i = val_end_i + embargo_bars

    Garantia: ∀ i, j (i ≠ j):
        train_i ∩ [val_j - purge, val_j + embargo] = ∅
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_bars: int = 200,
        embargo_bars: int = 50,
        expanding: bool = True
    ):
        self.n_splits: int = n_splits
        self.purge_bars: int = purge_bars
        self.embargo_bars: int = embargo_bars
        self.expanding: bool = expanding

    def split(self, X) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Gera pares (train_idx, val_idx) com purge e embargo.

        Verificação bidirecional de leakage:
        - Forward: train_end + purge ≤ val_start
        - Backward: val_end + embargo ≤ next_train_start (if any)
        """
        n_samples = len(X) if hasattr(X, '__len__') else X.shape[0]

        total_overhead = self.purge_bars + self.embargo_bars
        usable = n_samples - total_overhead
        val_size = usable // (self.n_splits + 1)

        if val_size < 100:
            raise ValueError(
                f"Validação muito pequena ({val_size} barras). "
                f"Reduza n_splits, purge_bars ou embargo_bars."
            )

        for i in range(self.n_splits):
            if self.expanding:
                train_start = 0
            else:
                train_start = max(0, val_size * i)

            train_end = val_size * (i + 1)
            val_start = train_end + self.purge_bars
            val_end = val_start + val_size

            if val_end > n_samples:
                break

            train_idx = np.arange(train_start, train_end)
            val_idx = np.arange(val_start, val_end)

            yield train_idx, val_idx

    def get_n_splits(self) -> int:
        return self.n_splits

    def describe(self, n_samples: int) -> None:
        """Imprime descrição visual dos splits."""
        print(f"\n  PurgedWalkForwardCV Configuration:")
        print(f"  {'─' * 50}")
        print(f"  Total samples:  {n_samples:,}")
        print(f"  N splits:       {self.n_splits}")
        print(f"  Purge gap:      {self.purge_bars} barras")
        print(f"  Embargo:        {self.embargo_bars} barras")
        print(f"  Expanding:      {self.expanding}")
        print(f"  {'─' * 50}")

        for i, (train_idx, val_idx) in enumerate(self.split(np.arange(n_samples))):
            train_pct = len(train_idx) / n_samples * 100
            val_pct = len(val_idx) / n_samples * 100
            print(f"  Fold {i+1}: Train [{train_idx[0]:5d} → {train_idx[-1]:5d}] "
                  f"({len(train_idx):5d}, {train_pct:.1f}%) │ "
                  f"Purge │ Val [{val_idx[0]:5d} → {val_idx[-1]:5d}] "
                  f"({len(val_idx):5d}, {val_pct:.1f}%)")


# =============================================================================
# 2. PERFORMANCE METRICS
# =============================================================================

@dataclass
class PerformanceReport:
    """Container para métricas de performance do backtest."""
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    information_ratio: float = 0.0
    information_coefficient: float = 0.0  # IC = Spearman(predicted, actual)
    ic_p_value: float = 1.0              # p-value do IC
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    hit_rate: float = 0.0
    profit_factor: float = 0.0
    annualized_return: float = 0.0
    annualized_volatility: float = 0.0
    total_return: float = 0.0
    n_trades: int = 0
    n_bars: int = 0
    avg_trade_return: float = 0.0
    regime_accuracy: Optional[Dict[str, float]] = None

    def __str__(self) -> str:
        regime_str = ""
        if self.regime_accuracy:
            regime_str = (
                f"\n  ╠════════════════════════════════════════════════════╣"
                f"\n  ║  Regime Accuracy:                                 ║"
            )
            for regime, acc in self.regime_accuracy.items():
                regime_str += (
                    f"\n  ║    {regime:20s} {acc:10.4f}              ║"
                )

        return (
            f"\n  ╔════════════════════════════════════════════════════╗"
            f"\n  ║           RELATÓRIO DE PERFORMANCE                ║"
            f"\n  ╠════════════════════════════════════════════════════╣"
            f"\n  ║  Sharpe Ratio:        {self.sharpe_ratio:+10.4f}              ║"
            f"\n  ║  Sortino Ratio:       {self.sortino_ratio:+10.4f}              ║"
            f"\n  ║  Information Ratio:   {self.information_ratio:+10.4f}              ║"
            f"\n  ║  Information Coeff:   {self.information_coefficient:+10.4f} (p={self.ic_p_value:.4f})  ║"
            f"\n  ║  Calmar Ratio:        {self.calmar_ratio:+10.4f}              ║"
            f"\n  ╠════════════════════════════════════════════════════╣"
            f"\n  ║  Max Drawdown:        {self.max_drawdown:+10.4f}              ║"
            f"\n  ║  MDD Duration:        {self.max_drawdown_duration:10d} barras      ║"
            f"\n  ║  Hit Rate:            {self.hit_rate:10.4f}              ║"
            f"\n  ║  Profit Factor:       {self.profit_factor:10.4f}              ║"
            f"\n  ╠════════════════════════════════════════════════════╣"
            f"\n  ║  Return (Total):      {self.total_return:+10.4f}              ║"
            f"\n  ║  Return (Annual):     {self.annualized_return:+10.4f}              ║"
            f"\n  ║  Volatility (Annual): {self.annualized_volatility:10.4f}              ║"
            f"\n  ║  Trades:              {self.n_trades:10d}              ║"
            f"\n  ║  Avg Trade Return:    {self.avg_trade_return:+10.6f}              ║"
            f"{regime_str}"
            f"\n  ╚════════════════════════════════════════════════════╝"
        )


class PerformanceMetrics:
    """
    Calcula métricas de performance quantitativas para tick bars.

    Todas as métricas são ajustadas para a frequência de tick bars
    usando o fator de anualização K = barras_por_dia × 252.
    """

    @staticmethod
    def compute(
        strategy_returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None,
        predicted_returns: Optional[np.ndarray] = None,
        actual_returns: Optional[np.ndarray] = None,
        total_bars: Optional[int] = None,
        n_days: int = 90,
        risk_free_rate: float = 0.0,
        signals: Optional[np.ndarray] = None,
        hurst_values: Optional[np.ndarray] = None
    ) -> PerformanceReport:
        """
        Calcula todas as métricas de performance.

        Parâmetros:
        -----------
        strategy_returns : np.ndarray
            Retornos da estratégia (signal × return).
        predicted_returns : np.ndarray, optional
            Retornos preditos (para IC).
        actual_returns : np.ndarray, optional
            Retornos reais (para IC).
        hurst_values : np.ndarray, optional
            Valores de Hurst por barra (para regime accuracy).
        """
        report = PerformanceReport()

        if len(strategy_returns) == 0:
            return report

        if total_bars is None:
            total_bars = len(strategy_returns)

        # Fator de anualização
        bars_per_day = total_bars / max(n_days, 1)
        K = bars_per_day * 252

        report.n_bars = len(strategy_returns)
        report.total_return = float(np.sum(strategy_returns))

        # --- Sharpe Ratio ---
        mean_ret = float(np.mean(strategy_returns)) - risk_free_rate
        std_ret = float(np.std(strategy_returns))
        report.sharpe_ratio = (
            mean_ret / std_ret * np.sqrt(K)
        ) if std_ret > 0 else 0.0

        # --- Sortino Ratio ---
        downside_returns = strategy_returns[strategy_returns < 0]
        downside_std = (
            float(np.std(downside_returns)) if len(downside_returns) > 0 else 1e-10
        )
        report.sortino_ratio = (
            mean_ret / downside_std * np.sqrt(K)
        ) if downside_std > 0 else 0.0

        # --- Max Drawdown ---
        cumulative = np.cumsum(strategy_returns)
        equity = 1 + cumulative
        rolling_max = np.maximum.accumulate(equity)
        drawdowns = (equity - rolling_max) / rolling_max
        report.max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

        # MDD Duration
        peak_idx = int(np.argmax(equity))
        if peak_idx < len(equity) - 1:
            post_peak = equity[peak_idx:]
            recovery_mask = post_peak >= equity[peak_idx]
            if np.any(recovery_mask[1:]):
                recovery_idx = int(np.where(recovery_mask[1:])[0][0]) + 1
                report.max_drawdown_duration = recovery_idx
            else:
                report.max_drawdown_duration = len(post_peak)

        # --- Hit Rate ---
        report.hit_rate = float(np.mean(strategy_returns > 0))

        # --- Profit Factor ---
        gross_profit = float(np.sum(strategy_returns[strategy_returns > 0]))
        gross_loss = float(abs(np.sum(strategy_returns[strategy_returns < 0])))
        report.profit_factor = (
            gross_profit / gross_loss
        ) if gross_loss > 0 else float('inf')

        # --- Annualized Return & Volatility ---
        report.annualized_return = mean_ret * K
        report.annualized_volatility = std_ret * np.sqrt(K)

        # --- Calmar Ratio ---
        report.calmar_ratio = (
            report.annualized_return / abs(report.max_drawdown)
        ) if report.max_drawdown != 0 else 0.0

        # --- Information Ratio ---
        if benchmark_returns is not None:
            active_returns = strategy_returns - benchmark_returns
        else:
            active_returns = strategy_returns

        active_std = float(np.std(active_returns))
        report.information_ratio = (
            float(np.mean(active_returns)) / active_std * np.sqrt(K)
        ) if active_std > 0 else 0.0

        # --- Information Coefficient (IC) [NOVO] ---
        if predicted_returns is not None and actual_returns is not None:
            report.information_coefficient, report.ic_p_value = (
                information_coefficient(predicted_returns, actual_returns)
            )

        # --- Trade Statistics ---
        if signals is not None:
            report.n_trades = int(np.sum(signals != 0))
            if report.n_trades > 0:
                traded_returns = strategy_returns[signals != 0]
                report.avg_trade_return = float(np.mean(traded_returns))
        else:
            report.n_trades = len(strategy_returns)
            report.avg_trade_return = float(np.mean(strategy_returns))

        # --- Directional Accuracy by Regime [NOVO] ---
        if hurst_values is not None and signals is not None:
            report.regime_accuracy = directional_accuracy_by_regime(
                strategy_returns, signals, hurst_values
            )

        return report


# =============================================================================
# INFORMATION COEFFICIENT [NOVO]
# =============================================================================

def information_coefficient(
    predicted: np.ndarray,
    actual: np.ndarray
) -> Tuple[float, float]:
    """
    Information Coefficient via Spearman Rank Correlation.

    IC = ρ_Spearman(ŷ, y)

    O IC mede a correlação entre o RANKING das predições e
    o RANKING dos retornos reais. É robusto a outliers porque
    usa rankings em vez de valores absolutos.

    Interpretação (Grinold & Kahn, 2000):
        IC > 0.05 → sinal preditivo fraco mas explorável
        IC > 0.10 → sinal preditivo bom
        IC > 0.20 → sinal excepcional (raro em prática)

    Retorna:
    --------
    (ic, p_value) : Tuple[float, float]
    """
    # Alinhar tamanhos
    min_len = min(len(predicted), len(actual))
    pred = predicted[:min_len]
    real = actual[:min_len]

    # Remover NaN
    mask = ~(np.isnan(pred) | np.isnan(real))
    pred = pred[mask]
    real = real[mask]

    if len(pred) < 10:
        return 0.0, 1.0

    try:
        ic, p_value = spearmanr(pred, real)
        return float(ic), float(p_value)
    except Exception:
        return 0.0, 1.0


# =============================================================================
# DIRECTIONAL ACCURACY BY REGIME [NOVO]
# =============================================================================

def directional_accuracy_by_regime(
    strategy_returns: np.ndarray,
    signals: np.ndarray,
    hurst_values: np.ndarray
) -> Dict[str, float]:
    """
    Acurácia direcional separada por regime de mercado.

    Regimes (via Hurst Exponent):
        H > 0.55 → Trending (momentum)
        0.45 ≤ H ≤ 0.55 → Random Walk (noise)
        H < 0.45 → Mean-Reverting (contrarian)

    Retorna:
    --------
    dict com acurácia por regime:
        {'trending': acc, 'random_walk': acc, 'mean_reverting': acc}
    """
    min_len = min(len(strategy_returns), len(signals), len(hurst_values))
    sr = strategy_returns[:min_len]
    sig = signals[:min_len]
    hurst = hurst_values[:min_len]

    # Remover NaN
    mask = ~np.isnan(hurst) & (sig != 0)

    regimes: Dict[str, float] = {}

    # Trending
    trending_mask = mask & (hurst > 0.55)
    if np.sum(trending_mask) > 10:
        regimes['Trending (H>0.55)'] = float(np.mean(sr[trending_mask] > 0))

    # Random Walk
    rw_mask = mask & (hurst >= 0.45) & (hurst <= 0.55)
    if np.sum(rw_mask) > 10:
        regimes['Random Walk (0.45≤H≤0.55)'] = float(np.mean(sr[rw_mask] > 0))

    # Mean Reverting
    mr_mask = mask & (hurst < 0.45)
    if np.sum(mr_mask) > 10:
        regimes['Mean-Revert (H<0.45)'] = float(np.mean(sr[mr_mask] > 0))

    return regimes


# =============================================================================
# 3. BACKTEST ENGINE
# =============================================================================

class BacktestEngine:
    """
    Motor de backtest com rigor estatístico.

    Integra PurgedWalkForwardCV com PerformanceMetrics
    para gerar relatórios completos e equity curves.
    """

    def __init__(
        self,
        purge_bars: int = 200,
        embargo_bars: int = 50,
        n_splits: int = 5,
        confidence_threshold: float = 0.60,
        n_days: int = 90
    ):
        self.purge_bars: int = purge_bars
        self.embargo_bars: int = embargo_bars
        self.n_splits: int = n_splits
        self.confidence_threshold: float = confidence_threshold
        self.n_days: int = n_days

        self.cv = PurgedWalkForwardCV(
            n_splits=n_splits,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
            expanding=True
        )

    def evaluate(
        self,
        ensemble_probs: np.ndarray,
        real_returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None,
        predicted_returns: Optional[np.ndarray] = None,
        hurst_values: Optional[np.ndarray] = None,
        total_bars: Optional[int] = None
    ) -> PerformanceReport:
        """
        Avalia performance com métricas completas incluindo IC.

        Parâmetros:
        -----------
        ensemble_probs : np.ndarray
            Probabilidades do ensemble P(up) ∈ [0, 1].
        real_returns : np.ndarray
            Retornos reais (log returns).
        predicted_returns : np.ndarray, optional
            Retornos preditos pelo modelo (para IC).
        hurst_values : np.ndarray, optional
            Hurst exponent por barra (para regime accuracy).
        """
        # Gerar sinais com filtro de confiança
        signals = self._generate_signals(ensemble_probs)

        # Retornos da estratégia
        strategy_returns = signals * real_returns

        # Calcular métricas
        report = PerformanceMetrics.compute(
            strategy_returns=strategy_returns,
            benchmark_returns=benchmark_returns,
            predicted_returns=predicted_returns,
            actual_returns=real_returns,
            total_bars=total_bars or len(real_returns),
            n_days=self.n_days,
            signals=signals,
            hurst_values=hurst_values
        )

        return report

    def evaluate_projection(
        self,
        projected_returns: np.ndarray,
        actual_returns: np.ndarray,
        total_bars: Optional[int] = None
    ) -> PerformanceReport:
        """
        Avalia performance da projeção OHLC (para o Conselho de Especialistas).

        Gera sinais diretamente dos retornos projetados
        (positivo → long, negativo → short).
        """
        signals = np.sign(projected_returns)
        strategy_returns = signals * actual_returns

        report = PerformanceMetrics.compute(
            strategy_returns=strategy_returns,
            predicted_returns=projected_returns,
            actual_returns=actual_returns,
            total_bars=total_bars or len(actual_returns),
            n_days=self.n_days,
            signals=signals
        )

        return report

    def _generate_signals(self, probs: np.ndarray) -> np.ndarray:
        """
        Gera sinais de trading com filtro de confiança.

        prob > threshold     → Long  (+1)
        prob < 1 - threshold → Short (-1)
        otherwise            → Flat  (0)
        """
        return np.where(
            probs > self.confidence_threshold, 1,
            np.where(probs < (1 - self.confidence_threshold), -1, 0)
        )

    def plot_results(
        self,
        ensemble_probs: np.ndarray,
        real_returns: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """Gera visualizações completas do backtest."""
        signals = self._generate_signals(ensemble_probs)
        strategy_returns = signals * real_returns

        fig, axes = plt.subplots(4, 1, figsize=(18, 16),
                                  gridspec_kw={'height_ratios': [3, 1, 1, 1]})

        # --- Painel 1: Equity Curve ---
        cum_strategy = np.cumsum(strategy_returns)
        cum_benchmark = np.cumsum(real_returns)

        axes[0].fill_between(range(len(cum_strategy)), cum_strategy,
                             alpha=0.3, color='green', label='Estratégia (Filled)')
        axes[0].plot(cum_strategy, color='green', linewidth=1.5, label='Estratégia')
        axes[0].plot(cum_benchmark, color='gray', alpha=0.5, linewidth=1.0,
                     label='Benchmark (Buy & Hold)')
        axes[0].set_title('Equity Curve — Conselho de Especialistas',
                          fontsize=14, fontweight='bold')
        axes[0].legend(loc='upper left', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylabel('Retorno Cumulativo')

        # --- Painel 2: Drawdown ---
        equity = 1 + cum_strategy
        rolling_max = np.maximum.accumulate(equity)
        drawdown = (equity - rolling_max) / rolling_max * 100

        axes[1].fill_between(range(len(drawdown)), drawdown,
                             alpha=0.5, color='red')
        axes[1].set_title('Drawdown (%)', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylabel('Drawdown %')

        # --- Painel 3: Sinais ---
        colors = np.where(signals == 1, 'green',
                          np.where(signals == -1, 'red', 'gray'))
        axes[2].scatter(range(len(signals)), signals, c=colors, s=1, alpha=0.5)
        axes[2].set_title('Sinais de Trading', fontsize=12)
        axes[2].set_ylabel('Sinal')
        axes[2].set_yticks([-1, 0, 1])
        axes[2].set_yticklabels(['Short', 'Flat', 'Long'])
        axes[2].grid(True, alpha=0.3)

        # --- Painel 4: Probabilidade do Ensemble ---
        axes[3].plot(ensemble_probs, color='blue', alpha=0.5, linewidth=0.5)
        axes[3].axhline(y=self.confidence_threshold, color='green',
                        linestyle='--', alpha=0.7,
                        label=f'Long threshold ({self.confidence_threshold})')
        axes[3].axhline(y=1 - self.confidence_threshold, color='red',
                        linestyle='--', alpha=0.7,
                        label=f'Short threshold ({1-self.confidence_threshold})')
        axes[3].set_title('Probabilidade do Ensemble P(Up)', fontsize=12)
        axes[3].legend(loc='upper right', fontsize=9)
        axes[3].grid(True, alpha=0.3)
        axes[3].set_ylabel('P(Up)')
        axes[3].set_xlabel('Tick Bar Index')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  📊 Gráfico salvo: {save_path}")

        plt.show()

    def validate_no_leakage(self, n_samples: int) -> bool:
        """
        Verifica formalmente que não há Data Leakage nos splits.

        Testa bidirecional:
            Forward:  max(train_i) + purge_bars ≤ min(val_j)
            Backward: max(val_i) + embargo ≤ min(train_{i+1}) (se expanding=False)
        """
        print("\n  🔍 Validação de Data Leakage (bidirecional):")
        all_valid = True
        prev_val_end = -1

        for i, (train_idx, val_idx) in enumerate(self.cv.split(np.arange(n_samples))):
            train_end = train_idx[-1]
            val_start = val_idx[0]
            val_end = val_idx[-1]

            # Forward check
            gap = val_start - train_end
            forward_valid = gap >= self.purge_bars

            # Backward check (treino não contém dados do val anterior + embargo)
            backward_valid = True
            if prev_val_end >= 0 and not self.cv.expanding:
                backward_gap = train_idx[0] - prev_val_end
                backward_valid = backward_gap >= self.embargo_bars

            valid = forward_valid and backward_valid
            status = "✅" if valid else "❌"
            print(f"    Fold {i+1}: train_end={train_end}, val=[{val_start}→{val_end}], "
                  f"gap={gap} {'≥' if forward_valid else '<'} {self.purge_bars} {status}")

            if not valid:
                all_valid = False

            prev_val_end = val_end

        if all_valid:
            print("    ✅ ZERO Data Leakage — Todos os gaps válidos!")
        else:
            print("    ❌ LEAKAGE DETECTADO — Corrija os parâmetros!")

        return all_valid


# =============================================================================
# EXECUÇÃO STANDALONE
# =============================================================================
if __name__ == "__main__":
    print("═" * 60)
    print("  BACKTEST ENGINE — TESTE DE INTEGRIDADE")
    print("═" * 60)

    # Teste com dados sintéticos
    n_samples = 5000
    np.random.seed(42)

    noise = np.random.normal(0, 0.15, n_samples)
    true_signal = np.sin(np.linspace(0, 10 * np.pi, n_samples)) * 0.1
    fake_probs = 0.5 + true_signal + noise
    fake_probs = np.clip(fake_probs, 0, 1)

    fake_returns = np.random.normal(0, 0.001, n_samples) + true_signal * 0.001
    fake_predicted = fake_returns + np.random.normal(0, 0.0005, n_samples)
    fake_hurst = 0.5 + np.sin(np.linspace(0, 4 * np.pi, n_samples)) * 0.15

    # 1. Testar PurgedWalkForwardCV
    engine = BacktestEngine(
        purge_bars=200,
        embargo_bars=50,
        n_splits=5,
        confidence_threshold=0.60
    )

    engine.cv.describe(n_samples)
    engine.validate_no_leakage(n_samples)

    # 2. Avaliar performance com IC e regime accuracy
    report = engine.evaluate(
        ensemble_probs=fake_probs,
        real_returns=fake_returns,
        predicted_returns=fake_predicted,
        hurst_values=fake_hurst,
        total_bars=n_samples
    )
    print(report)

    # 3. Testar IC isolado
    ic, p_val = information_coefficient(fake_predicted, fake_returns)
    print(f"\n  IC = {ic:.4f} (p-value = {p_val:.4f})")

    # 4. Plot
    engine.plot_results(fake_probs, fake_returns, save_path='backtest_results.png')

    print("\n  ✅ Backtest Engine operacional!")
