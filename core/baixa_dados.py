"""
===============================================================================
MÓDULO 1: EXTRAÇÃO DE DADOS E GERAÇÃO DE BARRAS AVANÇADAS
===============================================================================
Autor: Pipeline HFT — Conselho de Especialistas (MoE)
Ativo: USDJPY

Teoria:
    Barras alternativas amostram no espaço de ATIVIDADE de mercado,
    não no tempo. Isso elimina a sazonalidade intradiária e garante
    que cada barra carrega informação equivalente.

    Tipos de barra suportados:
        1. Tick Bars   — cada barra = N ticks (mudanças de preço)
        2. Volume Bars — cada barra = V unidades de volume (tick count proxy)
        3. Dollar Bars — cada barra = D unidades de price × volume

    Para Forex OTC (USDJPY), o tick count é o melhor proxy de volume
    disponível (correlação >0.90 com volume real em ECN).
    Volume Bars baseadas em tick count capturam a atividade nos horários
    de sobreposição Tóquio-Londres de forma estocástica.

    Referência: Mandelbrot (1963), Clark (1973), López de Prado (2018)
    "Advances in Financial Machine Learning" — Cap. 2: Financial Data Structures

Mudanças vs. versão anterior:
    1. Suporte a Volume Bars e Dollar Bars (além de Tick Bars)
    2. Auto-detecção de thresholds via média histórica
    3. _aggregate_variable_bars() para amostragem estocástica
    4. bar_id global contínuo entre chunks
    5. HDF5 com compressão blosc:zstd e chunking otimizado
===============================================================================
"""

from __future__ import annotations

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Literal
import time
import h5py
import os


# =============================================================================
# AGREGAÇÃO DE BARRAS VARIÁVEIS (Volume / Dollar)
# =============================================================================

def _aggregate_variable_bars(
    df_ticks: pd.DataFrame,
    bar_type: Literal['tick', 'volume', 'dollar'],
    threshold: float,
    global_bar_id: int
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Agrega ticks em barras variáveis baseadas em atividade de mercado.

    Parâmetros:
    -----------
    df_ticks : pd.DataFrame
        DataFrame de ticks brutos com colunas: time, bid, ask, spread,
        midprice, volume_real.
    bar_type : str
        'tick', 'volume', ou 'dollar'.
    threshold : float
        Limiar de acumulação:
          - tick:   número de ticks por barra
          - volume: soma de volume_real (tick count) por barra
          - dollar: soma de price × volume_real por barra
    global_bar_id : int
        ID global da próxima barra (continuidade entre chunks).

    Retorna:
    --------
    (bars_df, residual_ticks, next_bar_id)

    Lógica Matemática:
    ------------------
    Para cada tipo de barra, a acumulação segue:

        Tick:   cumulative_metric = count(ticks)
        Volume: cumulative_metric = Σ volume_real_i
        Dollar: cumulative_metric = Σ (bid_i × volume_real_i)

    Quando cumulative_metric >= threshold, uma nova barra é emitida com:
        Open  = bid do primeiro tick
        High  = max(bid) dos ticks acumulados
        Low   = min(bid) dos ticks acumulados
        Close = bid do último tick
    """
    if len(df_ticks) == 0:
        return pd.DataFrame(), df_ticks, global_bar_id

    bars: list[dict] = []
    acc_start: int = 0
    cumulative: float = 0.0

    bids = df_ticks['bid'].values
    volumes = df_ticks['volume_real'].values
    times = df_ticks['time'].values

    asks = df_ticks['ask'].values if 'ask' in df_ticks.columns else bids
    spreads = df_ticks['spread'].values if 'spread' in df_ticks.columns else np.zeros(len(bids))
    midprices = df_ticks['midprice'].values if 'midprice' in df_ticks.columns else bids

    for i in range(len(df_ticks)):
        if bar_type == 'tick':
            cumulative += 1.0
        elif bar_type == 'volume':
            cumulative += volumes[i]
        elif bar_type == 'dollar':
            cumulative += bids[i] * volumes[i]

        if cumulative >= threshold:
            # Emitir barra
            sl = slice(acc_start, i + 1)
            bar_bids = bids[sl]
            bar_asks = asks[sl]
            bar_spreads = spreads[sl]
            bar_vols = volumes[sl]

            bar = {
                'bar_id': global_bar_id,
                'Date': times[acc_start],
                'Open': bar_bids[0],
                'High': np.max(bar_bids),
                'Low': np.min(bar_bids),
                'Close': bar_bids[-1],
                'Ask_Open': bar_asks[0],
                'Ask_High': np.max(bar_asks),
                'Ask_Low': np.min(bar_asks),
                'Ask_Close': bar_asks[-1],
                'Volume': np.sum(bar_vols),
                'Tick_Count': len(bar_bids),
                'Spread_Mean': np.mean(bar_spreads),
                'Spread_Max': np.max(bar_spreads),
                'Midprice_Close': midprices[i],
            }
            bars.append(bar)
            global_bar_id += 1
            acc_start = i + 1
            cumulative = 0.0

    # Ticks residuais (não completaram uma barra)
    residual = df_ticks.iloc[acc_start:].copy() if acc_start < len(df_ticks) else pd.DataFrame()

    bars_df = pd.DataFrame(bars) if bars else pd.DataFrame()
    return bars_df, residual, global_bar_id


def _auto_threshold(
    total_metric: float,
    n_desired_bars: int,
    bar_type: str
) -> float:
    """
    Auto-detecta o threshold ideal para atingir ~n_desired_bars barras.

    threshold = total_metric / n_desired_bars

    Para tick bars, isso é simplesmente total_ticks / n_bars.
    Para volume bars, total_volume / n_bars.
    Para dollar bars, total_dollar_volume / n_bars.
    """
    threshold = total_metric / max(n_desired_bars, 1)
    print(f"  Auto-threshold ({bar_type}): {threshold:,.2f}")
    return threshold


# =============================================================================
# PIPELINE PRINCIPAL DE EXTRAÇÃO
# =============================================================================

def backfill_ticks_to_bars(
    symbol: str = "USDJPY",
    days_back: int = 90,
    bar_type: Literal['tick', 'volume', 'dollar'] = 'tick',
    threshold: Optional[float] = None,
    ticks_per_bar: int = 200,
    output_file: str = "../data/raw/dataset_raw.h5"
) -> Optional[pd.DataFrame]:
    """
    Extrai ticks brutos do MT5 e comprime em barras alternativas.

    Parâmetros:
    -----------
    symbol : str
        Símbolo do ativo no MT5.
    days_back : int
        Número de dias para extrair retroativamente.
    bar_type : str
        Tipo de barra: 'tick', 'volume', ou 'dollar'.
        - tick:   cada barra = ticks_per_bar ticks
        - volume: cada barra acumula threshold de tick count (volume proxy)
        - dollar: cada barra acumula threshold de price × tick count
    threshold : float, optional
        Limiar de acumulação para volume/dollar bars.
        Se None, auto-detecta via primeira passada nos dados.
        Para tick bars, usa ticks_per_bar.
    ticks_per_bar : int
        Usado apenas se bar_type='tick'. Número de ticks por barra.
    output_file : str
        Caminho do arquivo HDF5 de saída.

    Retorna:
    --------
    pd.DataFrame com as barras geradas, ou None se falhar.

    Nota sobre Forex OTC (USDJPY):
    -------------------------------
    volume_real no MT5 para Forex é tick count (não volume monetário).
    Tick count tem correlação >0.90 com volume real em ECN e serve como
    proxy robusto para atividade de mercado. Dollar Bars usarão
    price × tick_count como Dollar Volume proxy.
    """
    if not mt5.initialize():
        print("❌ Erro ao iniciar MT5. Verifique se o terminal está aberto.")
        return None

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    # Determinar threshold efetivo
    effective_threshold: float
    if bar_type == 'tick':
        effective_threshold = float(ticks_per_bar)
    elif threshold is not None:
        effective_threshold = threshold
    else:
        # Auto-detect: primeira passada rápida para estimar métricas
        print(f"  Estimando threshold automático para {bar_type} bars...")
        sample_ticks = mt5.copy_ticks_range(
            symbol, start_date, start_date + timedelta(days=min(7, days_back)),
            mt5.COPY_TICKS_ALL
        )
        if sample_ticks is not None and len(sample_ticks) > 0:
            df_sample = pd.DataFrame(sample_ticks)
            if 'volume_real' not in df_sample.columns:
                df_sample['volume_real'] = df_sample.get('volume', 1)

            if bar_type == 'volume':
                total_metric = df_sample['volume_real'].sum()
            else:  # dollar
                total_metric = (df_sample['bid'] * df_sample['volume_real']).sum()

            # Estimativa: ~50 barras por dia como target
            n_target = 50 * days_back
            daily_metric = total_metric / min(7, days_back)
            total_estimated = daily_metric * days_back
            effective_threshold = _auto_threshold(total_estimated, n_target, bar_type)
        else:
            print("  ⚠️ Não foi possível estimar threshold. Usando fallback = 200.")
            effective_threshold = 200.0

    all_bars: list[pd.DataFrame] = []
    global_bar_id: int = 0
    residual_ticks = pd.DataFrame()

    current_date = start_date
    chunk_size = timedelta(days=1)

    print(f"═══════════════════════════════════════════════════════════")
    print(f"  EXTRAÇÃO DE TICKS: {symbol}")
    print(f"  Período: {start_date.date()} → {end_date.date()} ({days_back} dias)")
    print(f"  Tipo de barra: {bar_type.upper()}")
    print(f"  Threshold: {effective_threshold:,.2f}")
    print(f"═══════════════════════════════════════════════════════════")

    total_ticks_extracted: int = 0
    days_processed: int = 0

    while current_date < end_date:
        chunk_end = min(current_date + chunk_size, end_date)

        # 1. Extração do bloco de ticks
        ticks = mt5.copy_ticks_range(
            symbol, current_date, chunk_end, mt5.COPY_TICKS_ALL
        )

        if ticks is not None and len(ticks) > 0:
            df_ticks = pd.DataFrame(ticks)
            total_ticks_extracted += len(df_ticks)

            # --- Resolução de timestamp via time_msc (milissegundos) ---
            if 'time_msc' in df_ticks.columns:
                df_ticks['time'] = pd.to_datetime(df_ticks['time_msc'], unit='ms')
            else:
                df_ticks['time'] = pd.to_datetime(df_ticks['time'], unit='s')

            # Calcular spread e midprice no nível de tick
            if 'ask' in df_ticks.columns and 'bid' in df_ticks.columns:
                df_ticks['spread'] = df_ticks['ask'] - df_ticks['bid']
                df_ticks['midprice'] = (df_ticks['ask'] + df_ticks['bid']) / 2.0
            else:
                df_ticks['spread'] = 0
                df_ticks['midprice'] = df_ticks['bid']

            # Garantir coluna de volume
            if 'volume_real' not in df_ticks.columns:
                df_ticks['volume_real'] = df_ticks.get('volume', 1)

            # --- Concatenar com ticks residuais do chunk anterior ---
            if len(residual_ticks) > 0:
                df_ticks = pd.concat([residual_ticks, df_ticks], ignore_index=True)
                residual_ticks = pd.DataFrame()

            # 2. Agregação via função unificada
            bars_chunk, residual_ticks, global_bar_id = _aggregate_variable_bars(
                df_ticks, bar_type, effective_threshold, global_bar_id
            )

            if len(bars_chunk) > 0:
                all_bars.append(bars_chunk)

        days_processed += 1
        pct = days_processed / days_back * 100
        n_bars_so_far = sum(len(b) for b in all_bars)
        print(f"  [{pct:5.1f}%] {current_date.date()} → {n_bars_so_far} barras | "
              f"{total_ticks_extracted:,} ticks extraídos", end='\r')

        current_date = chunk_end
        time.sleep(0.05)  # Pausa técnica para o terminal MT5

    mt5.shutdown()

    # 3. Consolidação final
    if len(all_bars) > 0:
        df_final = pd.concat(all_bars).reset_index(drop=True)

        if 'bar_id' in df_final.columns:
            df_final.drop(columns=['bar_id'], inplace=True)

        # Log Returns estacionários
        df_final['Returns'] = np.log(df_final['Close'] / df_final['Close'].shift(1))

        # Dollar Volume (proxy para atividade econômica)
        df_final['Dollar_Volume'] = df_final['Close'] * df_final['Volume']

        # Limpar NaN da primeira linha
        df_final.dropna(inplace=True)
        df_final.reset_index(drop=True, inplace=True)

        # 4. Persistência otimizada com compressão
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        _save_optimized_h5(df_final, output_file, symbol, days_back,
                           bar_type, effective_threshold)

        print(f"\n{'═' * 59}")
        print(f"  ✅ EXTRAÇÃO CONCLUÍDA!")
        print(f"  Tipo de barra: {bar_type.upper()}")
        print(f"  Total de barras: {len(df_final):,}")
        print(f"  Total de ticks processados: {total_ticks_extracted:,}")
        print(f"  Compressão: ~{total_ticks_extracted // max(len(df_final), 1)}:1")
        print(f"  Arquivo: {output_file}")
        print(f"{'═' * 59}")

        return df_final
    else:
        print("\n❌ Nenhum dado foi extraído.")
        return None


def _save_optimized_h5(
    df: pd.DataFrame,
    filepath: str,
    symbol: str,
    days_back: int,
    bar_type: str,
    threshold: float
) -> None:
    """
    Salva DataFrame em HDF5 com compressão e metadados.

    Usa format='table' para permitir queries seletivas (WHERE clauses)
    e compressão blosc:zstd para I/O otimizado.
    """
    # Salvar dados tabulares com compressão
    df.to_hdf(
        filepath, key='tick_bars', mode='w',
        format='table',
        complevel=6,
        complib='blosc:zstd'
    )

    # Adicionar metadados ao arquivo
    with h5py.File(filepath, 'a') as f:
        meta = f.require_group('metadata')
        meta.attrs['symbol'] = symbol
        meta.attrs['days_back'] = days_back
        meta.attrs['bar_type'] = bar_type
        meta.attrs['threshold'] = threshold
        meta.attrs['n_bars'] = len(df)
        meta.attrs['extraction_date'] = datetime.now().isoformat()
        meta.attrs['columns'] = list(df.columns)

    # Reportar tamanho do arquivo
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  💾 Arquivo salvo: {file_size_mb:.1f} MB (compressão blosc:zstd)")


# =============================================================================
# EXECUÇÃO DO BACKFILL
# =============================================================================
if __name__ == "__main__":
    from pathlib import Path
    Path('../data/raw').mkdir(parents=True, exist_ok=True)
    df_hist = backfill_ticks_to_bars(
        symbol="USDJPY",
        days_back=90,
        bar_type='tick',          # 'tick', 'volume', ou 'dollar'
        ticks_per_bar=200,
        output_file="../data/raw/dataset_raw.h5"
    )