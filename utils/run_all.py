# -*- coding: utf-8 -*-
"""
===============================================================================
ORQUESTRADOR DE PIPELINE — EXECUÇÃO SEQUENCIAL COM VALIDAÇÃO
===============================================================================
Executa os 5 estágios do pipeline MoE em sequência, verificando
a integridade de cada artefato antes de avançar ao próximo estágio.

Cadeia de dados:
    1. baixa_dados.py     → data/raw/dataset_raw.h5
    2. limpaArquivos.py   → data/processed/dataset_clean.h5
    3. calcula_alphas.py  → data/final/dataset_final.h5
    4. moe_to_daily.py    → models/{PAIR}/ + exports/inference/
    5. moe_visualization.py → exports/plots/

Uso:
    python run_all.py                    # Pipeline completo (1-5)
    python run_all.py --from 3           # A partir do estágio 3
    python run_all.py --only 4           # Apenas estágio 4
    python run_all.py --dry-run          # Verificar artefatos sem executar
    python run_all.py --pair EURUSD      # Par de moedas diferente
===============================================================================
"""
from __future__ import annotations

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional

# ── Forçar UTF-8 no console (Windows cp1252 não suporta emojis) ──
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ── Garantir que o diretório raiz do projeto (utils/) está no sys.path ──
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# ── Garantir que core/ está no sys.path para imports dos scripts ──
_CORE_DIR = _THIS_DIR / 'core'
if str(_CORE_DIR) not in sys.path:
    sys.path.insert(0, str(_CORE_DIR))

from pipeline_config import (
    PATHS, ARTIFACTS, ensure_dirs, get_model_dir,
    get_artifact_path, setup_pipeline_logger,
)

import pandas as pd


# =============================================================================
# CONFIGURAÇÃO DO PIPELINE
# =============================================================================

def _build_pipeline_stages(pair: str) -> dict:
    """Constrói mapa de estágios com caminhos resolvidos pelo pair."""
    return {
        1: {
            'name': 'Ingestão de Dados (MT5)',
            'output_file': str(get_artifact_path('dataset_raw', pair)),
            'output_key': 'tick_bars',
            'min_size_kb': 100,
            'description': 'Extrai ticks do MetaTrader 5 e comprime em Dollar/Tick/Volume Bars',
        },
        2: {
            'name': 'Denoising Multi-Camada',
            'input_file': str(get_artifact_path('dataset_raw', pair)),
            'output_file': str(get_artifact_path('dataset_clean', pair)),
            'output_key': 'data',
            'min_size_kb': 50,
            'description': 'Wavelet + Kalman + Marchenko-Pastur + Outlier Clipping',
        },
        3: {
            'name': 'Engenharia de Alphas',
            'input_file': str(get_artifact_path('dataset_clean', pair)),
            'output_file': str(get_artifact_path('dataset_final', pair)),
            'output_key': 'features',
            'min_size_kb': 50,
            'description': '13 Alphas + FracDiff + Triple Barrier + Normalização',
        },
        4: {
            'name': 'Treino MoE + Inferência',
            'input_file': str(get_artifact_path('dataset_final', pair)),
            'output_file': str(get_artifact_path('results_inference', pair)),
            'output_key': None,
            'min_size_kb': 1,
            'description': 'Walk-Forward + 4 Experts + Gating + Conformal Prediction',
        },
        5: {
            'name': 'Visualização Pós-Treino',
            'input_file': str(get_artifact_path('results_inference', pair)),
            'output_file': str(get_artifact_path('plot_analysis', pair)),
            'output_key': None,
            'min_size_kb': 10,
            'description': 'Candlestick + Ghost Projection + Rolling Backtest 30d',
        },
    }


# =============================================================================
# UTILIDADES DE VALIDAÇÃO
# =============================================================================

def validate_artifact(filepath: str, key: Optional[str], min_size_kb: int) -> dict:
    """
    Valida integridade de um artefato do pipeline.

    Returns
    -------
    dict com: exists, size_kb, n_rows (se HDF5), valid, error
    """
    result = {
        'exists': False, 'size_kb': 0, 'n_rows': None,
        'valid': False, 'error': None,
    }

    if not os.path.exists(filepath):
        result['error'] = 'Arquivo não encontrado'
        return result

    result['exists'] = True
    size_bytes = os.path.getsize(filepath)
    result['size_kb'] = size_bytes / 1024

    if result['size_kb'] < min_size_kb:
        result['error'] = f'Arquivo muito pequeno ({result["size_kb"]:.1f} KB < {min_size_kb} KB mínimo)'
        return result

    # Validar conteúdo para HDF5
    if filepath.endswith('.h5') and key is not None:
        try:
            df = pd.read_hdf(filepath, key=key)
            result['n_rows'] = len(df)
            if len(df) == 0:
                result['error'] = f'DataFrame vazio (key={key})'
                return result
            ohlc = ['Open', 'High', 'Low', 'Close']
            missing = [c for c in ohlc if c not in df.columns]
            if missing:
                result['error'] = f'Colunas OHLC ausentes: {missing}'
                return result
        except KeyError:
            result['error'] = f'Key "{key}" não encontrada no arquivo'
            return result
        except Exception as e:
            result['error'] = f'Erro ao ler HDF5: {e}'
            return result

    # Validar Parquet
    elif filepath.endswith('.parquet'):
        try:
            df = pd.read_parquet(filepath)
            result['n_rows'] = len(df)
            if len(df) == 0:
                result['error'] = 'Parquet vazio'
                return result
        except Exception as e:
            result['error'] = f'Erro ao ler Parquet: {e}'
            return result

    result['valid'] = True
    return result


def print_pipeline_status(stages: dict):
    """Imprime o status de todos os artefatos do pipeline."""
    print(f"\n{'=' * 70}")
    print(f"  STATUS DO PIPELINE")
    print(f"{'=' * 70}")

    for stage_id, stage in stages.items():
        output = stage['output_file']
        key = stage.get('output_key')
        min_kb = stage['min_size_kb']
        status = validate_artifact(output, key, min_kb)

        if status['valid']:
            rows_info = f" | {status['n_rows']:,} barras" if status['n_rows'] else ""
            icon = "[OK]"
            detail = f"{status['size_kb']:.1f} KB{rows_info}"
        elif status['exists']:
            icon = "[!!]"
            detail = status['error']
        else:
            icon = "[--]"
            detail = "Nao gerado"

        print(f"  {icon}  Estagio {stage_id}: {stage['name']}")
        print(f"       {output} -- {detail}")

    print(f"{'=' * 70}\n")


# =============================================================================
# EXECUÇÃO DE ESTÁGIOS
# =============================================================================

def run_stage(stage_id: int, stages: dict, pair: str, logger) -> bool:
    """
    Executa um estágio individual do pipeline.

    Returns
    -------
    bool : True se o estágio completou com sucesso.
    """
    stage = stages[stage_id]
    logger.info(f"INICIO Estagio {stage_id}: {stage['name']}")

    print(f"\n{'_' * 70}")
    print(f"  ESTAGIO {stage_id}/5: {stage['name']}")
    print(f"     {stage['description']}")
    print(f"{'_' * 70}")

    # Verificar dependência (input do estágio anterior)
    input_file = stage.get('input_file')
    if input_file and not os.path.exists(input_file):
        msg = f"DEPENDENCIA FALTANDO: {input_file}"
        print(f"  [ERRO] {msg}")
        logger.error(msg)
        return False

    model_dir = get_model_dir(pair)
    data_final = str(get_artifact_path('dataset_final', pair))
    data_raw = str(get_artifact_path('dataset_raw', pair))
    data_clean = str(get_artifact_path('dataset_clean', pair))
    model_path = str(get_artifact_path('moe_model', pair))
    config_path = str(get_artifact_path('moe_config', pair))
    results_path = str(get_artifact_path('results_inference', pair))
    plots_dir = str(PATHS['plots'])

    t0 = time.time()
    try:
        if stage_id == 1:
            from baixa_dados import backfill_ticks_to_bars
            backfill_ticks_to_bars(
                symbol=pair,
                days_back=90,
                bar_type='tick',
                ticks_per_bar=200,
                output_file=data_raw,
            )

        elif stage_id == 2:
            from limpaArquivos import run_denoising_pipeline
            run_denoising_pipeline(
                input_file=data_raw,
                output_file=data_clean,
            )

        elif stage_id == 3:
            from calcula_alphas import generate_alpha_features
            generate_alpha_features(
                input_file=data_clean,
                output_file=data_final,
            )

        elif stage_id == 4:
            from moe_to_daily import run_moe_pipeline
            from moe_gating import MoEConfig
            config = MoEConfig(
                input_file=data_final,
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
                model_save_path=model_path,
                results_save_path=results_path,
                config_save_path=config_path,
            )
            results = run_moe_pipeline(config=config)
            if results is None:
                raise RuntimeError("Pipeline MoE retornou None")

        elif stage_id == 5:
            from moe_visualization import (
                load_inference_results, load_config as load_viz_config,
                plot_moe_analysis, plot_ghost_projection,
            )
            results = load_inference_results(results_path)

            if os.path.exists(config_path):
                config_data = load_viz_config(config_path)
                if 'fold_metrics' in config_data:
                    results['fold_metrics'] = config_data['fold_metrics']

            # Carregar histórico
            df_history = None
            for filepath, key in [(data_final, 'features'), (data_clean, 'data')]:
                try:
                    df_history = pd.read_hdf(filepath, key=key)
                    break
                except (FileNotFoundError, KeyError):
                    continue

            if df_history is None:
                bp = results['base_price']
                df_history = pd.DataFrame({
                    'Open': [bp], 'High': [bp], 'Low': [bp], 'Close': [bp]
                })

            analysis_path = str(PATHS['plots'] / ARTIFACTS['plot_analysis'])
            ghost_path = str(PATHS['plots'] / ARTIFACTS['plot_ghost'])

            plot_moe_analysis(
                df_history=df_history, results=results,
                n_history=80, save_path=analysis_path,
            )
            plot_ghost_projection(
                df_history=df_history, results=results,
                n_history=60, save_path=ghost_path,
            )

    except Exception as e:
        elapsed = time.time() - t0
        msg = f"Estagio {stage_id} FALHOU apos {elapsed:.1f}s: {e}"
        print(f"\n  [ERRO] {msg}")
        logger.error(msg)
        import traceback
        traceback.print_exc()
        return False

    elapsed = time.time() - t0

    # Validar output
    output_file = stage['output_file']
    output_key = stage.get('output_key')
    min_kb = stage['min_size_kb']
    status = validate_artifact(output_file, output_key, min_kb)

    if status['valid']:
        rows_info = f" ({status['n_rows']:,} barras)" if status['n_rows'] else ""
        msg = f"Estagio {stage_id} COMPLETO em {elapsed:.1f}s | {output_file} -- {status['size_kb']:.1f} KB{rows_info}"
        print(f"\n  [OK] {msg}")
        logger.info(msg)
        return True
    else:
        msg = f"Estagio {stage_id} output invalido: {status['error']}"
        print(f"\n  [!!] {msg}")
        logger.warning(msg)
        return False


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Orquestrador do Pipeline MoE',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Cadeia de dados:
  1. baixa_dados.py     -> data/raw/dataset_raw.h5
  2. limpaArquivos.py   -> data/processed/dataset_clean.h5
  3. calcula_alphas.py  -> data/final/dataset_final.h5
  4. moe_to_daily.py    -> models/{PAIR}/ + exports/inference/
  5. moe_visualization.py -> exports/plots/
        """
    )

    parser.add_argument(
        '--from', type=int, choices=[1, 2, 3, 4, 5], default=1,
        dest='from_stage', help='Estagio inicial (default: 1)',
    )
    parser.add_argument(
        '--only', type=int, choices=[1, 2, 3, 4, 5], default=None,
        help='Executar apenas este estagio',
    )
    parser.add_argument(
        '--pair', type=str, default='USDCHF',
        help='Par de moedas (default: USDCHF)',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Verificar status dos artefatos sem executar',
    )

    args = parser.parse_args()
    pair = args.pair.upper()

    # Garantir estrutura de diretórios
    ensure_dirs()
    get_model_dir(pair)

    # Setup logger
    logger = setup_pipeline_logger()
    logger.info(f"Pipeline iniciado | Pair: {pair}")

    stages = _build_pipeline_stages(pair)

    print("=" * 70)
    print(f"  MoE PIPELINE ORCHESTRATOR -- {pair}")
    print("  Cadeia: Ticks -> Denoising -> Alphas -> MoE -> Visualizacao")
    print("=" * 70)

    # Dry-run: apenas mostrar status
    if args.dry_run:
        print_pipeline_status(stages)
        return

    # Determinar quais estágios executar
    if args.only is not None:
        stages_to_run = [args.only]
    else:
        stages_to_run = list(range(args.from_stage, 6))

    print(f"\n  Estagios a executar: {stages_to_run}")
    print_pipeline_status(stages)

    # Executar
    t_total = time.time()
    results = {}

    for stage_id in stages_to_run:
        success = run_stage(stage_id, stages, pair, logger)
        results[stage_id] = success
        if not success:
            print(f"\n  [STOP] Pipeline interrompido no estagio {stage_id}.")
            logger.error(f"Pipeline interrompido no estagio {stage_id}")
            break

    elapsed_total = time.time() - t_total

    # Resumo final
    print(f"\n{'=' * 70}")
    print(f"  RESUMO DA EXECUCAO (total: {elapsed_total:.1f}s)")
    print(f"{'=' * 70}")

    for stage_id, success in results.items():
        icon = "[OK]" if success else "[ERRO]"
        name = stages[stage_id]['name']
        print(f"  {icon}  Estagio {stage_id}: {name}")

    print_pipeline_status(stages)
    logger.info(f"Pipeline finalizado em {elapsed_total:.1f}s | Resultados: {results}")

    # Exit code
    all_ok = all(results.values())
    if all_ok:
        print("  Pipeline completo com sucesso!")
    else:
        print("  Pipeline finalizado com erros.")
        sys.exit(1)


if __name__ == "__main__":
    main()
