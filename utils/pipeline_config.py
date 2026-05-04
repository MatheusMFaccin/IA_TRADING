# -*- coding: utf-8 -*-
"""
===============================================================================
PIPELINE CONFIG — Configuração Centralizada de Caminhos e Logging
===============================================================================
Todas as constantes de diretório e caminhos de artefatos do pipeline MoE
são definidas aqui. Nenhum script deve usar caminhos hardcoded.

Uso:
    from pipeline_config import PATHS, get_model_dir, ensure_dirs
===============================================================================
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

# =============================================================================
# RAIZ DO PROJETO (diretório onde este arquivo reside)
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent

# =============================================================================
# MAPA DE DIRETÓRIOS
# =============================================================================
PATHS: Dict[str, Path] = {
    'root':            PROJECT_ROOT,
    'core':            PROJECT_ROOT / 'core',
    'data_raw':        PROJECT_ROOT / 'data' / 'raw',
    'data_processed':  PROJECT_ROOT / 'data' / 'processed',
    'data_final':      PROJECT_ROOT / 'data' / 'final',
    'models':          PROJECT_ROOT / 'models',
    'plots':           PROJECT_ROOT / 'exports' / 'plots',
    'inference':       PROJECT_ROOT / 'exports' / 'inference',
    'logs':            PROJECT_ROOT / 'logs',
}

# =============================================================================
# NOMES DE ARTEFATOS (constantes — sem caminho)
# =============================================================================
ARTIFACTS = {
    'dataset_raw':           'dataset_raw.h5',
    'dataset_clean':         'dataset_clean.h5',
    'dataset_final':         'dataset_final.h5',
    'moe_model':             'moe_model.keras',
    'moe_config':            'moe_config.json',
    'results_inference':     'results_inference.parquet',
    'plot_analysis':         'moe_analysis.html',
    'plot_ghost':            'moe_ghost_projection.html',
    'plot_daily':            'moe_daily_projection.html',
    'plot_daily_comparison': 'moe_daily_comparison.html',
    'plot_rolling':          'moe_rolling_30d.html',
    'plot_comparison':       'moe_comparison.html',
    'plot_denoising':        'denoising_comparison.png',
    'plot_projection':       'projecao_futura.html',
}


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def ensure_dirs() -> None:
    """Cria todos os diretórios do pipeline se não existirem."""
    for key, path in PATHS.items():
        if key != 'root':
            path.mkdir(parents=True, exist_ok=True)


def get_model_dir(pair: str = 'USDJPY') -> Path:
    """Retorna o diretório de modelos para um par de moedas."""
    d = PATHS['models'] / pair
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_artifact_path(artifact_key: str, pair: str = 'USDJPY') -> Path:
    """
    Resolve o caminho completo de um artefato pelo seu key.

    Mapeamento:
        dataset_raw        → data/raw/dataset_raw.h5
        dataset_clean      → data/processed/dataset_clean.h5
        dataset_final      → data/final/dataset_final.h5
        moe_model          → models/{pair}/moe_model.keras
        moe_config         → models/{pair}/moe_config.json
        results_inference  → exports/inference/results_inference.parquet
        plot_*             → exports/plots/{name}
    """
    name = ARTIFACTS[artifact_key]

    if artifact_key == 'dataset_raw':
        return PATHS['data_raw'] / name
    elif artifact_key == 'dataset_clean':
        return PATHS['data_processed'] / name
    elif artifact_key == 'dataset_final':
        return PATHS['data_final'] / name
    elif artifact_key in ('moe_model', 'moe_config'):
        return get_model_dir(pair) / name
    elif artifact_key == 'results_inference':
        return PATHS['inference'] / name
    elif artifact_key.startswith('plot_'):
        return PATHS['plots'] / name
    else:
        return PROJECT_ROOT / name


# =============================================================================
# LOGGING DO PIPELINE
# =============================================================================

_pipeline_logger: logging.Logger | None = None


def setup_pipeline_logger(name: str = 'MoE_Pipeline') -> logging.Logger:
    """
    Configura logger dual (console + arquivo) para o pipeline.

    Arquivo: logs/pipeline.log (append mode)
    Console: formatação com timestamps
    """
    global _pipeline_logger
    if _pipeline_logger is not None:
        return _pipeline_logger

    ensure_dirs()

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    log_file = PATHS['logs'] / 'pipeline.log'
    fh = logging.FileHandler(str(log_file), encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    _pipeline_logger = logger
    return logger
