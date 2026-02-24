import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta

def calcular_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def prever_proximos_dias(file_path, dias_previsao=3):
    # 1. Carregar dados atuais
    df_base = pd.read_csv(file_path)
    df_base['Data'] = pd.to_datetime(df_base['Data'])
    df = df_base.sort_values('Data').reset_index(drop=True)

    # 2. Criar Features (Igual ao treino)
    for i in range(1, 11):
        df[f'lag_{i}'] = df['Último'].shift(i)
    df['ma_7'] = df['Último'].rolling(window=7).mean()
    df['rsi'] = calcular_rsi(df['Último'])
    df['impulso'] = df['Último'].diff() 
    df['dist_media'] = df['Último'] - df['ma_7']
    
    df = df.dropna()
    features = [f'lag_{i}' for i in range(1, 11)] + ['ma_7', 'rsi', 'impulso', 'dist_media']

    # 3. Preparar dados de hoje
    ultimo_registro = df.iloc[-1]
    preco_atual = ultimo_registro['Último']
    ultima_data = ultimo_registro['Data']
    features_atuais = df.tail(1)[features]

    print(f"\n--- BÚSSOLA EM TEMPO REAL ---")
    print(f"Preço Atual: ${preco_atual:.2f} ({ultima_data.date()})")
    print(f"Projetando próximos {dias_previsao} dias...")

    previsoes_futuras = []
    datas_futuras = []

    # 4. Treinar e Prever
    for h in range(1, dias_previsao + 1):
        # Treina com todo o histórico para o horizonte H
        y = df['Último'].shift(-h) - df['Último']
        y = y.dropna()
        X = df[features].iloc[:len(y)]
        
        model = RandomForestRegressor(
            n_estimators=1000, 
            min_samples_leaf=10, 
            random_state=42, 
            n_jobs=-1
        )
        model.fit(X, y)
        
        # Faz a predição para o futuro
        variacao_predita = model.predict(features_atuais)[0]
        preco_predito = preco_atual + variacao_predita
        
        previsoes_futuras.append(preco_predito)
        datas_futuras.append(ultima_data + timedelta(days=h))

    # 5. Exibir Guia da Bússola
    print("\nPROJEÇÃO:")
    for d, p in zip(datas_futuras, previsoes_futuras):
        direcao = "SUBIDA 🟢" if p > preco_atual else "QUEDA 🔴"
        print(f"Data: {d.date()} | Alvo: ${p:.2f} | Direção: {direcao}")

    return datas_futuras, previsoes_futuras

# --- EXECUÇÃO ---
arquivo = 'XAU_USD Historical Data_LIMPO.csv'

# Primeiro roda o backtest para você ver a confiança
#backtest_anual_3_em_3(arquivo)

# Depois gera a previsão para o futuro (amanhã, depois e depois)
datas_f, preps_f = prever_proximos_dias(arquivo, dias_previsao=3)