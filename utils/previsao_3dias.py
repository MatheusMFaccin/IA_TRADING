import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta
import warnings

warnings.filterwarnings("ignore")

def calcular_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def prever_futuro_usdjpy(file_path):
    # 1. Carregamento e Preparação (Usando 100% dos dados para o treino final)
    df_base = pd.read_csv(file_path)
    df_base['Data'] = pd.to_datetime(df_base['Data'])
    df = df_base.sort_values('Data').reset_index(drop=True)

    # 2. Features de Antecipação
    df['retorno'] = np.log(df['Último'] / df['Último'].shift(1))
    df['rsi'] = calcular_rsi(df['Último'])
    df['rsi_delta'] = df['rsi'].diff()
    df['ma_20'] = df['Último'].rolling(window=20).mean()
    df['dist_ma20'] = (df['Último'] - df['ma_20']) / (df['ma_20'] + 1e-9)
    df['range'] = (df['Máxima'] - df['Mínima'])
    df['volatilidade'] = df['range'].rolling(window=10).mean()
    df['z_volatilidade'] = (df['volatilidade'] - df['volatilidade'].rolling(20).mean()) / (df['volatilidade'].rolling(20).std() + 1e-9)
    df['momentum_fast'] = df['Último'].pct_change(periods=2) 
    
    for i in range(1, 6):
        df[f'retorno_lag_{i}'] = df['retorno'].shift(i)

    df_treino = df.dropna().reset_index(drop=True)
    features = ['retorno', 'rsi_delta', 'dist_ma20', 'z_volatilidade', 'momentum_fast'] + [f'retorno_lag_{i}' for i in range(1, 6)]

    # Captura o estado atual para projetar
    last_row = df_treino.tail(1)
    current_price = last_row['Último'].values[0]
    last_date = last_row['Data'].values[0]
    X_current = last_row[features]

    print(f"\n" + "="*40)
    print(f"💴 PROJEÇÃO FUTURA USD/JPY (3 DIAS)")
    print(f"="*40)
    print(f"Preço de Fechamento Atual: {current_price:.3f} ¥")
    print("-" * 40)

    previsoes_futuras = []
    datas_futuras = []

    # 3. Gerar previsões para os próximos 3 passos
    for h in range(1, 4):
        # Alvo: Retorno logarítmico para o horizonte H
        y_target = np.log(df_treino['Último'].shift(-h) / df_treino['Último'])
        
        # Alinhamento
        X_fit = df_treino[features].iloc[:-h]
        y_fit = y_target.dropna()
        
        # Treino Robusto
        model = RandomForestRegressor(
            n_estimators=1200, 
            max_features='log2',
            min_samples_leaf=10, 
            random_state=42, 
            n_jobs=-1
        )
        model.fit(X_fit, y_fit)
        
        # Predição do Retorno -> Conversão para Preço
        pred_ret = model.predict(X_current)[0]
        preco_previsto = current_price * np.exp(pred_ret)
        
        data_prevista = last_date + timedelta(days=h)
        previsoes_futuras.append(preco_previsto)
        datas_futuras.append(data_prevista)
        
        direcao = "ALTA 🟢" if preco_previsto > current_price else "BAIXA 🔴"
        print(f"D+{h} ({data_prevista.date()}): {preco_previsto:.3f} ¥ | {direcao}")

    # 4. Gráfico de Contexto e Projeção
    df_hist = df_treino.tail(15) # Últimos 15 dias para contexto
    
    fig = go.Figure()
    # Histórico
    fig.add_trace(go.Scatter(x=df_hist['Data'], y=df_hist['Último'], name='Histórico Real', line=dict(color='cyan', width=2)))
    # Projeção (Ponte do último preço até as previsões)
    fig.add_trace(go.Scatter(
        x=[last_date] + datas_futuras, 
        y=[current_price] + previsoes_futuras, 
        name='Bússola IA (Projeção)', 
        line=dict(color='white', dash='dot', width=3),
        mode='lines+markers'
    ))

    fig.update_layout(
        title='Projeção USD/JPY - Próximos 3 Dias',
        template='plotly_dark',
        xaxis_title='Data',
        yaxis_title='Iene (JPY)'
    )
    fig.show()

# Executar
prever_futuro_usdjpy('USD_JPY Historical Data_LIMPO.csv')