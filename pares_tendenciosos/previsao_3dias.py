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

def prever_e_plotar_eurusd(file_path):
    # 1. Carregamento
    df = pd.read_csv(file_path)
    df['Data'] = pd.to_datetime(df['Data'])
    df = df.sort_values('Data').reset_index(drop=True)

    # 2. Features para EUR/USD
    for i in range(1, 11):
        df[f'lag_{i}'] = df['Último'].shift(i)
    
    df['ma_7'] = df['Último'].rolling(window=7).mean()
    df['ma_20'] = df['Último'].rolling(window=20).mean()
    df['rsi'] = calcular_rsi(df['Último'])
    df['roc_3'] = df['Último'].pct_change(periods=3)
    df['std_20'] = df['Último'].rolling(window=20).std()
    df['dist_ma20'] = (df['Último'] - df['ma_20']) / (df['std_20'] + 1e-9)
    
    df = df.dropna().reset_index(drop=True)
    features = [f'lag_{i}' for i in range(1, 11)] + ['ma_7', 'ma_20', 'rsi', 'roc_3', 'dist_ma20']

    # 3. Treino para o Futuro
    last_row = df.tail(1)
    current_price = last_row['Último'].values[0]
    current_date = last_row['Data'].values[0]
    
    previsoes = []
    for h in range(1, 4):
        y = df['Último'].shift(-h) - df['Último']
        X_train = df[features].iloc[:-h]
        y_train = y.dropna()
        
        model = RandomForestRegressor(n_estimators=2000, min_samples_leaf=20, random_state=900, n_jobs=-1)
        model.fit(X_train, y_train)
        
        pred_var = model.predict(last_row[features])[0]
        previsoes.append(current_price + pred_var)

    # 4. Plotagem
    df_plot = df.tail(20)
    datas_futuras = [current_date] + [current_date + timedelta(days=i) for i in range(1, 4)]
    precos_futuros = [current_price] + previsoes

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df_plot['Data'], open=df_plot['Abertura'], high=df_plot['Máxima'], low=df_plot['Mínima'], close=df_plot['Último'], name='Histórico'))
    fig.add_trace(go.Scatter(x=datas_futuras, y=precos_futuros, mode='lines+markers', name='Bússola EUR/USD', line=dict(color='cyan', width=4, dash='dot')))
    
    fig.update_layout(title=f'Bússola EUR/USD - Projeção 3 Dias', template='plotly_dark', xaxis_rangeslider_visible=False)
    fig.show()
    
    print(f"Previsão D+3 para EUR/USD: {previsoes[-1]:.5f}")

# Executar
prever_e_plotar_eurusd('EUR_USD Historical Data_LIMPO.csv')