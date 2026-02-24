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

def prever_futuro_7_dias_btc(file_path):
    # 1. Carregamento e Preparação Total
    df_base = pd.read_csv(file_path)
    df_base['Data'] = pd.to_datetime(df_base['Data'])
    df = df_base.sort_values('Data').reset_index(drop=True)

    # 2. Features Anti-Atraso
    df['retorno'] = np.log(df['Último'] / df['Último'].shift(1))
    df['rsi'] = calcular_rsi(df['Último'])
    df['rsi_delta'] = df['rsi'].diff()
    df['volatilidade'] = (df['Máxima'] - df['Mínima']).rolling(window=10).mean()
    df['z_volatilidade'] = (df['volatilidade'] - df['volatilidade'].rolling(20).mean()) / (df['volatilidade'].rolling(20).std() + 1e-9)
    df['momentum_fast'] = df['Último'].pct_change(periods=2) 
    
    for i in range(1, 6):
        df[f'retorno_lag_{i}'] = df['retorno'].shift(i)

    df_completo = df.dropna().reset_index(drop=True)
    features = ['retorno', 'rsi_delta', 'z_volatilidade', 'momentum_fast'] + [f'retorno_lag_{i}' for i in range(1, 6)]

    # Dados do "Hoje"
    last_row = df_completo.tail(1)
    current_price = last_row['Último'].values[0]
    last_date = last_row['Data'].values[0]
    current_features = last_row[features]

    print(f"\n" + "="*40)
    print(f"🚀 PROJEÇÃO BÚSSOLA BTC - PRÓXIMOS 7 DIAS")
    print(f"="*40)
    print(f"Preço Atual: ${current_price:.2f}")

    proximos_precos = []
    datas_futuras = []

    # 3. Treinar modelos para D+1 até D+7
    # Para o BTC, aumentamos o leaf para 12 em projeções longas para ser mais conservador
    for h in range(1, 8):
        y_target = np.log(df_completo['Último'].shift(-h) / df_completo['Último'])
        
        X_train = df_completo[features].iloc[:-h]
        y_train = y_target.dropna()
        
        model = RandomForestRegressor(
            n_estimators=1000, 
            max_features='sqrt',
            min_samples_leaf=12, 
            random_state=42, 
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        pred_retorno = model.predict(current_features)[0]
        preco_futuro = current_price * np.exp(pred_retorno)
        
        proximos_precos.append(preco_futuro)
        data_f = pd.to_datetime(last_date) + timedelta(days=h)
        datas_futuras.append(data_f)
        
        tendencia = "ALTA 🟢" if preco_futuro > current_price else "BAIXA 🔴"
        print(f"D+{h} ({data_f.date()}): ${preco_futuro:.2f} | {tendencia}")

    # 4. Gráfico de Produção
    df_plot = df_completo.tail(25) 
    datas_linha = [last_date] + datas_futuras
    precos_linha = [current_price] + proximos_precos

    fig = go.Figure()

    # Histórico
    fig.add_trace(go.Candlestick(
        x=df_plot['Data'], open=df_plot['Abertura'], high=df_plot['Máxima'],
        low=df_plot['Mínima'], close=df_plot['Último'], name='Histórico Real'
    ))

    # Projeção 7 Dias
    fig.add_trace(go.Scatter(
        x=datas_linha, y=precos_linha,
        mode='lines+markers',
        name='Bússola IA (7 Dias)',
        line=dict(color='orange', width=4, dash='dot'),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title='Projeção Bitcoin - Próximos 7 Dias (Lógica Anti-Atraso)',
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        yaxis_title='Preço USD',
        hovermode='x unified'
    )
    
    fig.show()

# Executar
prever_futuro_7_dias_btc('Bitcoin Historical Data_LIMPO.csv')