from datetime import timedelta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import warnings

# Silenciar avisos de processamento paralelo
warnings.filterwarnings("ignore", category=UserWarning)

def calcular_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def prever_futuro_3_dias(file_path):
    # 1. Carregar e preparar todos os dados disponíveis
    df_base = pd.read_csv(file_path)
    df_base['Data'] = pd.to_datetime(df_base['Data'])
    df = df_base.sort_values('Data').reset_index(drop=True)

    # 2. Criar as mesmas Features do backtest
    for i in range(1, 11):
        df[f'lag_{i}'] = df['Último'].shift(i)
    df['ma_7'] = df['Último'].rolling(window=7).mean()
    df['rsi'] = calcular_rsi(df['Último'])
    df['impulso'] = df['Último'].diff() 
    df['std_7'] = df['Último'].rolling(window=7).std()
    df['dist_media'] = (df['Último'] - df['ma_7']) / (df['std_7'] + 1e-9)
    
    df_completo = df.dropna().reset_index(drop=True)
    features = [f'lag_{i}' for i in range(1, 11)] + ['ma_7', 'rsi', 'impulso', 'dist_media']

    # 3. Pegar os dados do último dia disponível (o "Hoje")
    last_row = df_completo.tail(1)
    current_price = last_row['Último'].values[0]
    last_date = last_row['Data'].values[0]
    current_features = last_row[features]

    print(f"\n--- PROJEÇÃO DA BÚSSOLA (PRÓXIMOS 3 DIAS) ---")
    print(f"Data Base: {pd.to_datetime(last_date).date()} | Preço Atual: {current_price:.5f}")

    proximos_precos = []
    
    # 4. Treinar modelos para H+1, H+2 e H+3 usando TODO o histórico
    for h in range(1, 4):
        # Alvo: Variação em H dias
        y_target = df_completo['Último'].shift(-h) - df_completo['Último']
        
        # Alinhamento
        X_train = df_completo[features].iloc[:-h]
        y_train = y_target.dropna()
        
        model = RandomForestRegressor(
            n_estimators=1200, 
            max_features='log2',
            min_samples_leaf=12,
            random_state=900, 
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Predição da variação futura
        pred_var = model.predict(current_features)[0]
        preco_futuro = current_price + pred_var
        proximos_precos.append(preco_futuro)
        
        data_futura = pd.to_datetime(last_date) + timedelta(days=h)
        direcao = "ALTA 🟢" if preco_futuro > current_price else "BAIXA 🔴"
        
        print(f"Dia {h} ({data_futura.date()}): {preco_futuro:.5f} | Tendência: {direcao}")

    return proximos_precos

def plotar_previsao_futura(file_path, proximos_precos):
    # 1. Carregar os últimos 20 dias para contexto visual
    df = pd.read_csv(file_path)
    df['Data'] = pd.to_datetime(df['Data'])
    df = df.sort_values('Data').tail(20) # Pegamos apenas o final para o gráfico não ficar gigante

    ultima_data = df['Data'].iloc[-1]
    ultimo_preco = df['Último'].iloc[-1]

    # 2. Criar as datas e os preços para a linha do futuro
    # A linha precisa começar do último preço real para não ficar "solta" no gráfico
    datas_futuras = [ultima_data] + [ultima_data + timedelta(days=i) for i in range(1, 4)]
    precos_futuros = [ultimo_preco] + proximos_precos

    # 3. Criar o Gráfico
    fig = go.Figure()

    # Adicionar Velas do Histórico Recente
    fig.add_trace(go.Candlestick(
        x=df['Data'],
        open=df['Abertura'],
        high=df['Máxima'],
        low=df['Mínima'],
        close=df['Último'],
        name='Histórico Real'
    ))

    # Adicionar Linha da Bússola (Futuro)
    fig.add_trace(go.Scatter(
        x=datas_futuras,
        y=precos_futuros,
        mode='lines+markers+text',
        name='Projeção IA (3 dias)',
        line=dict(color='orange', width=4, dash='dot'),
        text=[f"", f"D+1", f"D+2", f"D+3"],
        textposition="top center"
    ))

    # Configurações de Layout
    fig.update_layout(
        title=f'Bússola EUR/GBP: Projeção para os Próximos 3 Dias',
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        yaxis_title='Preço',
        hovermode='x unified'
    )

    fig.show()

# --- PARA EXECUTAR TUDO JUNTO ---
arquivo = 'EUR_GBP Historical Data_LIMPO.csv'
# 1. Calcula os preços futuros
previsoes = prever_futuro_3_dias(arquivo)
# 2. Plota o gráfico com a "ponte" para o futuro
plotar_previsao_futura(arquivo, previsoes)