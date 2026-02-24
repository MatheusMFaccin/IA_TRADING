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

def forecast_future_7_days_trend(file_path):
    # 1. Carregamento e Preparação Total
    df_base = pd.read_csv(file_path)
    df_base['Data'] = pd.to_datetime(df_base['Data'])
    df_base = df_base.sort_values('Data').reset_index(drop=True)

    # 2. Criação de Features
    df = df_base.copy()
    for i in range(1, 11):
        df[f'lag_{i}'] = df['Último'].shift(i)
    
    df['ma_7'] = df['Último'].rolling(window=7).mean()
    df['rsi'] = calcular_rsi(df['Último'])
    df['impulso'] = df['Último'].diff() 
    
    df = df.dropna().reset_index(drop=True)

    features = [f'lag_{i}' for i in range(1, 11)] + ['ma_7', 'rsi', 'impulso']

    # 3. Treinamento com 100% dos dados disponíveis
    # Usamos o último preço conhecido como base para as projeções futuras
    last_real_price = df.iloc[-1]['Último']
    last_known_features = df.tail(1)[features]
    last_date = df.iloc[-1]['Data']
    
    predictions_final_price = []
    future_dates = []
    
    print("Iniciando Projeção Real: Treinando 7 IAs para o Futuro...")

    # 4. Treinamento de 7 IAs Independentes para cada horizonte futuro
    for h in range(1, 8):
        # Alvo: Quanto o preço vai variar daqui a 'h' dias em relação a hoje
        y_trend = df['Último'].shift(-h) - df['Último']
        y_trend = y_trend.dropna()
        X_trend = df[features].iloc[:len(y_trend)]
        
        # n_jobs=-1 para usar toda a potência da CPU
        model = RandomForestRegressor(
            n_estimators=2800,           # Aumentamos um pouco para dar estabilidade
            max_features='sqrt',        # Melhora a diversidade entre as árvores
            min_samples_leaf=5,         # Evita overfitting em ruídos curtos
            random_state=900, 
            n_jobs=-1
        )
        model.fit(X_trend, y_trend)
        
        # Prevê a variação futura
        pred_variation = model.predict(last_known_features)[0]
        
        # Constrói o preço futuro
        predictions_final_price.append(last_real_price + pred_variation)
        future_dates.append(last_date + timedelta(days=h))
        
        print(f"IA Futuro Dia {h}/7 concluída.")

    # 5. Gráfico Plotly de Projeção
    fig = go.Figure()

    # Histórico Recente (últimos 20 dias para contexto visual)
    hist_view = df.tail(20)
    fig.add_trace(go.Candlestick(
        x=hist_view['Data'],
        open=hist_view['Abertura'],
        high=hist_view['Máxima'],
        low=hist_view['Mínima'],
        close=hist_view['Último'],
        name='Histórico Real'
    ))

    # Linha de Projeção Futura (Partindo do último fechamento)
    # Adicionamos o último ponto real para a linha conectar perfeitamente
    proj_dates = [last_date] + future_dates
    proj_values = [last_real_price] + predictions_final_price

    fig.add_trace(go.Scatter(
        x=proj_dates,
        y=proj_values,
        mode='lines+markers',
        name='Projeção 7 Dias (Tendência)',
        line=dict(color='lime', width=4, dash='dot'),
        marker=dict(size=8, symbol='diamond')
    ))

    # 6. Estilização do Painel
    fig.update_layout(
        title='Projeção de Tendência Real: Próximos 7 Dias',
        xaxis_title='Data',
        yaxis_title='Preço',
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    fig.show()

# Executar Projeção
forecast_future_7_days_trend('XAU_USD Historical Data_LIMPO.csv')