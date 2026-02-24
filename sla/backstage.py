import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score
from datetime import timedelta

def calcular_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def backtest_anual_3_em_3(file_path):
    # 1. Carregamento
    df_base = pd.read_csv(file_path)
    df_base['Data'] = pd.to_datetime(df_base['Data'])
    df_base = df_base.sort_values('Data').reset_index(drop=True)

    # 2. Criação de Features (Aumentando o contexto para a bússola)
    df = df_base.copy()
    for i in range(1, 11):
        df[f'lag_{i}'] = df['Último'].shift(i)
    
    df['ma_7'] = df['Último'].rolling(window=7).mean()
    df['rsi'] = calcular_rsi(df['Último'])
    df['impulso'] = df['Último'].diff() 
    # Nova feature: Distância da média para sentir esticamento
    df['dist_media'] = df['Último'] - df['ma_7']
    
    df = df.dropna().reset_index(drop=True)

    features = [f'lag_{i}' for i in range(1, 11)] + ['ma_7', 'rsi', 'impulso', 'dist_media']

    # --- LÓGICA DE BACKTEST ---
    dias_totais = 70
    passo = 7
    indice_inicio = max(len(df) - dias_totais, 15)
    
    todas_previsoes = []
    todos_reais = []
    todas_datas = []
    todas_aberturas = []
    todas_maximas = []
    todas_minimas = []
    
    direcoes_corretas = []

    print(f"Iniciando Bússola de Tendência...")

    for i in range(indice_inicio, len(df) - passo, passo):
        df_train = df.iloc[:i].copy()
        df_test_real = df.iloc[i:i+passo].copy()
        
        last_price = df_train.iloc[-1]['Último']
        last_train_features = df_train.tail(1)[features]

        # Previsão da janela
        preds_janela = []
        for h in range(1, passo + 1):
            # Calculamos a variação futura que queremos prever
            # O alvo é: Preço daqui a 'h' dias menos o preço de 'hoje'
            y_trend = df_train['Último'].shift(-h) - df_train['Último']
            
            # Alinhamos o X com o y (removemos as últimas h linhas que ficaram sem alvo)
            X_trend = df_train[features].iloc[:-h]
            y_trend = y_trend.dropna()
            
            model = RandomForestRegressor(
                n_estimators=2000, # 1000 já é excelente para evitar ruído
                max_features='sqrt',
                min_samples_leaf=5, 
                random_state=900, 
                n_jobs=None
            )
            model.fit(X_trend, y_trend)
            
            # Prevemos a variação a partir do ÚLTIMO preço conhecido do treino
            pred_var = model.predict(last_train_features)[0]
            preds_janela.append(last_price + pred_var)

        # Lógica da Bússola: Acertou a direção?
        for j in range(len(df_test_real)):
            real_atual = df_test_real.iloc[j]['Último']
            pred_atual = preds_janela[j]
            
            # Se ambos subiram ou ambos caíram em relação ao preço inicial do teste
            direcao_real = 1 if real_atual > last_price else 0
            direcao_pred = 1 if pred_atual > last_price else 0
            direcoes_corretas.append(1 if direcao_real == direcao_pred else 0)

        todas_previsoes.extend(preds_janela)
        todos_reais.extend(df_test_real['Último'].tolist())
        todas_datas.extend(df_test_real['Data'].tolist())
        todas_aberturas.extend(df_test_real['Abertura'].tolist())
        todas_maximas.extend(df_test_real['Máxima'].tolist())
        todas_minimas.extend(df_test_real['Mínima'].tolist())
        print(f"Janela {i} a {i+passo} - Previsões: {[f'${p:.2f}' for p in preds_janela]} | Reais: {[f'${r:.2f}' for r in df_test_real['Último'].tolist()]}")
    # 5. Métricas da Bússola
    acuracia_bussola = (sum(direcoes_corretas) / len(direcoes_corretas)) * 100
    mae = mean_absolute_error(todos_reais, todas_previsoes)
    
    print(f"\n--- RELATÓRIO DA BÚSSOLA ---")
    print(f"Precisão Direcional: {acuracia_bussola:.2f}%")
    print(f"Erro Médio (MAE): ${mae:.2f}")

    # 6. Gráfico com a Bússola
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=todas_datas, open=todas_aberturas, high=todas_maximas,
        low=todas_minimas, close=todos_reais, name='Mercado Real'
    ))

    # Linha da IA suavizada (Bússola)
    fig.add_trace(go.Scatter(
        x=todas_datas, y=todas_previsoes,
        mode='lines', name='Bússola (Tendência IA)',
        line=dict(color='orange', width=3, shape='spline')
    ))

    fig.update_layout(
        title=f'Bússola de Ouro - Precisão Direcional: {acuracia_bussola:.2f}%',
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        yaxis_title='Preço USD'
    )
    
    fig.show()

# Executar
backtest_anual_3_em_3('EUR_GBP Historical Data_LIMPO.csv')