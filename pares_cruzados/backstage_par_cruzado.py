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

def backtest_bussola_eurgbp(file_path):
    # 1. Carregamento e Preparação
    df_base = pd.read_csv(file_path)
    df_base['Data'] = pd.to_datetime(df_base['Data'])
    df_base = df_base.sort_values('Data').reset_index(drop=True)

    # 2. Criação de Features (Otimizadas para Reversão/EURGBP)
    df = df_base.copy()
    for i in range(1, 11):
        df[f'lag_{i}'] = df['Último'].shift(i)
    
    df['ma_7'] = df['Último'].rolling(window=7).mean()
    df['rsi'] = calcular_rsi(df['Último'])
    df['impulso'] = df['Último'].diff() 
    
    # Adicionando Volatilidade (Bandas de Bollinger simplificadas)
    df['std_7'] = df['Último'].rolling(window=7).std()
    df['dist_media'] = (df['Último'] - df['ma_7']) / (df['std_7'] + 1e-9)
    
    df = df.dropna().reset_index(drop=True)

    features = [f'lag_{i}' for i in range(1, 11)] + ['ma_7', 'rsi', 'impulso', 'dist_media']

    # --- LÓGICA DE BACKTEST ---
    dias_totais = 80 # Período de teste
    passo = 3        # Horizonte de previsão curto para EUR/GBP
    indice_inicio = max(len(df) - dias_totais, 20)
    
    todas_previsoes = []
    todos_reais = []
    todas_datas = []
    todas_aberturas = []
    todas_maximas = []
    todas_minimas = []
    direcoes_corretas = []

    print(f"Iniciando Bússola EUR/GBP (Passo: {passo} dias)...")

    for i in range(indice_inicio, len(df) - passo, passo):
        df_train = df.iloc[:i].copy()
        df_test_real = df.iloc[i:i+passo].copy()
        
        last_price = df_train.iloc[-1]['Último']
        last_features = df_train.tail(1)[features]

        preds_janela = []
        for h in range(1, passo + 1):
            # ALVO: Variação real do preço daqui a H dias
            y_target = df_train['Último'].shift(-h) - df_train['Último']
            
            # ALINHAMENTO: Removemos as últimas 'h' linhas para o X e y baterem
            X_fit = df_train[features].iloc[:-h]
            y_fit = y_target.dropna()
            
            model = RandomForestRegressor(
                n_estimators=1200, 
                max_features='log2',
                min_samples_leaf=12,
                random_state=900, 
                n_jobs=-1
            )
            model.fit(X_fit, y_fit)
            
            # Prever a variação e somar ao último preço
            pred_var = model.predict(last_features)[0]
            preds_janela.append(last_price + pred_var)

            print(f"Previsão para {h} dias: ${preds_janela[-1]:.5f} (Último: ${last_price:.5f})")

        # Avaliar acerto da bússola nesta janela
        for j in range(len(df_test_real)):
            real_val = df_test_real.iloc[j]['Último']
            pred_val = preds_janela[j]
            
            acertou = 1 if (real_val > last_price and pred_val > last_price) or \
                           (real_val < last_price and pred_val < last_price) else 0
            direcoes_corretas.append(acertou)

        todas_previsoes.extend(preds_janela)
        todos_reais.extend(df_test_real['Último'].tolist())
        todas_datas.extend(df_test_real['Data'].tolist())
        todas_aberturas.extend(df_test_real['Abertura'].tolist())
        todas_maximas.extend(df_test_real['Máxima'].tolist())
        todas_minimas.extend(df_test_real['Mínima'].tolist())

    # 5. Relatório Final
    acuracia = (sum(direcoes_corretas) / len(direcoes_corretas)) * 100
    mae = mean_absolute_error(todos_reais, todas_previsoes)
    
    print(f"\n--- RELATÓRIO DA BÚSSOLA EUR/GBP ---")
    print(f"Precisão Direcional: {acuracia:.2f}%")
    print(f"Erro Médio Absoluto: {mae:.6f}") # MAE pequeno é normal em Forex

    # 6. Gráfico Candlestick + Bússola IA
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=todas_datas, open=todas_aberturas, high=todas_maximas,
        low=todas_minimas, close=todos_reais, name='EUR/GBP Real'
    ))
    fig.add_trace(go.Scatter(
        x=todas_datas, y=todas_previsoes,
        mode='lines', name='Bússola IA',
        line=dict(color='orange', width=3, shape='spline')
    ))
    fig.update_layout(
        title=f'Bússola EUR/GBP - Acurácia: {acuracia:.2f}%',
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        yaxis_title='Preço'
    )
    fig.show()

# Executar (certifique-se de que o nome do arquivo está correto)
backtest_bussola_eurgbp('EUR_GBP Historical Data_LIMPO.csv')