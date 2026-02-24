import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from datetime import timedelta
import warnings

# Silenciar avisos
warnings.filterwarnings("ignore", category=UserWarning)

def calcular_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def backstage_eurusd(file_path):
    # 1. Carregamento e Preparação
    df_base = pd.read_csv(file_path)
    df_base['Data'] = pd.to_datetime(df_base['Data'])
    df_base = df_base.sort_values('Data').reset_index(drop=True)

    # 2. Criação de Features (Aumentando percepção de tendência)
    df = df_base.copy()
    for i in range(1, 11):
        df[f'lag_{i}'] = df['Último'].shift(i)
    
    df['ma_7'] = df['Último'].rolling(window=7).mean()
    df['ma_20'] = df['Último'].rolling(window=20).mean()
    df['rsi'] = calcular_rsi(df['Último'])
    df['roc_3'] = df['Último'].pct_change(periods=3)
    
    # Feature de Sentimento (Preço fechou perto da máxima ou mínima?)
    df['sentimento'] = (df['Último'] - df['Mínima']) / (df['Máxima'] - df['Mínima'] + 1e-9)
    
    df['std_20'] = df['Último'].rolling(window=20).std()
    df['dist_ma20'] = (df['Último'] - df['ma_20']) / (df['std_20'] + 1e-9)

    df = df.dropna().reset_index(drop=True)
    features = [f'lag_{i}' for i in range(1, 11)] + ['ma_7', 'ma_20', 'rsi', 'roc_3', 'dist_ma20', 'sentimento']

    # --- LÓGICA DE BACKTEST ---
    dias_totais = 80 
    passo = 3
    indice_inicio = max(len(df) - dias_totais, 25)
    
    todas_previsoes = []
    todos_reais = []
    todas_datas = []
    todas_aberturas = []
    todas_maximas = []
    todas_minimas = []
    direcoes_corretas = []

    print(f"Iniciando Backstage Bússola EUR/USD...")

    for i in range(indice_inicio, len(df) - passo, passo):
        df_train = df.iloc[:i].copy()
        df_test_real = df.iloc[i:i+passo].copy()
        
        last_price = df_train.iloc[-1]['Último']
        last_features = df_train.tail(1)[features]

        preds_janela = []
        for h in range(1, passo + 1):
            y_target = df_train['Último'].shift(-h) - df_train['Último']
            X_fit = df_train[features].iloc[:-h]
            y_fit = y_target.dropna()
            
            model = RandomForestRegressor(
                n_estimators=1000, # Reduzi um pouco para ser mais rápido e focar na média
                max_features='sqrt',
                min_samples_leaf=15, 
                random_state=900, 
                n_jobs=None
            )
            model.fit(X_fit, y_fit)
            
            pred_var = model.predict(last_features)[0]
            preds_janela.append(last_price + pred_var)
            print(f"Previsão para {h} dias à frente: {preds_janela[-1]:.5f} (Variação: {pred_var:.5f})")

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

    # 5. Relatórios e Veredito
    acuracia = (sum(direcoes_corretas) / len(direcoes_corretas)) * 100
    mae = mean_absolute_error(todos_reais, todas_previsoes)
    
    # Lógica de Veredito da Bússola
    ult_pred = todas_previsoes[-1]
    ult_real = todos_reais[-1]
    ult_rsi = df['rsi'].iloc[-1]
    
    print(f"\n" + "="*40)
    print(f"🧭 VEREDITO DA BÚSSOLA IA")
    print(f"="*40)
    
    diff_percent = ((ult_pred - ult_real) / ult_real) * 100
    
    if abs(diff_percent) < 0.05:
        print("TENDÊNCIA: Lateral / Indefinida ⚪")
        conselho = "O mercado está sem força clara. Evite entradas agressivas."
    elif ult_pred > ult_real:
        print(f"TENDÊNCIA: Alta Projetada 🟢 ({diff_percent:.2f}%)")
        conselho = "A bússola aponta para cima. Procure gatilhos de compra."
    else:
        print(f"TENDÊNCIA: Baixa Projetada 🔴 ({diff_percent:.2f}%)")
        conselho = "A bússola aponta para baixo. Procure gatilhos de venda."

    if ult_rsi > 70:
        conselho += " ATENÇÃO: RSI indica SOBRECOMPRA. Risco de correção."
    elif ult_rsi < 30:
        conselho += " ATENÇÃO: RSI indica SOBREVENDA. Risco de repique."

    print(f"CONSELHO: {conselho}")
    print(f"Precisão do Backstage: {acuracia:.2f}%")
    print(f"="*40)

    # 6. Gráfico
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=todas_datas, open=todas_aberturas, high=todas_maximas,
        low=todas_minimas, close=todos_reais, name='EUR/USD Real'
    ))
    fig.add_trace(go.Scatter(
        x=todas_datas, y=todas_previsoes,
        mode='lines', name='Bússola IA',
        line=dict(color='cyan', width=3, shape='spline', dash='dot')
    ))
    fig.update_layout(
        title=f'Backstage EUR/USD - Precisão Direcional: {acuracia:.2f}%',
        template='plotly_dark',
        xaxis_rangeslider_visible=False
    )
    fig.show()

# Executar
backstage_eurusd('EUR_USD Historical Data_LIMPO.csv')