import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from datetime import timedelta
import warnings

warnings.filterwarnings("ignore")

def calcular_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def backstage_btc_anti_atraso(file_path):
    # 1. Carregamento
    df_base = pd.read_csv(file_path)
    df_base['Data'] = pd.to_datetime(df_base['Data'])
    df = df_base.sort_values('Data').reset_index(drop=True)

    # 2. Features de Antecipação (Foco em Mudança, não em Valor)
    # Retorno Logarítmico: Captura a velocidade da queda antes do valor nominal
    df['retorno'] = np.log(df['Último'] / df['Último'].shift(1))
    
    # Delta RSI: Se o RSI cai enquanto o preço sobe, a IA detecta exaustão
    df['rsi'] = calcular_rsi(df['Último'])
    df['rsi_delta'] = df['rsi'].diff()
    
    # Volatilidade Z-Score: Mede picos de pânico ou euforia
    df['volatilidade'] = (df['Máxima'] - df['Mínima']).rolling(window=10).mean()
    df['z_volatilidade'] = (df['volatilidade'] - df['volatilidade'].rolling(20).mean()) / (df['volatilidade'].rolling(20).std() + 1e-9)

    # Momentum de Curto Prazo (Antecipação rápida)
    df['momentum_fast'] = df['Último'].pct_change(periods=2) 
    
    # Lags de Retorno (Em vez de lags de preço, usamos lags de variação)
    for i in range(1, 6):
        df[f'retorno_lag_{i}'] = df['retorno'].shift(i)

    df = df.dropna().reset_index(drop=True)

    # Features que não "atrasam"
    features = ['retorno', 'rsi_delta', 'z_volatilidade', 'momentum_fast'] + [f'retorno_lag_{i}' for i in range(1, 6)]

    # --- LÓGICA DE BACKTEST ---
    dias_totais = 70
    passo = 3
    indice_inicio = len(df) - dias_totais
    
    todas_previsoes, todos_reais, todas_datas = [], [], []
    direcoes_corretas = []

    print(f"Iniciando Bússola BTC (Lógica Anti-Atraso)...")

    for i in range(indice_inicio, len(df) - passo, passo):
        df_train = df.iloc[:i]
        df_test = df.iloc[i:i+passo]
        last_price = df_train.iloc[-1]['Último']
        
        for h in range(1, passo + 1):
            # O ALVO agora é o RETORNO acumulado, não o preço fixo
            # Isso faz a IA prever a "porcentagem" de subida ou queda
            y_target = np.log(df_train['Último'].shift(-h) / df_train['Último'])
            
            X_fit = df_train[features].iloc[:-h]
            y_fit = y_target.dropna()
            
            model = RandomForestRegressor(
                n_estimators=1000, 
                max_features='sqrt', 
                min_samples_leaf=8, # Um pouco mais sensível para reagir rápido
                random_state=42, 
                n_jobs=-1
            )
            model.fit(X_fit, y_fit)
            
            # Prever a variação percentual
            pred_retorno = model.predict(df_train.tail(1)[features])[0]
            
            # Converte de volta para preço: Preço * e^(retorno)
            preds_h = last_price * np.exp(pred_retorno)
            
            todas_previsoes.append(preds_h)
            real_val = df_test.iloc[h-1]['Último']
            todos_reais.append(real_val)
            todas_datas.append(df_test.iloc[h-1]['Data'])

            # Direção
            acertou = 1 if (real_val > last_price and preds_h > last_price) or \
                           (real_val < last_price and preds_h < last_price) else 0
            direcoes_corretas.append(acertou)
        print(f"Janela {i} a {i+passo}: Previsão H+{h} = {preds_h:.2f} | Real = {real_val:.2f} | {'ACERTOU' if acertou else 'ERROU'}")
    # Relatório Final
    acuracia = (sum(direcoes_corretas) / len(direcoes_corretas)) * 100
    ult_fechamento = df['Último'].iloc[-1]
    projecao_final = todas_previsoes[-1]

    print("\n" + "="*40)
    print("🧭 ESTRATÉGIA ANTI-ATRASO BTC")
    print("="*40)
    print(f"Precisão Direcional: {acuracia:.2f}%")
    print(f"Preço Atual: ${ult_fechamento:.2f}")
    print(f"Projeção IA: ${projecao_final:.2f}")
    
    direcao_ia = "ALTA 🟢" if projecao_final > ult_fechamento else "BAIXA 🔴"
    print(f"SINAL IA: {direcao_ia}")
    print("="*40)

    # Gráfico
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=todas_datas, y=todos_reais, name='BTC Real', line=dict(color='gold')))
    fig.add_trace(go.Scatter(x=todas_datas, y=todas_previsoes, name='Bússola (Anti-Lag)', line=dict(color='orange', dash='dot')))
    fig.update_layout(title=f'Bússola BTC Anti-Atraso - Precisão: {acuracia:.2f}%', template='plotly_dark')
    fig.show()

# Executar
backstage_btc_anti_atraso('Bitcoin Historical Data_LIMPO.csv')