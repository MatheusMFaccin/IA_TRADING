import pandas as pd

def limpar_dados_investing(arquivo_entrada, arquivo_saida='XAUUSD_LIMPO.csv'):
    # 1. Carregar o arquivo tratando as aspas e os separadores
    # O quotechar='"' remove as aspas duplas dos valores
    df = pd.read_csv(arquivo_entrada, quotechar='"')

    # 2. Renomear as colunas
    traducao = {
        'Date': 'Data',
        'Price': 'Último',
        'Open': 'Abertura',
        'High': 'Máxima',
        'Low': 'Mínima'
    }
    df = df.rename(columns=traducao)

    # 3. Filtrar colunas
    colunas_finais = ['Data', 'Último', 'Abertura', 'Máxima', 'Mínima']
    df = df[colunas_finais]

    # 4. Converter Data (Investig costuma usar MM/DD/YYYY ou DD/MM/YYYY)
    df['Data'] = pd.to_datetime(df['Data'])

    # 5. Limpeza de números (O PONTO DE INFLEXÃO)
    for col in ['Último', 'Abertura', 'Máxima', 'Mínima']:
        # Garante que é string antes de substituir
        df[col] = df[col].astype(str).str.replace(',', '', regex=False)
        # Converte para float
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 6. Check de segurança: ver quantas linhas sobraram
    linhas_antes = len(df)
    df = df.dropna()
    linhas_depois = len(df)
    
    if linhas_depois == 0:
        print("ERRO: O arquivo ficou vazio! Verifique o formato dos números.")
    else:
        print(f"Sucesso! Removidas {linhas_antes - linhas_depois} linhas com erro.")
        print(f"Total de registros salvos: {linhas_depois}")

    # 7. Salvar
    df.to_csv(arquivo_saida, index=False)
    print(df.head())

nome_arquivo = "USD_JPY Historical Data.csv"
limpar_dados_investing(nome_arquivo, arquivo_saida=nome_arquivo.replace('.csv', '_LIMPO.csv'))