import pandas as pd
from sqlalchemy import create_engine, inspect, text

def carregar_dados_do_postgres(connection_string):
    engine = create_engine(connection_string)
    return engine

def carregar_planilha(arquivo):
    if arquivo.name.endswith('.csv'):
        df = pd.read_csv(arquivo)
    elif arquivo.name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(arquivo)
    else:
        raise ValueError("Formato de arquivo não suportado")
    
    engine = create_engine('sqlite:///:memory:', echo=False)
    df.to_sql('dados', engine, index=False, if_exists='replace')
    return engine

def listar_tabelas(engine):
    inspector = inspect(engine)
    return inspector.get_table_names()

def obter_schema(engine, table_name):
    inspector = inspect(engine)
    return inspector.get_columns(table_name)

def executar_query(engine, query):
    with engine.connect() as conn:
        result = conn.execute(text(query))
        return pd.DataFrame(result.fetchall(), columns=result.keys())