from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.callbacks import get_openai_callback
from database import listar_tabelas, obter_schema, executar_query
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime, timedelta

class AnalyticsEngine:
    def __init__(self, engine):
        self.engine = engine
    
    def gerar_grafico(self, df, tipo='linha'):
        """Gera gráfico baseado no DataFrame fornecido"""
        # Identificar coluna temporal
        data_cols = [col for col in df.columns if any(term in col.lower() 
                    for term in ['data', 'date', 'dt', 'período', 'periodo'])]
        if data_cols:
            x_col = data_cols[0]
            df[x_col] = pd.to_datetime(df[x_col])
        else:
            x_col = df.columns[0]
            
        # Identificar coluna numérica
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            y_col = [col for col in num_cols if 'quantidade' in col.lower() or 'qtd' in col.lower()]
            if y_col:
                y_col = y_col[0]
            else:
                y_col = num_cols[0]
        else:
            y_col = df.columns[1]
        
        # Agregar dados se necessário
        if pd.api.types.is_datetime64_any_dtype(df[x_col]):
            df = df.groupby(x_col)[y_col].sum().reset_index()
        
        # Criar gráfico
        if tipo == 'linha':
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode='lines+markers',
                name='Dados',
                line=dict(width=2, color='#2E86C1'),
                marker=dict(size=8)
            ))
        else:  # barra
            fig = go.Figure(data=[
                go.Bar(x=df[x_col], y=df[y_col], name='Dados', marker_color='#2E86C1')
            ])
        
        # Layout
        fig.update_layout(
            title=f'Análise de {y_col} por {x_col}',
            xaxis_title=x_col,
            yaxis_title=y_col,
            template='plotly_white',
            height=500,
            hovermode='x unified',
            showlegend=True
        )
        
        if pd.api.types.is_datetime64_any_dtype(df[x_col]):
            fig.update_xaxes(tickangle=45)
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    def fazer_previsao(self, df, periodos=12):
        """Realiza previsão usando Prophet"""
        # Identificar coluna temporal e numérica
        data_cols = [col for col in df.columns if any(term in col.lower() 
                    for term in ['data', 'date', 'dt', 'período', 'periodo'])]
        num_cols = [col for col in df.columns if any(term in col.lower() 
                   for term in ['quantidade', 'qtd', 'valor', 'total'])]
        
        if not data_cols or not num_cols:
            return None, "Não foi possível identificar colunas adequadas para previsão"
        
        date_col = data_cols[0]
        value_col = num_cols[0]
        
        # Preparar dados
        df_prophet = pd.DataFrame({
            'ds': pd.to_datetime(df[date_col]),
            'y': df[value_col].astype(float)
        })
        
        # Treinar modelo
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.95
        )
        model.fit(df_prophet)
        
        # Criar datas futuras para 2024
        future_dates = pd.date_range(
            start=df_prophet['ds'].max(),
            end='2024-12-31',
            freq='M'
        )
        future = pd.DataFrame({'ds': future_dates})
        
        # Fazer previsão
        forecast = model.predict(future)
        
        # Criar gráfico
        fig = go.Figure()
        
        # Dados históricos
        fig.add_trace(go.Scatter(
            x=df_prophet['ds'],
            y=df_prophet['y'],
            mode='lines+markers',
            name='Dados Históricos',
            line=dict(color='#2E86C1', width=2),
            marker=dict(size=8)
        ))
        
        # Previsão
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            mode='lines+markers',
            name='Previsão',
            line=dict(color='#E74C3C', width=2, dash='dash'),
            marker=dict(size=8)
        ))
        
        # Intervalo de confiança
        fig.add_trace(go.Scatter(
            x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
            y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(46, 134, 193, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Intervalo de Confiança 95%'
        ))
        
        # Layout
        fig.update_layout(
            title=f'Previsão de {value_col} para 2024',
            xaxis_title='Data',
            yaxis_title=value_col,
            template='plotly_white',
            height=500,
            hovermode='x unified'
        )
        
        # Obter valores previstos para retornar
        previsao_dict = {
            'valores_previstos': forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict('records'),
            'ultima_data_historica': df_prophet['ds'].max().strftime('%d/%m/%Y'),
            'media_historica': df_prophet['y'].mean(),
            'total_previsto_2024': forecast['yhat'].sum()
        }
        
        return previsao_dict, fig.to_html(full_html=False, include_plotlyjs='cdn')

def criar_agente(engine):
    """Cria o agente com capacidades analíticas"""
    db = SQLDatabase(engine)
    
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.3
    )
    
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )
    
    analytics = AnalyticsEngine(engine)
    
    return agent_executor, analytics

def fazer_pergunta(agente, engine, analytics, pergunta):
    """Processa a pergunta e retorna resposta com visualizações"""
    tabelas = listar_tabelas(engine)
    schemas = {table: obter_schema(engine, table) for table in tabelas}
    
    contexto = """Você é um Cientista de Dados Expert em IA. 
    RESPONDA SEMPRE EM PORTUGUÊS DO BRASIL de forma clara e objetiva.
    
    Para fazer previsões, você deve:
    1. Buscar os dados históricos ordenados por data
    2. Incluir quantidade e valor nas análises
    3. Usar GROUP BY para agregar os dados por data
    4. Ordenar os resultados por data
    
    Exemplo de SQL para previsão:
    SELECT 
        DATE_TRUNC('day', data_venda) as data,
        SUM(quantidade) as quantidade,
        SUM(valor) as valor
    FROM vendas
    WHERE material = '300000'
    GROUP BY DATE_TRUNC('day', data_venda)
    ORDER BY data
    """
    
    contexto += f"As tabelas disponíveis são: {', '.join(tabelas)}. "
    for table, schema in schemas.items():
        colunas = [f"{col['name']} ({col['type']})" for col in schema]
        contexto += f"A tabela {table} tem as seguintes colunas: {', '.join(colunas)}. "
    
    pergunta_completa = f"{contexto}\n\nPergunta do usuário: {pergunta}"
    
    try:
        with get_openai_callback() as cb:
            resposta = agente.run(pergunta_completa)
        
        sql_usado = extrair_sql_da_resposta(resposta)
        
        if sql_usado:
            df = pd.read_sql(sql_usado, engine)
            
            # Se for pedido de previsão
            if any(palavra in pergunta.lower() for palavra in ['previsão', 'previsao', 'prever', 'futuro', '2024']):
                try:
                    previsao_dict, grafico_previsao = analytics.fazer_previsao(df)
                    
                    if previsao_dict:
                        media_2024 = np.mean([v['yhat'] for v in previsao_dict['valores_previstos']])
                        total_2024 = previsao_dict['total_previsto_2024']
                        
                        resposta = f"""
                        Com base nos dados históricos até {previsao_dict['ultima_data_historica']}, 
                        realizei uma análise preditiva para 2024:

                        • Média mensal prevista: {media_2024:,.2f}
                        • Total previsto para 2024: {total_2024:,.2f}
                        
                        {resposta}
                        
                        Aqui está o gráfico com a previsão:
                        
                        {grafico_previsao}
                        
                        Legenda:
                        - Linha azul: dados históricos
                        - Linha vermelha tracejada: previsão
                        - Área sombreada: intervalo de confiança (95%)
                        """
                except Exception as e:
                    resposta += f"\n\nNão foi possível gerar a previsão: {str(e)}"
            
            # Se for pedido de gráfico
            elif any(palavra in pergunta.lower() for palavra in ['gráfico', 'grafico', 'visualizar', 'mostrar']):
                tipo = 'barra' if any(palavra in pergunta.lower() 
                                    for palavra in ['barra', 'coluna', 'colunas']) else 'linha'
                try:
                    grafico = analytics.gerar_grafico(df, tipo)
                    resposta = f"""
                    Análise dos dados solicitados:
                    
                    {resposta}
                    
                    Aqui está a visualização:
                    
                    {grafico}
                    """
                except Exception as e:
                    resposta += f"\n\nNão foi possível gerar o gráfico: {str(e)}"
        
        return resposta, sql_usado
    
    except Exception as e:
        return f"Erro ao processar a pergunta: {str(e)}", None

def extrair_sql_da_resposta(resposta):
    """Extrai a query SQL da resposta"""
    import re
    sql_match = re.search(r'```sql\n(.*?)\n```', resposta, re.DOTALL)
    if sql_match:
        return sql_match.group(1)
    return ""