import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

class AdvancedAnalytics:
    def __init__(self, engine):
        self.engine = engine
        
    def generate_plot(self, df, plot_type, x_col, y_col, title=None):
        """
        Gera gráficos usando Plotly
        
        Args:
            df: DataFrame com os dados
            plot_type: 'line' ou 'bar'
            x_col: nome da coluna para eixo x
            y_col: nome da coluna para eixo y
            title: título do gráfico
        """
        if plot_type == 'line':
            fig = px.line(df, x=x_col, y=y_col, title=title)
        elif plot_type == 'bar':
            fig = px.bar(df, x=x_col, y=y_col, title=title)
        
        # Configuração do layout
        fig.update_layout(
            template='plotly_white',
            xaxis_title=x_col,
            yaxis_title=y_col,
            showlegend=True
        )
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    def forecast_future(self, df, date_col, value_col, periods=12):
        """
        Realiza previsão usando Prophet
        
        Args:
            df: DataFrame com os dados históricos
            date_col: nome da coluna de data
            value_col: nome da coluna de valor
            periods: número de períodos futuros para prever
        """
        # Preparar dados para o Prophet
        prophet_df = df.rename(columns={date_col: 'ds', value_col: 'y'})
        
        # Inicializar e treinar o modelo
        model = Prophet(yearly_seasonality=True, 
                       weekly_seasonality=True,
                       daily_seasonality=False)
        model.fit(prophet_df)
        
        # Criar DataFrame para previsão
        future = model.make_future_dataframe(periods=periods, freq='M')
        forecast = model.predict(future)
        
        # Preparar dados para visualização
        result_df = pd.DataFrame({
            'Data': forecast['ds'],
            'Previsão': forecast['yhat'],
            'Limite Inferior': forecast['yhat_lower'],
            'Limite Superior': forecast['yhat_upper']
        })
        
        # Gerar gráfico
        fig = go.Figure()
        
        # Dados históricos
        fig.add_trace(go.Scatter(
            x=prophet_df['ds'],
            y=prophet_df['y'],
            name='Dados Históricos',
            mode='lines'
        ))
        
        # Previsão
        fig.add_trace(go.Scatter(
            x=result_df['Data'],
            y=result_df['Previsão'],
            name='Previsão',
            mode='lines',
            line=dict(dash='dash')
        ))
        
        # Intervalo de confiança
        fig.add_trace(go.Scatter(
            x=result_df['Data'].tolist() + result_df['Data'].tolist()[::-1],
            y=result_df['Limite Superior'].tolist() + result_df['Limite Inferior'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Intervalo de Confiança'
        ))
        
        fig.update_layout(
            title='Previsão para 2024',
            xaxis_title='Data',
            yaxis_title='Valor',
            template='plotly_white',
            showlegend=True
        )
        
        return result_df, fig.to_html(full_html=False, include_plotlyjs='cdn')