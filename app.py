import streamlit as st
from database import carregar_dados_do_postgres, carregar_planilha, executar_query, listar_tabelas
from agent import criar_agente, fazer_pergunta
from utils import formatar_resposta
import os
from dotenv import load_dotenv
from PIL import Image
import plotly.express as px
import pandas as pd

# Carregar variáveis de ambiente
load_dotenv()

# Verificar chave da API
if not os.getenv("OPENAI_API_KEY"):
    st.error("Por favor configure a OPENAI_API_KEY nas variáveis de ambiente ou no arquivo .env")
    st.stop()

# Configuração da página
st.set_page_config(
    layout="wide",
    page_title="Agent AI - Text to SQL",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
    <style>
    .main-title {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .assistant-message {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
    .sql-used {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 0.5rem;
        font-family: monospace;
    }
    .gradient-text {
        background: linear-gradient(45deg, #2E86C1, #3498DB);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
""", unsafe_allow_html=True)

# Função para exibir mensagens no chat
def exibir_mensagem(role, content, sql_usado=None):
    if role == "Usuário":
        st.markdown(f"<div class='user-message'><strong>Usuário:</strong> {content}</div>", 
                   unsafe_allow_html=True)
    else:
        # Verificar se há HTML de gráfico na resposta
        if "<div>" in str(content):
            partes = str(content).split("<div>")
            st.markdown(f"<div class='assistant-message'><strong>Assistente:</strong> {partes[0]}</div>", 
                       unsafe_allow_html=True)
            for parte in partes[1:]:
                if "</div>" in parte:
                    grafico_html = f"<div>{parte.split('</div>')[0]}</div>"
                    st.components.v1.html(grafico_html, height=600, scrolling=True)
                    texto_restante = parte.split('</div>')[1]
                    if texto_restante.strip():
                        st.markdown(f"<div class='assistant-message'>{texto_restante}</div>", 
                                  unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='assistant-message'><strong>Assistente:</strong> {content}</div>", 
                       unsafe_allow_html=True)
        
        if sql_usado:
            with st.expander("Ver SQL utilizado"):
                st.code(sql_usado, language='sql')

# Inicialização do histórico de chat
if 'mensagens' not in st.session_state:
    st.session_state.mensagens = []
if 'dados_carregados' not in st.session_state:
    st.session_state.dados_carregados = False

# Área principal
col1, col2 = st.columns([1, 5])

with col1:
    try:
        # Carregar e exibir a logo
        logo = Image.open("fig.png")
        st.image(logo, width=100)
    except FileNotFoundError:
        st.write("📊")

with col2:
    st.markdown("<h1 class='main-title'>ADÃO - <span class='gradient-text'>Agent Data Analyst</span></h1>", 
                unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Transforme suas perguntas em consultas SQL poderosas</p>", 
                unsafe_allow_html=True)

# Barra lateral
st.sidebar.title("Configurações")

# Opção para escolher entre banco de dados ou planilha
opcao = st.sidebar.radio("Escolha a fonte de dados:", 
                        ("Banco de Dados PostgreSQL", "Carregar Planilha"))

if opcao == "Banco de Dados PostgreSQL":
    connection_string = st.sidebar.text_input(
        "String de conexão PostgreSQL:",
        type="password",
        help="Formato: postgresql://user:password@host:port/database"
    )
    if st.sidebar.button("Conectar"):
        try:
            with st.spinner("Conectando ao banco de dados..."):
                engine = carregar_dados_do_postgres(connection_string)
                st.session_state.engine = engine
                agente, analytics = criar_agente(engine)
                st.session_state.agente = agente
                st.session_state.analytics = analytics
                tabelas = listar_tabelas(engine)
                st.session_state.dados_carregados = True
                st.sidebar.success("✅ Conectado com sucesso!")
                st.sidebar.write("📋 Tabelas disponíveis:", tabelas)
        except Exception as e:
            st.sidebar.error(f"❌ Erro ao conectar: {str(e)}")
else:
    arquivo = st.sidebar.file_uploader(
        "Carregue sua planilha (CSV, XLS, XLSX)",
        type=['csv', 'xls', 'xlsx']
    )
    if arquivo is not None:
        try:
            with st.spinner("Carregando planilha..."):
                engine = carregar_planilha(arquivo)
                st.session_state.engine = engine
                agente, analytics = criar_agente(engine)
                st.session_state.agente = agente
                st.session_state.analytics = analytics
                tabelas = listar_tabelas(engine)
                st.session_state.dados_carregados = True
                st.sidebar.success("✅ Planilha carregada com sucesso!")
                st.sidebar.write("📋 Tabelas disponíveis:", tabelas)
                
                if st.sidebar.checkbox("👀 Visualizar dados"):
                    df = pd.read_sql(f"SELECT * FROM {tabelas[0]} LIMIT 5", engine)
                    st.sidebar.dataframe(df)
        except Exception as e:
            st.sidebar.error(f"❌ Erro ao carregar planilha: {str(e)}")

# Botão para limpar histórico
if st.sidebar.button("🗑️ Limpar Histórico"):
    st.session_state.mensagens = []
    st.experimental_rerun()

# Área de chat
st.subheader("Chat com IA")

# Exibir mensagens do chat
for mensagem in st.session_state.mensagens:
    exibir_mensagem(mensagem['role'], mensagem['content'], mensagem.get('sql_usado'))

# Área de entrada de pergunta
pergunta = st.text_input(
    "Faça uma pergunta sobre os dados:",
    placeholder="Ex: Mostre um gráfico de vendas ou faça uma previsão para 2024..."
)

# Botão de enviar
if st.button("Enviar"):
    if pergunta:
        if hasattr(st.session_state, 'agente') and hasattr(st.session_state, 'engine'):
            try:
                with st.spinner("Analisando sua pergunta..."):
                    resposta, sql_usado = fazer_pergunta(
                        st.session_state.agente,
                        st.session_state.engine,
                        st.session_state.analytics,
                        pergunta
                    )
                    st.session_state.mensagens.append({
                        "role": "Usuário",
                        "content": pergunta
                    })
                    st.session_state.mensagens.append({
                        "role": "Assistente",
                        "content": resposta,
                        "sql_usado": sql_usado
                    })
                    st.experimental_rerun()
            except Exception as e:
                st.error("❌ Erro ao processar pergunta!")
                with st.expander("Ver detalhes do erro"):
                    st.error(f"Tipo: {type(e).__name__}")
                    st.error(f"Mensagem: {str(e)}")
        else:
            st.error("⚠️ Por favor, conecte-se ao banco de dados ou carregue uma planilha primeiro.")

# Exemplos de perguntas
with st.sidebar.expander("💡 Exemplos de perguntas"):
    st.markdown("""
    **Análises básicas:**
    - Mostre um resumo dos dados
    - Qual a quantidade total vendida por mês?
    
    **Visualizações:**
    - Gere um gráfico de linha das vendas
    - Mostre um gráfico de barras por material
    
    **Previsões:**
    - Qual a previsão de vendas para 2024?
    - Faça uma previsão do material 300000
    - Como será a tendência para abril de 2024?
    """)

# Informações adicionais
with st.sidebar.expander("ℹ️ Sobre"):
    st.markdown("""
    **Funcionalidades:**
    - Análise de dados via SQL
    - Geração de gráficos
    - Previsões para 2024
    - Suporte a CSV e Excel
    - Conexão com PostgreSQL
    
    **Dicas:**
    - Use perguntas claras e específicas
    - Especifique o tipo de gráfico desejado
    - Para previsões, mencione o período
    """)

# Footer com versão
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style='text-align: center'>
        <small>Powered by Streamlit, LangChain e OpenAI</small>
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.text(f"Versão: {st.__version__}")