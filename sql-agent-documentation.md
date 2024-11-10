# Documentação Técnica: Agente SQL com Interface Streamlit

## 1. Visão Geral do Projeto
Este projeto implementa um agente de consulta SQL com uma interface de usuário baseada em Streamlit. O sistema permite que os usuários façam perguntas em linguagem natural sobre dados armazenados em um banco de dados PostgreSQL ou em uma planilha, e recebam respostas geradas por um modelo de linguagem.

## 2. Arquitetura do Sistema
O sistema é composto por quatro componentes principais:
1. Agente SQL (`agent.py`)
2. Interface do Usuário (`app.py`)
3. Gerenciamento de Banco de Dados (`database.py`)
4. Utilitários (`utils.py`)

## 3. Componentes do Sistema

### 3.1 Agente SQL (`agent.py`)
Este componente é responsável pela criação e operação do agente SQL.

#### Principais Funções:
- `criar_agente(engine)`: Cria um executor de agente SQL.
- `fazer_pergunta(agente, engine, pergunta)`: Prepara e executa a consulta ao agente.
- `extrair_sql_da_resposta(resposta)`: Extrai o SQL da resposta do agente.

#### Detalhes Técnicos:
- Utiliza `langchain` para criar o agente SQL.
- Usa o modelo "gpt-4o-mini" com temperatura 0 para consistência nas respostas.
- Implementa `AgentType.ZERO_SHOT_REACT_DESCRIPTION` para respostas sem treinamento prévio.

### 3.2 Interface do Usuário (`app.py`)
Responsável pela interface do usuário usando Streamlit.

#### Principais Funcionalidades:
- Configuração da página e carregamento de CSS personalizado.
- Sidebar para seleção da fonte de dados (PostgreSQL ou planilha).
- Área de chat para interação com o agente SQL.

#### Detalhes Técnicos:
- Utiliza `streamlit` para a interface.
- Integra-se com `database.py` para carregar dados.
- Usa `PIL` para exibição de logo.

### 3.3 Gerenciamento de Banco de Dados (`database.py`)
Gerencia conexões e operações de banco de dados.

#### Principais Funções:
- `carregar_dados_do_postgres(connection_string)`: Conecta ao PostgreSQL.
- `carregar_planilha(arquivo)`: Cria um banco SQLite temporário a partir de uma planilha.
- `listar_tabelas(engine)` e `obter_schema(engine, table_name)`: Obtêm metadados do banco.

#### Detalhes Técnicos:
- Usa `SQLAlchemy` para conexões de banco de dados.
- Cria banco SQLite em memória para dados de planilhas.

### 3.4 Utilitários (`utils.py`)
Contém funções auxiliares para o projeto.

#### Principais Funções:
- `formatar_resposta(resposta, sql_usado)`: Formata a resposta incluindo o SQL.
- `extrair_sql_da_resposta(resposta)`: Extrai SQL da resposta usando regex.
- `limpar_texto(texto)` e `truncar_texto(texto, max_length)`: Manipulação de texto.

## 4. Fluxo de Trabalho
1. O usuário inicia a aplicação Streamlit.
2. Seleciona a fonte de dados na sidebar (PostgreSQL ou planilha).
3. A conexão com o banco de dados é estabelecida.
4. O usuário faz uma pergunta na interface de chat.
5. O agente SQL processa a pergunta e gera uma consulta SQL.
6. A consulta é executada no banco de dados.
7. O resultado é formatado e apresentado ao usuário.

## 5. Configuração e Dependências
- As dependências do projeto estão listadas em `requirements.txt`.
- Principais dependências: `streamlit`, `pandas`, `sqlalchemy`, `langchain`.
- O estilo visual é definido em `style.css`.

## 6. Considerações de Desenvolvimento
- O projeto utiliza um modelo de linguagem para interpretar perguntas em linguagem natural.
- A segurança das consultas SQL geradas deve ser considerada em um ambiente de produção.
- O desempenho pode variar dependendo do tamanho do banco de dados e da complexidade das consultas.

## 7. Expansão Futura
- Implementação de autenticação de usuários.
- Suporte para mais tipos de fontes de dados.
- Otimização de desempenho para grandes conjuntos de dados.
- Implementação de cache para consultas frequentes.
