import re

def formatar_resposta(resposta, sql_usado):
    """
    Formata a resposta do agente, incluindo o SQL usado se disponível.
    """
    if sql_usado:
        return f"{resposta}\n\nSQL usado:\n```sql\n{sql_usado}\n```"
    return resposta

def extrair_sql_da_resposta(resposta):
    """
    Extrai o SQL de uma resposta formatada, se presente.
    """
    sql_match = re.search(r'```sql\n(.*?)\n```', resposta, re.DOTALL)
    if sql_match:
        return sql_match.group(1)
    return None

def limpar_texto(texto):
    """
    Remove caracteres especiais e formata o texto para uso em nomes de arquivos ou IDs.
    """
    return re.sub(r'[^a-zA-Z0-9_]', '_', texto).lower()

def truncar_texto(texto, max_length=100):
    """
    Trunca o texto para um comprimento máximo, adicionando reticências se necessário.
    """
    if len(texto) <= max_length:
        return texto
    return texto[:max_length-3] + "..."