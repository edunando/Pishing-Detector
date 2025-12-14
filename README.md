# Pishing-Detector
Combate ao Phishing com Inteligência Artificial!

# Desafio Final: Sistema de Detecção de Phishing com IA
**Bootcamp:** Analista em Segurança da Informação e Cibersegurança IA Expert

**Projeto:** SecureMail Corp. Anti-Phishing System

## 1. Objetivo
Este projeto visa desenvolver um classificador de e-mails automatizado capaz de distinguir entre mensagens legítimas e ataques de phishing. A solução utiliza técnicas de Processamento de Linguagem Natural (NLP) e Machine Learning para analisar padrões textuais e proteger a empresa fictícia "SecureMail Corp." de ameaças cibernéticas.

## 2. Metodologia e Escolhas Técnicas

### 2.1. Pré-processamento de Dados
Para preparar os dados brutos para o modelo, realizei as seguintes etapas de limpeza e normalização:
* **Concatenação de Contexto:** Uni o *Assunto* (Subject) e o *Corpo* (Body) do e-mail, pois gatilhos de engenharia social frequentemente aparecem no assunto (ex: "Urgente", "Bloqueio").
* **Remoção de Stopwords:** Eliminei palavras comuns (artigos, preposições) que não agregam valor semântico para a detecção de fraudes.
* **Stemming (RSLP):** Reduzi as palavras às suas raízes (ex: "bloqueada" -> "bloq") para que o modelo entenda variações da mesma palavra como um único conceito.
* **Vetorização TF-IDF:** Utilizei o *Term Frequency-Inverse Document Frequency* para transformar texto em números. Diferente de uma contagem simples, o TF-IDF destaca palavras raras e específicas (como "senha", "clique") que são fortes indicativos no contexto de segurança, penalizando palavras genéricas.

### 2.2. Seleção do Modelo
Optei pelo algoritmo **Naive Bayes (MultinomialNB)**.
* **Justificativa:** É o padrão histórico da indústria para filtragem de spam devido à sua alta eficiência computacional e excelente desempenho com dados textuais de alta dimensão, mesmo com datasets menores.
