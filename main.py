# @Pishing Detector


# Configuração e Importação de Bibliotecas 
import pandas as pd
import numpy as np
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Baixando recursos do NLTK 
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('rslp')

# Carregamento do Dataset
csv_data = """subject,body,sender,label
Atualize sua senha imediatamente,"Caro usuário, sua conta está em risco. Atualize sua senha clicando no link abaixo.",security@bank.com,1
Relatório semanal de vendas,Segue anexo o relatório semanal de vendas para sua análise.,sales@company.com,0
Oferta especial para você!,Aproveite esta oferta especial por tempo limitado! Não perca!,promo@shopping.com,1
Confirmação de compra,Obrigado pela sua compra. O seu pedido será enviado em breve.,orders@store.com,0
Sua conta foi bloqueada,"Sua conta foi bloqueada por questões de segurança. Por favor, entre em contato.",support@service.com,1
Reunião agendada para segunda-feira,Lembrete: nossa reunião está agendada para segunda-feira às 10h.,hr@company.com,0
"Parabéns, você ganhou um prêmio!",Você foi selecionado para receber um prêmio exclusivo! Clique para resgatar.,prizes@contest.com,1
Atualização importante da política de privacidade,Estamos atualizando nossa política de privacidade. Veja os detalhes.,privacy@company.com,0
Solicitação de reembolso,Recebemos sua solicitação de reembolso. O valor será creditado em breve.,support@store.com,0
Convite para evento exclusivo,Você está convidado para um evento exclusivo. Confirme sua presença.,events@company.com,0
"""

df = pd.read_csv(StringIO(csv_data))

print("--- Visualização Inicial dos Dados ---")
print(df.head())

# Pré-processamento de Dados (NLP)

# Função de limpeza de texto
stop_words = nltk.corpus.stopwords.words('portuguese')
stemmer = nltk.stem.RSLPStemmer()

def preprocess_text(text):
    try:
        # Tokenização
        tokens = nltk.word_tokenize(text.lower())
        # Remoção de stopwords e pontuação + Stemming
        tokens_processed = [stemmer.stem(t) for t in tokens if t.isalpha() and t not in stop_words]
        return " ".join(tokens_processed)
    except Exception as e:
        print(f"Erro ao processar: {text} | Erro: {e}")
        return ""

# Combinar Assunto e Corpo
df['text_full'] = df['subject'] + " " + df['body']

# Aplicar a limpeza
df['text_clean'] = df['text_full'].apply(preprocess_text)

# Vetorização (TF-IDF)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text_clean'])
y = df['label']

# Divisão Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Desenvolvimento do Modelo 
model = MultinomialNB()
model.fit(X_train, y_train)

# Previsões
y_pred = model.predict(X_test)

# Avaliação e Métricas 
print("\n--- Relatório de Performance ---")
print(f"Acurácia: {accuracy_score(y_test, y_pred):.2f}")
print(f"Precisão: {precision_score(y_test, y_pred, zero_division=0):.2f}")
print(f"Recall: {recall_score(y_test, y_pred, zero_division=0):.2f}")
print(f"F1-Score: {f1_score(y_test, y_pred, zero_division=0):.2f}")

print("\n--- Matriz de Confusão ---")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legítimo', 'Phishing'], yticklabels=['Legítimo', 'Phishing'])
plt.ylabel('Real')
plt.xlabel('Previsto')
plt.show()

# Teste prático
print("\n--- Teste Prático ---")
novos_emails = ["URGENTE: Sua senha expirou, clique aqui", "Relatório mensal anexo"]
novos_vetores = vectorizer.transform([preprocess_text(t) for t in novos_emails])
predicoes = model.predict(novos_vetores)
print(f"'{novos_emails[0]}' classificado como: {'Phishing' if predicoes[0]==1 else 'Legítimo'}")
print(f"'{novos_emails[1]}' classificado como: {'Phishing' if predicoes[1]==1 else 'Legítimo'}")
