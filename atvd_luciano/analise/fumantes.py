import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os
import pickle

def carregar_dados(caminho_csv):
    return pd.read_csv(caminho_csv)

def tratar_dados(df):
    colunas_para_remover = ['Id', 'Unnamed: 0'] if 'Unnamed: 0' in df.columns else ['Id']
    df.drop(columns=[col for col in colunas_para_remover if col in df.columns], inplace=True)

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    df = pd.get_dummies(df, drop_first=True)

    return df

def salvar_grafico(figura, caminho):
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    figura.savefig(caminho)
    plt.close(figura)

def explorar_dados(df_original):
    palette = {"Sim": "#FF6347", "Não": "#4682B4"}

    if "Status_Fumo" not in df_original.columns:
        print("Coluna 'Status_Fumo' não encontrada para visualização.")
        return

    fig1 = plt.figure()
    sns.scatterplot(data=df_original, x="Idade", y="IMC", hue="Status_Fumo", palette=palette)
    salvar_grafico(fig1, "saida/scatter_idade_imc.png")

    fig2 = plt.figure()
    sns.violinplot(data=df_original, x="Status_Fumo", y="Idade", hue="Status_Fumo", palette=palette, inner="quartile", legend=False)
    salvar_grafico(fig2, "saida/violin_idade_statusfumo.png")

def treinar_modelo(df):
    if "Doença_Pulmonar_Crônica_Sim" not in df.columns:
        raise ValueError("A coluna 'Doença_Pulmonar_Crônica_Sim' não foi encontrada após a codificação.")

    X = df.drop(columns=["Doença_Pulmonar_Crônica_Sim"])
    y = df["Doença_Pulmonar_Crônica_Sim"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

    model = RandomForestClassifier(random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("🔍 Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))

    print("\n📊 Relatório de Classificação:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Salvar modelo e colunas usadas
    with open('modelo_randomforest.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('colunas_modelo.pkl', 'wb') as f:
        pickle.dump(X.columns.tolist(), f)

def prever_risco(novo_dado_dict):
    with open('modelo_randomforest.pkl', 'rb') as f:
        modelo = pickle.load(f)
    with open('colunas_modelo.pkl', 'rb') as f:
        colunas = pickle.load(f)

    novo_df = pd.DataFrame([novo_dado_dict])
    novo_df = pd.get_dummies(novo_df)
    novo_df = novo_df.reindex(columns=colunas, fill_value=0)

    previsao = modelo.predict(novo_df)[0]
    probabilidade = modelo.predict_proba(novo_df)[0][1]

    if previsao == 1:
        print(f"⚠️ O paciente tem risco de doença pulmonar crônica (probabilidade: {probabilidade:.2%})")
    else:
        print(f"✅ Sem risco aparente de doença pulmonar crônica (probabilidade: {probabilidade:.2%})")

def main():
    caminho_csv = "dados/fumantes.csv"
    df_original = carregar_dados(caminho_csv)

    explorar_dados(df_original)

    df_processado = tratar_dados(df_original.copy())

    treinar_modelo(df_processado)

    # Exemplo de novo paciente
    novo_paciente = {
        "Idade": 60,
        "IMC": 28.5,
        "Cigarros_por_Dia": 15,
        "Anos_Fumando": 35,
        "Status_Fumo": "Atual",
        "Gênero": "Masculino",
        "Histórico_Familiar": "Sim",
        "Qualidade_da_Dieta": "Pobre",
        "Exposição_Fumaça_Secundária": "Alto",
        "Acesso_a_Saúde": "Pobre",
        "Nível_Poluição_do_Ar": "Alto",
        "Nível_de_Renda": "Baixa",
        "Nível_de_Escolaridade": "Primário",
        "Frequência_de_Rastreamento": "Nunca",
        "Exposição_Ocupacional": "Amianto",
        "Marcas_Geneticas_Positivas": 1,
        "Ano_Diagnóstico": 2022
    }

    prever_risco(novo_paciente)

if __name__ == "__main__":
    main()
