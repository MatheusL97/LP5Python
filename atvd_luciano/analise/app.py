from flask import Flask, request, render_template_string
import pickle
import pandas as pd

# Carregamento do modelo e colunas
with open('modelo_randomforest.pkl', 'rb') as f:
    modelo = pickle.load(f)
with open('colunas_modelo.pkl', 'rb') as f:
    colunas = pickle.load(f)

app = Flask(__name__)

# Carrega o form diretamente do HTML
with open('templates/form.html', encoding='utf-8') as f:
    form_html = f.read()

@app.route('/', methods=['GET'])
def form():
    return render_template_string(form_html)

@app.route('/', methods=['POST'])
def resultado():
    dados = {}
    for key, val in request.form.items():
        dados[key] = val if not val.isnumeric() else float(val)

    # Converter "Sim"/"Não" em binários 1/0
    bin_map = {"Sim":1, "Não":0, "Masculino":1, "Feminino":0}
    for k, v in dados.items():
        if v in bin_map:
            dados[k] = bin_map[v]

    df = pd.DataFrame([dados])
    df = pd.get_dummies(df)
    df = df.reindex(columns=colunas, fill_value=0)

    prob = modelo.predict_proba(df)[0][1]
    risco = modelo.predict(df)[0]

    if risco == 1:
        msg = f"⚠️ Risco de doença pulmonar crônica detectado (probabilidade: {prob:.2%})."
    else:
        msg = f"✅ Sem risco aparente. (probabilidade: {prob:.2%})."

    return f"<h2>{msg}</h2><a href='/'>Voltar</a>"

if __name__ == '__main__':
    app.run(debug=True)
