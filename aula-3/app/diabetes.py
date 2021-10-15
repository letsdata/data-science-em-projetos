import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import streamlit as st
from sklearn.preprocessing import OneHotEncoder


def predicao_diabetes(caracteristicas):
    '''
    Função para realizar a predição de diabetes
    '''

    df_caracteristicas = pd.DataFrame(
        columns=['num_gravidez', 'glicose', 'pressao_sanguinea', 'espessura_pele',
       'insulina', 'imc', 'teste_tolerancia_glicose', 'idade'])

    df_caracteristicas.loc[0] = caracteristicas


    modelo = joblib.load('./models/classifier.pkl')
    scaler = joblib.load('./models/scaler.pkl')
    X = scaler.transform(df_caracteristicas)
    tem_diabetes = modelo.predict(X)

    return tem_diabetes


# Criando a interface

cabecalho = st.container()
features = st.container()

with cabecalho:
    st.image('./img/header.png', width=600)
    st.write('\n')
    st.title("Previsão de Diabetes")

with features:


    st.sidebar.title('Informe os dados clínicos do paciente')

    num_gravidez = st.sidebar.slider("Números de Gravidezes", min_value=0, max_value=17)
    glicose = st.sidebar.slider("Glicose", min_value=0., max_value=200.)
    pressao_sanguinea = st.sidebar.slider("Pressao", min_value=0., max_value=140.)
    espessura_pele = st.sidebar.slider("Espessura Pele", min_value=0., max_value=100.)
    insulina = st.sidebar.slider("Insulina", min_value=0., max_value=900.)
    imc = st.sidebar.slider("IMC", min_value=0., max_value=67.1)
    teste_tolerancia_glicose = st.sidebar.slider("Teste de tolerância a glicose", min_value=.05, max_value=3.)
    idade = st.sidebar.slider("Idade", min_value=15., max_value=80.)


    if st.sidebar.button("Prever condição de diabetes do paciente"):
        caracteristicas = [num_gravidez,
                           glicose,
                           pressao_sanguinea,
                           espessura_pele,
                           insulina,
                           imc,
                           teste_tolerancia_glicose,
                           idade]

        print(caracteristicas)
        predicao = predicao_diabetes(caracteristicas)

        print(f"*** TEM DIABETES? {predicao}")

        with cabecalho:
            texto = 'tem' if predicao == 1 else 'não tem'
            st.text(f"O PACIENTE {texto} diabetes! (lembrando que não podemos usar em produção!!)")
