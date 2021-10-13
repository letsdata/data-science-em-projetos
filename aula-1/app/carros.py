import joblib
import pandas as pd
import streamlit as st
from sklearn.preprocessing import OneHotEncoder


def buscar_categorias():
    carros = pd.read_parquet('./data/carros.parquet')

    marcas = list(carros['marca'].unique())
    tipos_vendedor = list(carros['tipo_vendedor'].unique())
    tipos_combustivel = list(carros['tipo_combustivel'].unique())
    cambios = list(carros['cambio'].unique())

    return marcas, tipos_vendedor, tipos_combustivel, cambios


def predicao_carros(caracteristicas):
    '''
    Função para realizar a predição de carros
    '''
    carros = pd.read_parquet('./data/carros.parquet')

    colunas_categoricas = carros.select_dtypes(include=[object]).columns

    df_caracteristicas = pd.DataFrame(
        columns=['idade_veiculo', 'kms_rodados', 'consumo', 'motor', 'potencia_maxima', 'assentos',
                 'preco_minimo_custo',
                 'preco_maximo_custo', 'marca', 'tipo_vendedor', 'tipo_combustivel', 'cambio'])

    df_caracteristicas.loc[0] = caracteristicas

    ordem_colunas = ['marca', 'preco_minimo_custo', 'preco_maximo_custo', 'idade_veiculo',
                     'kms_rodados', 'tipo_vendedor', 'tipo_combustivel', 'cambio', 'consumo',
                     'motor', 'potencia_maxima', 'assentos']

    df_caracteristicas = df_caracteristicas[ordem_colunas]

    colunas_categoricas = carros.select_dtypes(include=[object]).columns
    colunas_categoricas = [coluna for coluna in colunas_categoricas if coluna in df_caracteristicas.columns]

    ohe = OneHotEncoder(sparse=False, drop='first')

    for coluna_categorica in colunas_categoricas:
        ohe.fit(carros[[coluna_categorica]])
        colunas_ohe_teste = ohe.transform(df_caracteristicas[[coluna_categorica]])

        categorias_ohe = ohe.categories_[0][1:]

        for indice, nome_categoria in enumerate(categorias_ohe):
            df_caracteristicas[nome_categoria] = colunas_ohe_teste[:, indice]

        df_caracteristicas = df_caracteristicas.drop(coluna_categorica, axis=1)

    modelo = joblib.load('./models/linear_model.pkl')
    preco = modelo.predict(df_caracteristicas)

    return preco


# Criando a interface

cabecalho = st.container()
features = st.container()
resultado = st.container()

with cabecalho:
    st.image('./img/header.png', width=600)
    st.write('\n')
    st.title("Previsão de Preços de Carros")

with features:
    marcas, tipos_vendedor, tipos_combustivel, cambios = buscar_categorias()

    st.sidebar.title('Informe as características do carro')

    idade_veiculo = st.sidebar.slider("Idade do veículo", min_value=0, max_value=9)
    kms_rodados = st.sidebar.slider("KMs rodados (em milhares)", min_value=0., max_value=200.) * 1000
    consumo = st.sidebar.slider("Consumo", min_value=4., max_value=40.)
    motor = st.sidebar.slider("Motor (CC)", min_value=700., max_value=7000.)
    potencia = st.sidebar.slider("Potência", min_value=30., max_value=700.)
    assentos = st.sidebar.slider("Assentos", min_value=1., max_value=9.)

    preco_minimo = st.sidebar.slider("Preço mínimo de custo (em milhares)", min_value=300., max_value=1400.) * 1000
    preco_maximo = st.sidebar.slider("Preço máximo de custo (em milhares)", min_value=400., max_value=1900.) * 1000

    marca = st.sidebar.selectbox("Qual a marca do carro?", options=marcas)
    tipo_vendedor = st.sidebar.selectbox("Qual o tipo do vendedor?", options=tipos_vendedor)
    tipo_combustivel = st.sidebar.selectbox("Qual o tipo do combustível?", options=tipos_combustivel)
    cambio = st.sidebar.selectbox("Qual o câmbio?", options=cambios)

    if st.sidebar.button("Calcular preço"):
        caracteristicas = [idade_veiculo,
                           kms_rodados,
                           consumo,
                           motor,
                           potencia,
                           assentos,
                           preco_minimo,
                           preco_maximo,
                           marca,
                           tipo_vendedor,
                           tipo_combustivel,
                           cambio]

        print(caracteristicas)
        preco_predito = predicao_carros(caracteristicas)

        print(f"*** PRECO: {preco_predito}")

        with cabecalho:
            st.text(f"PRECO DO CARANGO: {preco_predito[0]:.2f} rupias! (quanto é isso em reais mesmo?)")
