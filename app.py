import pickle
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from modelo_regressor import regressor


@st.cache_data
def base_dados():
    try:
        return pd.read_csv('./dataset_aluguel/base_aluguel.csv')
    except FileNotFoundError:
        st.error("O arquivo 'base_aluguel.csv' não foi encontrado!")
        return pd.DataFrame()

def pre_processamento(
    bairro: str,
    tipo: str,
    area: float,
    banheiros: int,
    suites: int,
    quartos: int,
    vaga: int,
    condominio: float,
    iptu: float
):
    nova_linha = {
        'bairro': bairro,
        'tipo_imovel': tipo,
        'area_util': area,
        'banheiros': banheiros,
        'suites': suites,
        'quartos': quartos,
        'vaga_garagem': vaga,
        'taxa_condominio': condominio,
        'iptu_ano': iptu
    }
    nova_linha = pd.DataFrame(nova_linha, index=[0])
    dados = base_dados()
    dados = dados.iloc[:, :8]
    dados = pd.concat([dados, nova_linha], ignore_index=True)

    variavel_num = [
        'area_util',
        'banheiros',
        'suites',
        'quartos',
        'vaga_garagem',
        'taxa_condominio',
        'iptu_ano'
    ]
    variavel_cat = ['bairro', 'tipo_imovel']

    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, variavel_num),
            ('cat', cat_transformer, variavel_cat),
        ]
    )

    processed = preprocessor.fit_transform(dados)
    return processed[-1].reshape(1, -1)


def filtro_bairro_tipo(dados: pd.DataFrame):
    bairro = st.selectbox(
        'Escolha o Bairro:',
        dados['bairro'].unique(),
        index=None
    )
    tipo_imovel = st.selectbox(
        'Escolha o tipo de imóvel:', 
        dados['tipo_imovel'].unique(), 
        index=None
    )
    return bairro, tipo_imovel


def main():
    st.set_page_config(page_title='Previsão de Aluguéis', page_icon='🏘')
    st.markdown(
        "<h3 style='text-align:center; font-family:Verdana'>Previsão de Preço de Aluguéis Residenciais <br> na cidade de São Paulo <br> 🏘</h3>",
        unsafe_allow_html=True,
    )
    st.header('')
    st.sidebar.markdown(
        "<h4 style='text-align:center; font-family:Verdana'>Escolha a Configuração do Imóvel:</h4>",
        unsafe_allow_html=True,
    )

    dataset = base_dados()

    bairro, tipo_imovel = filtro_bairro_tipo(dataset)

    area_util = st.sidebar.slider('Área útil:', 0, 2000, 500)
    banheiros = st.sidebar.slider('Banheiros:', 0, 5, 2)
    suites = st.sidebar.slider('Suítes:', 0, 3, 1)
    quartos = st.sidebar.slider('Quartos:', 0, 4, 2)
    vaga_garagem = st.sidebar.slider('Vagas de Garagem:', 0, 4, 1)
    taxa_condominio = st.sidebar.number_input('Taxa de condomínio:', 0.00)
    iptu_ano = st.sidebar.number_input('IPTU do imóvel:', 0.00)

    if st.button('CONSULTAR'):
        if area_util <= 0:
            st.error('A área útil deve ser maior que zero!')
            return
        if taxa_condominio < 0:
            st.error('A taxa de condomínio não pode ser negativa!')
            return
        if iptu_ano < 0:
            st.error('O valor do IPTU não pode ser negativo!')
            return
        if bairro == None:
            st.error('Deve selecionar um bairro!')
            return
        if tipo_imovel == None:
            st.error('Deve selecionar um tipo de imóvel!')
            return

        processed = pre_processamento(
            bairro,
            tipo_imovel,
            area_util,
            banheiros,
            suites,
            quartos,
            vaga_garagem,
            taxa_condominio,
            iptu_ano
        )

        modelo = regressor()
        previsao = modelo.predict(processed)[0]
        valor_compra = f'R$ {previsao:.2f}'
        st.header(valor_compra)

        dados_filtros = {
            'Bairro': bairro,
            'Tipo de Imóvel': tipo_imovel,
            'Área Útil': area_util,
            'Banheiros': banheiros,
            'Suítes': suites,
            'Quartos': quartos,
            'Vagas de Garagem': vaga_garagem,
            'Taxa de Condomínio': taxa_condominio,
            'IPTU': iptu_ano
        }

        st.table(dados_filtros)

if __name__ == '__main__':
    main()
