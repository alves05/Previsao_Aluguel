import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def base_dados():
    return pd.read_csv('./dataset_aluguel/base_aluguel.csv')


def base_treino_teste():
    with open('dataset_aluguel/base_treino_teste.pkl', 'rb') as f:
        X_treino, X_teste, y_treino, y_teste = pickle.load(f)
    return X_treino, X_teste, y_treino, y_teste


def pre_processamento(
    bairro: str,
    tipo: str,
    area: float,
    banheiros: int,
    suites: int,
    quartos: int,
    vaga: int,
    condominio: float,
    iptu: float,
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
        'iptu_ano': iptu,
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
        'iptu_ano',
    ]
    variavel_cat = ['bairro', 'tipo_imovel']

    num_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
        ]
    )
    cat_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, variavel_num),
            ('cat', cat_transformer, variavel_cat),
        ]
    )

    processed = preprocessor.fit_transform(dados)
    return processed[-1].reshape(1, -1)
