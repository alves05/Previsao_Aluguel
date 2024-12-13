import pickle
from sklearn.ensemble import RandomForestRegressor


def base_treino_teste():
    with open('dataset_aluguel/base_treino_teste.pkl', 'rb') as f:
        X_treino, X_teste, y_treino, y_teste = pickle.load(f)
    return X_treino, X_teste, y_treino, y_teste    

def regressor():
        modelo = RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                min_samples_leaf=1,
                min_samples_split=2,
                max_features='sqrt',
                bootstrap=False
        )
        X_treino, _, y_treino, _ = base_treino_teste()
        modelo.fit(X_treino, y_treino)
        return modelo