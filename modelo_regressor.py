from sklearn.ensemble import RandomForestRegressor
from preprocessamento import base_treino_teste


def regressor():
    modelo = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=1,
        min_samples_split=2,
        max_features='sqrt',
        bootstrap=False,
    )
    X_treino, _, y_treino, _ = base_treino_teste()
    modelo.fit(X_treino, y_treino)
    return modelo
