import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import geopandas as gpd
import numpy as np
from sklearn.linear_model import LinearRegression
import descartes


if __name__ == "__main__":
    pathname = os.path.abspath(os.path.dirname(__file__))

    data = {'Fecha': ['2020-03-03', '2020-03-04', '2020-03-05', '2020-03-06', '2020-03-07', '2020-03-08',
                      '2020-03-09', '2020-03-10', '2020-03-11', '2020-03-12', '2020-03-13', '2020-03-14',
                      '2020-03-15', '2020-03-16'],
            'Casos acumulados': [1, 3, 4, 5, 6, 10, 13, 17, 23, 33, 43, 61, 75, 156]}

    df = pd.DataFrame.from_dict(data)
    df['Fecha'] = pd.to_datetime(df['Fecha'], format='%Y-%m-%d')
    df['Detectados en el dia'] = df['Casos acumulados'].diff()
    df['log casos acumulados'] = np.log(df['Casos acumulados'])

    # Set up the matplotlib figure
    sns.set(style="white", palette="muted", color_codes=True)
    f, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
    sns.despine(left=True)
    f.suptitle("Evolución del coronavirus en Chile", fontsize=16)
    sns.lineplot(x='Fecha', y='Casos acumulados', data=df, ax=axes[0])
    sns.lineplot(x='Fecha', y='Detectados en el dia', data=df, ax=axes[1])
    sns.lineplot(x='Fecha', y='log casos acumulados', data=df, ax=axes[2])

    # Plotear mapa
    shp_path = '/raw/shapes/natural_earth_10m_chile_provinces_utm19sPolygon.shp'
    mapa = gpd.read_file(pathname + shp_path).to_crs({'init': 'epsg:4326'})
    mapa['infectados'] = np.array([0, 1, 0, 16, 1,
                                   123, 1, 1, 0, 1,
                                   2, 0, 0, 1, 9])

    # 0 Arica y Parinacota
    # 1 Los Lagos - X
    # 2 O 'Higgins
    # 3 BioBio - VIII
    # 4 Los Rios - XIV
    # 5 Region Metropolitana | RM
    # 6 La Araucana - IX
    # 7 Valparaiso
    # 8 Coquimbo
    # 9 Atacama
    # 10 Antofagasta
    # 11 Tarapaca
    # 12 Magallanes y Antartica Chilena
    # 13 Aisen | Aysen
    # 14 Maule - VII

    fig2, ax2 = plt.subplots(1, 1)
    mapa.plot(column='infectados', cmap='YlOrRd', edgecolor='black', ax=ax2, legend=True)

    # Regresion lineal
    df.reset_index(drop=False, inplace=True)
    X = np.array(df['index']).reshape(-1, 1)
    y = np.array(df['log casos acumulados']).reshape(-1, 1)
    reg = LinearRegression().fit(X, y)
    print(reg.score(X, y))
    print(reg.coef_)
    print(reg.intercept_)
    print(np.exp(reg.predict(np.array([[len(X) + 1]]))))


