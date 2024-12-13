<h1 style='text-align: center'>Previsão de Preço de Aluguéis Residenciais <br> na cidade de São Paulo <br>🏘</h1>

### Equipe:
- Danilo Pontes;
- Jonatha Silva;
- Marcos Antônio;
- Sanderson Rawan;
- Saulo Bernardo;
- Wiliams Alves.

### Objetivo

O objetivo deste projeto é desenvolver um modelo de machine learning de regressão capaz de prever com uma precisão de 75% os preços de aluguéis dos imóveis residenciais da cidade de São Paulo. O escopo do projeto abrange a implementação desse modelo em um aplicativo de dados, proporcionando aos usuários uma ferramenta prática para estimar o valor do aluguel com base em diferentes características imobiliárias.

### Sobre a Base de Dados Usada no Projeto

A base de dados usada foi o dataset [housing_sp_city](https://www.kaggle.com/datasets/ex0ticone/house-prices-of-sao-paulo-city/data) que está disponível na plataforma kaggle.com. Os dados são referentes a preços de anúncios de imóveis na cidade de São Paulo entre os anos de 2011 a 2019.

### Cenário
Uma imobiliária que opera em São Paulo deseja otimizar a precificação de aluguéis para seus clientes. Atualmente, a empresa enfrenta desafios na definição de preços competitivos e justos para os diferentes imóveis que gerencia na cidade. A falta de um sistema de precificação eficiente resulta em inconsistências nos valores de aluguel, levando a perdas financeiras e insatisfação entre proprietários e inquilinos.

### Problema de Negócio
As dificuldades em estabelecer preços justos de aluguéis que reflitam de maneira precisa as características únicas de cada imóvel e as condições específicas do mercado em cada cidade. A abordagem atual baseia-se em avaliações subjetivas, o que leva a variações nos preços que não são necessariamente justificadas por fatores objetivos. Isso resulta em imóveis que podem estar subvalorizados, prejudicando a rentabilidade da imobiliária, ou superavaliados, o que pode afetar a atração de inquilinos.

### Proposta de Solução
Implementação de um modelo de machine learning de regressão em um data app, para prever preços de aluguéis visando resolver o problema de negócio enfrentado pela imobiliária, proporcionando uma abordagem mais eficiente e precisa na precificação de seus imóveis com um modelo que preveja os preços com no mínimo 75% de assertividade.

### Objetivos Alcançados 
- Algoritmo usado: RandomForestRegressor;
- Modelo com assertividade de 87,63% acima da meta estabelecida de 75%;
- Criação e implementação do modelo preditivo no data app.

### Executando o Data App
- Clicando no box abaixo será direcionado para o aplicativo de previsão de preços de alugueis.

[![App Previsão de Aluguel](https://img.shields.io/badge/App--Previsão--de--Preço--de--Alugueis-v1.0-fff?style=for-the-badge&labelColor=blue)]()

### Conclusão

Com base nos resultados obtidos utilizando o modelo `RandomForestRegressor`, podemos concluir que o modelo desenvolvido apresentou um bom desempenho, ficando acima dos resultados esperados, apresentando um escore de aproximadamente `0.8472` nos dados de treino e aproximadamente `0.8741` nos dados de teste, evidenciando a capacidade do modelo em generalizar novos dados.

Em resumo, os resultados deste projeto indicam que o modelo criado usando o algoritmo pode ser uma ótima ferramenta para prever os preços de aluguel com relativa precisão, considerando a influência significativa da localidade e do tamanho do imóvel na determinação dos preços. Este trabalho fornece uma base sólida para futuras análises e tomadas de decisão no âmbito do mercado imobiliário.

### Como Usar Este Repositório
Atente-se às políticas de Licenças do projeto e siga os passos a seguir:

- Clone este repositório para sua máquina local;
- Execute o arquivo Jupyter Notebook [Previsao_Aluguel_SP.ipynb](Previsao_Aluguel_SP.ipynb) para explorar o desenvolvimento do projeto;
- Acesse o data app seguindo as instruções neste README ou execute localmente seguindo os passos abaixo:
    - Abra o terminal no reposiório onde salvou o projeto;
    - Copie e cole o comando abaixo para abrir o data app localmente:
    
    ```
    streamlit run app.py
    ```

- Estude os códigos do app e do notebook para desenvolver seus conhecimentos.