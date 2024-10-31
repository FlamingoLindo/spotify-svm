[PT-BR](https://github.com/FlamingoLindo/spotify-svm) 🇧🇷 | [ENG](https://github.com/FlamingoLindo/spotify-svm/blob/main/ENG/Popularity_Analysis_and_Classification.md) 🇺🇸

---

# Análise de Popularidade e Classificação de Gêneros Musicais Brasileiros Usando Inteligência Artificial

O primeiro passo é importar as bibliotecas necessárias para o funcionamento do projeto, são elas:

```python
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import matplotlib
 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
```

Versões das bibliotecas e python que foram utilizadas na execução do projeto:

* Python 3.12.3

* Pandas 2.2.2

* Numpy 1.26.4

* Seaborn 0.13.2

* Matplotlib 3.9..0

* Sklearn 1.5.2

Em seguida foi importado o dataset [🎹 Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)

```python
file_path = 'https://raw.githubusercontent.com/FlamingoLindo/spotify-svm/main/spotify.csv'
df = pd.read_csv(file_path)
df.head()
```

O dataset é composto pelas seguintes colunas:

| Coluna              | Descrição                                                                                                                                                                                                                                                                                                                                                                                                              |
|---------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **track_id**        | O ID da faixa no Spotify.                                                                                                                                                                                                                                                                                                                                                                                               |
| **artists**         | Os nomes dos artistas que performaram a faixa. Se houver mais de um artista, eles são separados por ponto e vírgula (`;`).                                                                                                                                                                                                                                                                                              |
| **album_name**      | O nome do álbum no qual a faixa aparece.                                                                                                                                                                                                                                                                                                                                                                                |
| **track_name**      | O nome da faixa.                                                                                                                                                                                                                                                                                                                                                                                                        |
| **popularity**      | Um valor entre 0 e 100 que indica a popularidade da faixa, sendo 100 a mais popular. Calculado algoritmicamente com base em fatores como o número total de execuções e a contagem de execuções recentes. Faixas duplicadas (por exemplo, a mesma faixa de um single e de um álbum) são avaliadas independentemente. A popularidade de artistas e álbuns é derivada matematicamente da popularidade das faixas. |
| **duration_ms**     | A duração da faixa em milissegundos.                                                                                                                                                                                                                                                                                                                                                                                    |
| **explicit**        | Indica se a faixa possui letras explícitas (`true` = sim; `false` = não ou desconhecido).                                                                                                                                                                                                                                                                                                                               |
| **danceability**    | Uma medida de 0.0 a 1.0 que indica a adequação da faixa para dançar, baseada em elementos como tempo, estabilidade do ritmo, força da batida e regularidade. Valores maiores indicam maior dançabilidade.                                                                                                                                                                       |
| **energy**          | Uma medida de 0.0 a 1.0 que representa a intensidade e atividade da faixa. Faixas com alta energia costumam parecer rápidas, altas e ruidosas. Por exemplo, death metal tem alta energia, enquanto um prelúdio de Bach tem baixa energia.                                                                                                                                      |
| **key**             | A tonalidade da faixa, representada por um número inteiro. Utiliza a notação de Classe de Altura: 0 = Dó, 1 = Dó♯/Ré♭, 2 = Ré, etc. Se nenhuma tonalidade for detectada, o valor é `-1`.                                                                                                                                                                                       |
| **loudness**        | A intensidade geral da faixa em decibéis (dB).                                                                                                                                                                                                                                                                                                                                                                         |
| **mode**            | A modalidade da escala da faixa. `1` representa maior e `0` representa menor.                                                                                                                                                                                                                                                                                                                                           |
| **speechiness**     | Mede a presença de palavras faladas. Valores próximos a 1.0 indicam gravações majoritariamente faladas (ex: audiolivros). Entre 0.33 e 0.66 sugere músicas com discurso (ex: rap), enquanto abaixo de 0.33 representa, em sua maioria, música.                                                                                                                                  |
| **acousticness**    | Medida de confiança de 0.0 a 1.0 sobre se a faixa é acústica, com 1.0 representando alta confiança de que é acústica.                                                                                                                                                                                                                                                            |
| **instrumentalness** | Prediz se uma faixa contém ou não vocais. Valores mais próximos de 1.0 indicam maior probabilidade de não ter vocais. Sons de "ooh" e "aah" são considerados instrumentais, mas faixas de rap ou fala são vocais.                                                                                                                                                              |
| **liveness**        | Detecta a presença de uma audiência. Valores altos aumentam a probabilidade de que a faixa foi gravada ao vivo, com valores acima de 0.8 indicando forte probabilidade de gravação ao vivo.                                                                                                                                                                                     |
| **valence**         | Uma medida de 0.0 a 1.0 da positividade musical da faixa. Valores altos transmitem emoções mais positivas (ex: feliz, alegre), enquanto valores baixos indicam emoções mais negativas (ex: triste, irritado).                                                                                                                                                                  |
| **tempo**           | O tempo estimado em batidas por minuto (BPM), refletindo a velocidade ou ritmo da faixa.                                                                                                                                                                                                                                                                                         |
| **time_signature**  | Uma assinatura de tempo estimada, variando de 3 a 7, indicando métricas como 3/4 até 7/4.                                                                                                                                                                                                                                                                                        |
| **track_genre**     | O gênero ao qual a faixa pertence.                                                                                                                                                                                                                                                                                                                                                                                     |

Então foi feito uma análise de valores únicos da coluna `track_genre` e esse foi o resultado:
```python 
array(['acoustic', 'afrobeat', 'alt-rock', 'alternative', 'ambient',
       'anime', 'black-metal', 'bluegrass', 'blues', 'brazil',
       'breakbeat', 'british', 'cantopop', 'chicago-house', 'children',
       'chill', 'classical', 'club', 'comedy', 'country', 'dance',
       'dancehall', 'death-metal', 'deep-house', 'detroit-techno',
       'disco', 'disney', 'drum-and-bass', 'dub', 'dubstep', 'edm',
       'electro', 'electronic', 'emo', 'folk', 'forro', 'french', 'funk',
       'garage', 'german', 'gospel', 'goth', 'grindcore', 'groove',
       'grunge', 'guitar', 'happy', 'hard-rock', 'hardcore', 'hardstyle',
       'heavy-metal', 'hip-hop', 'honky-tonk', 'house', 'idm', 'indian',
       'indie-pop', 'indie', 'industrial', 'iranian', 'j-dance', 'j-idol',
       'j-pop', 'j-rock', 'jazz', 'k-pop', 'kids', 'latin', 'latino',
       'malay', 'mandopop', 'metal', 'metalcore', 'minimal-techno', 'mpb',
       'new-age', 'opera', 'pagode', 'party', 'piano', 'pop-film', 'pop',
       'power-pop', 'progressive-house', 'psych-rock', 'punk-rock',
       'punk', 'r-n-b', 'reggae', 'reggaeton', 'rock-n-roll', 'rock',
       'rockabilly', 'romance', 'sad', 'salsa', 'samba', 'sertanejo',
       'show-tunes', 'singer-songwriter', 'ska', 'sleep', 'songwriter',
       'soul', 'spanish', 'study', 'swedish', 'synth-pop', 'tango',
       'techno', 'trance', 'trip-hop', 'turkish', 'world-music'],
      dtype=object)
``` 


Após à análise foi observado que haviam diversos tipos de gêneros musicais dentro desse dataset, então optamos por apenas utilizar os gêneros brasileiros: `brazil`, `mpb`, `pagode`, `samba` e `sertanejo`.

```python
df = df[(df['track_genre'].isin(['brazil','mpb','pagode','samba','sertanejo']))]
```

Então é criado um novo dataframe chamdo `train_df`, para que todas as alterações seguintes sejam feitas nele ao invés de serem feitas no dataframe original.

```python
train_df = df.copy()
```

Após a criação do novo dataframe é feita uma outra verificação só que dessa vez para ánalisar se há algum valor nulo no dataframe, caso haja eles serão deletados.

```python
train_df.dropna(inplace=True)
train_df.isna().sum()
```

Antes de separar os dados em treino e teste é necessário já remover duas colunas que são desnecessárias para o treinamento do modelo, são elas: `Unnamed: 0` e `track_id`

```python
train_df.drop([ 'Unnamed: 0', 'track_id'], axis=1, inplace=True)
```

Então é feito uma contagem de quantidade de linhas que sobraram na cópia do dataset, e a quantidade restante é 5000.

```python
print(len(train_df))
5000
```

O ultímo passo do pré-processamento dos dados é transformar as colunas categorias (`artists`, `album_name`, `track_name` e `track_genre`) em colunas com valores númericos. 

Esse processo é feito utilizando um tipo de *encoding*, nessa pesquisa foi utilizado o `labelEcoder.

```python
le = LabelEncoder()

train_df['artists'] = le.fit_transform(train_df['artists'])

train_df['album_name'] = le.fit_transform(train_df['album_name'])

train_df['track_name'] = le.fit_transform(train_df['track_name'])

train_df['track_genre'] = le.fit_transform(train_df['track_genre'])
```

Depois de todas as etapas do pré-processamento concluídas começa-se o treinamento do modelo.

Primeiro é feito a divisão dos dados em `X`e `y`, onde `X` serão as todas as colunas menos a coluna de popularidade (`popularity`) e o `y` será apenas a coluna de popularidade.

Então os dados são dividos em 80% para treino e 20% para testes.

É importante usar o valor **42** no parâmetro `random_state` para que os resultados sejam os mesmos não importa em qual máquina esse script seja executado.

```python
X = train_df.drop('popularity', axis=1)
y = train_df['popularity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Em sequência é feito o primeiro treino do modelo, apenas utilizando o SVC (Classificação de Vetores de Suporte) puro.

```python
model = SVC()
model.fit(X_train_scaled, y_train)
```

Então é criado um gráfico do estilo matriz de confusão onde é possivel verificar os resultados desse primeiro teste.

```python
y_pred = model.predict(X_test_scaled)
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(14, 11))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.title('Matriz de Confusão - SVM Linear')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.show()
print(classification_report(y_test, y_pred))
```

![matrizPop1](https://github.com/FlamingoLindo/spotify-svm/blob/main/images/matrix/MATRIX1(POPULARITY)_Sat%20Oct%2019%2012-22-16%202024.png?raw=true "Matriz de confusão Popularidade 1")

![matrizGen1](https://github.com/FlamingoLindo/spotify-svm/blob/main/images/1.png?raw=true "Matriz de confusão Genêro 1")

Então é feito um novo treino do modelo só que dessa vez utilizando uma grade de hiperparâmetros com diferentes valores para `C`, `gamma` e `kernel`.

E tambem é feito um *grid search* com valor igual a **5**, métrica de avalição `accuracy`, `n_jobs = -1` e `verbose = 3` apenas para exibir informações no console de acordo com o progresso do treino.

```python
svm = SVC(probability=True)
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['linear', 'rbf', 'poly']
}

grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=3)

grid_search.fit(X_train_scaled, y_train)
```

Após o termino do segundo treino é impresso melhores parâmetros e estimadores.

```python
# POPULARIDADE
print('Melhores parâmetros: ', grid_search.best_params_)
print('Melhor estimador: ', grid_search.best_estimator_)

Melhores parâmetros:  {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'}
Melhor estimador:  SVC(C=10, gamma=0.1, probability=True)
```

```python
# GENÊRO
print('Melhores parâmetros: ', grid_search.best_params_)
print('Melhor estimador: ', grid_search.best_estimator_)

Melhores parâmetros:  {'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}
Melhor estimador:  SVC(C=10, gamma=0.01, probability=True)
```

E então é feita a criação da matriz de confusão utilizandos os estimadores.

```python
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(14, 11))
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.title('Matriz de Confusão')
plt.show()

print(classification_report(y_test, y_pred))
```

![matrizPop2](https://github.com/FlamingoLindo/spotify-svm/blob/main/images/matrix/MATRIX2(POPULARITY)_Sat%20Oct%2019%2012-58-03%202024.png?raw=true "Matriz de confusão Popularidade 2")

![matrizGen2](https://github.com/FlamingoLindo/spotify-svm/blob/main/images/2.png?raw=true "Matriz de confusão Genêro 2")

E por fim é feito um gráfica para a análise das *features* do dataset e a sua importância para a classificação de popularidade ou genêro.

```python
result = permutation_importance(best_svm, X_test_scaled, y_test, n_repeats=10, random_state=42)
sorted_idx = result.importances_mean.argsort()
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), result.importances_mean[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), [X.columns[i] for i in sorted_idx])
plt.xlabel('Importância das features')
plt.title('Importância das Features via Permutação no dataset')
plt.show()
```

![featPop](https://github.com/FlamingoLindo/spotify-svm/blob/main/images/features/FEATURES(POPULARITY)_Sat%20Oct%2019%2012-58-07%202024.png?raw=true "Features Popularidade")

![featGen](https://github.com/FlamingoLindo/spotify-svm/blob/main/images/3.png?raw=true "Features Genêro")
