[pt-br](https://github.com/FlamingoLindo/spotify-svm) &nbsp ;&nbsp ;&nbsp ; [eng](https://github.com/FlamingoLindo/spotify-svm)

---

# Análise de Popularidade e Classificação de Gêneros Musicais Brasileiros Usando Inteligência Artificial

## Análise de Popularidade

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

==Python ==

==Pandas ==

==Numpy ==

==Seaborn ==

==Matplotlib ==

==Sklearn ==

Em seguida foi importado o dataset [🎹 Spotify Tracks Dataset]()

```python
file_path = ''
df = pd.read_csv(file_path)
df.head()
```

O dataset é composto pelas seguintes colunas:

1. Unnamed: 0 (explicação)

2. track_id (explicação)

Após uma análise foi observado que haviam diversos tipos de gêneros musicais dentro desse dataset, por isso optamos por apenas pegar as musicas dos gêneros: `brazil`, `mpb`, `pagode`, `samba` e `sertanejo`.

```python
df = df[(df['track_genre'].isin(['brazil','mpb','pagode','samba','sertanejo']))]
```
