[PT-BR](https://github.com/FlamingoLindo/spotify-svm) 🇧🇷 | [ENG](https://github.com/FlamingoLindo/spotify-svm/blob/main/ENG/Popularity_Analysis_and_Classification.md) 🇺🇸

---

# Brazilian Songs Popularity and Genres Classification Using Artificial Intelligence

The first step is importing the necessaries libraries for the project functioning, and they are:

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

Python and libraries being used in the project:

* Python 3.12.3

* Pandas 2.2.2

* Numpy 1.26.4

* Seaborn 0.13.2

* Matplotlib 3.9..0

* Sklearn 1.5.2

Then the dataset is imported [🎹 Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)

```python
file_path = 'https://raw.githubusercontent.com/FlamingoLindo/spotify-svm/main/spotify.csv'
df = pd.read_csv(file_path)
df.head()
```

The dataset is made out of the following columns:

| **Attribute**      | **Description**                                                                                                                                              |
|--------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| track_id           | The Spotify ID for the track                                                                                                                                  |
| artists            | The artists' names who performed the track. If there is more than one artist, they are separated by a `;`                                                      |
| album_name         | The album name in which the track appears                                                                                                                     |
| track_name         | Name of the track                                                                                                                                             |
| popularity         | The popularity of a track, a value between 0 and 100, based on total number of plays and recency of plays. Higher values indicate more popularity.             |
| duration_ms        | The track length in milliseconds                                                                                                                             |
| explicit           | Whether or not the track has explicit lyrics (`true` = yes; `false` = no or unknown)                                                                          |
| danceability       | Danceability is a measure of how suitable a track is for dancing based on musical elements like tempo, rhythm stability, and beat strength (0.0 to 1.0).       |
| energy             | A measure from 0.0 to 1.0 representing the intensity and activity of the track. Energetic tracks feel fast, loud, and noisy.                                   |
| key                | The key the track is in. Integers map to pitches using standard Pitch Class notation (e.g., 0 = C, 1 = C♯/D♭, etc.). If no key detected, value is -1.          |
| loudness           | The overall loudness of a track in decibels (dB)                                                                                                             |
| mode               | Indicates the modality (major or minor) of the track. Major = 1, Minor = 0                                                                                   |
| speechiness        | Detects the presence of spoken words. Values above 0.66 represent mostly spoken words, 0.33-0.66 may contain both music and speech, and below 0.33 is music.   |
| acousticness       | Confidence measure (0.0 to 1.0) of whether the track is acoustic. 1.0 indicates high confidence the track is acoustic.                                      |
| instrumentalness   | Predicts whether a track contains no vocals. A value closer to 1.0 suggests the track has no vocals.                                                         |
| liveness           | Detects the presence of an audience in the recording. Higher values indicate a live performance (value > 0.8 suggests strong likelihood of being live).       |
| valence            | A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. High values sound more positive (happy, cheerful), low values are negative. |
| tempo              | The overall estimated tempo of a track in beats per minute (BPM). Indicates the speed or pace of the piece.                                                   |
| time_signature     | An estimated time signature (meter) ranging from 3 to 7, indicating how many beats are in each bar (e.g., 3/4, 4/4, etc.).                                  |
| track_genre        | The genre in which the track belongs                                                                                                                          |

Then a analysis was made at the `track_genre` column and this was the result:
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

After the analysis it was possible to see that there were a lot of diversity in the genres, so it was decided to only use the brazilian genres that are: `brazil`, `mpb`, `pagode`, `samba` e `sertanejo`.

```python
df = df[(df['track_genre'].isin(['brazil','mpb','pagode','samba','sertanejo']))]
```

Then a copy of the dataframe is made `train_df`, so that all the following changes are made in the copy insted of being make over at the original one.

```python
train_df = df.copy()
```

After the copying the dataset, its check to see if there are any null values in the dataset, if so they are removed.

```python
train_df.dropna(inplace=True)
train_df.isna().sum()
```

Before splitting the dataset between train and test is necessary to remove two columns, they are: `Unnamed: 0` and `track_id`

```python
train_df.drop([ 'Unnamed: 0', 'track_id'], axis=1, inplace=True)
```

Then a couting is made to see how many lines were left in the new dataset, and the amount is 5000.

```python
print(len(train_df))
5000
```

The last step of data pre-processing is to transform the category columns (`artists`, `album_name`, `track_name` and `track_genre`) into columns with numeric values. 

This process is done using a type of *encoding*, in this research `labelEcoder` was used.

```python
le = LabelEncoder()

train_df['artists'] = le.fit_transform(train_df['artists'])

train_df['album_name'] = le.fit_transform(train_df['album_name'])

train_df['track_name'] = le.fit_transform(train_df['track_name'])

train_df['track_genre'] = le.fit_transform(train_df['track_genre'])
```

After all pre-processing steps are completed, model training begins.

First, the data is divided into `X` and `y`, where `X` will be all the columns except the popularity column (`popularity`) and `y` will be just the popularity column.

Then the data is divided into 80% for training and 20% for testing.

It is important to use the value **42** in the `random_state` parameter so that the results are the same no matter which machine this script is run on.

```python
X = train_df.drop('popularity', axis=1)
y = train_df['popularity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Next, the model is first trained, using SVC (Support Vector Classification).

```python
model = SVC()
model.fit(X_train_scaled, y_train)
```

Then a confusion matrix style graph is created where it is possible to check the results of this first test.

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

Then the model is retrained, but this time using a grid of hyperparameters with different values ​​for `C`, `gamma` and `kernel`.

And a *grid search* is also done with a value equal to **5**, evaluation metrics `accuracy`, `n_jobs = -1` and `verbose = 3` just to display information in the console according to the progress of the training.

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

After the end of the second training, better parameters and estimators are printed.

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

And then the confusion matrix is ​​created using the estimators.

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

And finally, a graph is made to analyze the *features* of the dataset and their importance for the popularity or genre classification.

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
