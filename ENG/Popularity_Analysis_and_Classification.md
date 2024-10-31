
# Popularity Analysis and Classification of Brazilian Music Genres Using Artificial Intelligence

## Importing Necessary Libraries

The first step is to import the libraries needed for the project:

```python
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
```

### Library and Python Versions Used

==Python 3.12.3==
==Pandas 2.2.2==
==Numpy 1.26.4==
==Seaborn 0.13.2==
==Matplotlib 3.9.0==
==Sklearn 1.5.2==

## Loading the Dataset

The dataset used is the [🎹 Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset):

```python
file_path = 'https://raw.githubusercontent.com/FlamingoLindo/spotify-svm/main/spotify.csv'
df = pd.read_csv(file_path)
df.head()
```

The dataset consists of the following columns:

| Column           | Description |
|------------------|-------------|
| **track_id**     | The Spotify ID for the track. |
| **artists**      | The names of the artists who performed the track. If there is more than one artist, they are separated by a semicolon (`;`). |
| **album_name**   | The name of the album in which the track appears. |
| **track_name**   | The name of the track. |
| **popularity**   | A value between 0 and 100 indicating the track's popularity, with 100 being the most popular. Calculated algorithmically based on factors like total plays and recent play count. Duplicate tracks are rated independently. |
| **duration_ms**  | The track's length in milliseconds. |
| **explicit**     | Indicates whether the track has explicit lyrics (`true` = yes; `false` = no or unknown). |
| **danceability** | A measure from 0.0 to 1.0 indicating how suitable the track is for dancing, based on factors like tempo, rhythm stability, beat strength, and regularity. |
| **energy**       | A measure from 0.0 to 1.0 representing the track's intensity and activity. Higher values mean higher intensity. |
| **key**          | The key the track is in, represented as an integer. |
| **loudness**     | The track's overall loudness in decibels (dB). |
| **mode**         | The modality of the track's scale. `1` for major, `0` for minor. |
| **speechiness**  | A measure of spoken words in the track. Closer to 1.0 indicates mostly spoken words (e.g., audiobooks). |
| **acousticness** | A confidence measure from 0.0 to 1.0 indicating if the track is acoustic. |
| **instrumentalness** | Predicts whether the track contains no vocals. Higher values indicate instrumental content. |
| **liveness**     | Detects the presence of an audience. Higher values indicate a higher probability that the track was performed live. |
| **valence**      | A measure from 0.0 to 1.0 describing the musical positiveness. Higher values sound more positive. |
| **tempo**        | The track's tempo in beats per minute (BPM). |
| **time_signature** | An estimated time signature (e.g., 3/4, 4/4). |
| **track_genre**  | The genre of the track. |

## Preprocessing Steps

Following this, unique values in the `track_genre` column were analyzed to select Brazilian genres only: `brazil`, `mpb`, `pagode`, `samba`, `sertanejo`.

```python
df = df[df['track_genre'].isin(['brazil', 'mpb', 'pagode', 'samba', 'sertanejo'])]
train_df = df.copy()
train_df.dropna(inplace=True)
train_df.drop(['Unnamed: 0', 'track_id'], axis=1, inplace=True)
```

## Encoding Categorical Columns

The categorical columns (`artists`, `album_name`, `track_name`, `track_genre`) were converted into numeric values:

```python
le = LabelEncoder()
train_df['artists'] = le.fit_transform(train_df['artists'])
train_df['album_name'] = le.fit_transform(train_df['album_name'])
train_df['track_name'] = le.fit_transform(train_df['track_name'])
train_df['track_genre'] = le.fit_transform(train_df['track_genre'])
```

## Model Training

After preprocessing, data was split into training and testing sets:

```python
X = train_df.drop('popularity', axis=1)
y = train_df['popularity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### First Model Training (SVC)

```python
model = SVC()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
```

![matrizPop1](https://github.com/FlamingoLindo/spotify-svm/blob/main/images/matrix/MATRIX1(POPULARITY)_Sat%20Oct%2019%2012-22-16%202024.png?raw=true "Matriz de confusão Popularidade 1")

![matrizGen1](https://github.com/FlamingoLindo/spotify-svm/blob/main/images/1.png?raw=true "Matriz de confusão Genêro 1")

### Hyperparameter Tuning

A grid search with cross-validation was performed for the `SVC` model:

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

## Confusion Matrix and Feature Importance

Confusion matrix visualization and feature importance analysis were then performed:

```python
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()
result = permutation_importance(grid_search.best_estimator_, X_test_scaled, y_test, n_repeats=10, random_state=42)
sorted_idx = result.importances_mean.argsort()
plt.barh(range(len(sorted_idx)), result.importances_mean[sorted_idx], align='center')
plt.xlabel('Feature Importance')
plt.show()
```

![featPop](https://github.com/FlamingoLindo/spotify-svm/blob/main/images/features/FEATURES(POPULARITY)_Sat%20Oct%2019%2012-58-07%202024.png?raw=true "Features Pop")

![featGen](https://github.com/FlamingoLindo/spotify-svm/blob/main/images/3.png?raw=true "Features Gen")
