Cross Validation Results
=============================

The cross validation results were generated using a seed of 123 and 30% of the
dataset for testing. The data was limited to only data pulled from version 6.12
of the game. These are the cross validation estimates of the loss given a
10-fold cross validation.

| Network       | Cross Validation Estimate |
| ------------- | ------------------------- |
| mlp384        | 25.63                     |
| cosine384     | 25.48                     |
| gated384      | 25.43                     |
| gated768      | 25.39                     |


And here are the individual results for each fold for the given model.

##mlp384

| Fold       | Loss  |
| ---------- | ----- |
| 1          | 25.32 |
| 2          | 25.28 |
| 3          | 25.85 |
| 4          | 25.82 |
| 5          | 25.82 |
| 6          | 25.87 |
| 7          | 25.87 |
| 8          | 25.82 |
| 9          | 25.32 |
| 10         | 25.35 |
| Avg        | 25.63 |

##cosine384

| Fold       | Loss  |
| ---------- | ----- |
| 1          | 25.52 |
| 2          | 25.46 |
| 3          | 25.50 |
| 4          | 25.55 |
| 5          | 25.50 |
| 6          | 25.48 |
| 7          | 25.47 |
| 8          | 25.57 |
| 9          | 25.52 |
| 10         | 25.31 |
| Avg        | 25.48 |

##gated384

| Fold       | Loss  |
| ---------- | ----- |
| 1          | 25.35 |
| 2          | 25.45 |
| 3          | 25.39 |
| 4          | 25.36 |
| 5          | 25.60 |
| 6          | 25.38 |
| 7          | 25.58 |
| 8          | 25.33 |
| 9          | 25.47 |
| 10         | 25.43 |
| Avg        | 25.43 |

##gated768

| Fold       | Loss  |
| ---------- | ----- |
| 1          | 25.38 |
| 2          | 25.36 |
| 3          | 25.36 |
| 4          | 25.45 |
| 5          | 25.42 |
| 6          | 25.44 |
| 7          | 25.36 |
| 8          | 25.36 |
| 9          | 25.39 |
| 10         | 25.46 |
| Avg        | 25.39 |
