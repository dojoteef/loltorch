Validation Results
=============================

  * [Overview](#intro)
  * [Methodology](#methods)
  * [Model Architectures](#architectures)
  * [Pretrained Models](#pretrained)
  * [Cross Validation](#cv)
  * [Caveats](#caveats)

<a name="intro"/>
## Overview

This is an overview of the validation results for the included pretrained
models. The models were trained on data pulled from the League of Legends API
using [lualol](https://github.com/dojoteef/lualol). It was gathered by sourcing
the match lists of all the Master and Challenger ranked players then pulling
those matches. Since the Master and Challenger players have played games with
players of various rankings, it does include games against participants who are
not in the Master and Challenger leagues, though likely at a smaller percentage.
The networks were then trained on data from versions 6.11, 6.12, and 6.13 as
version 6.14 had yet to be released.

<a name="methods"/>
## Methodology

The networks output a vector for each target feature. There are sixty-nine
target features that correspond to two spells, six items, thirty runes, and
thirty masteries. The loss is determined as the cosine similarity of the target
features versus the predicted features, where it maximizes the similarity when
the outcome of the match is a win and minimizes it for a loss. The total loss is
then the sum of the loss for each of these features. This has the effect of
trying to make the predictions look more like winning champion builds versus
those which lose.

The model architectures were guided by the use of 10-fold cross validation.
While many architectures were tried, the most promising ones are detailed below.

<a name="architectures"/>
## Model Architectures

A number of model architectures were included in the final results for this
project. The way all of the models work is that each of the input features are
separated into their own network, then summed to make a final prediction of the
target features.

### MLP Models

These use a simple multi-layer perceptron for each input vector with 40%-50%
dropout. They use linear layers with a Hard Shrink transfer function that have
an activation threshold of 0.25. Use of the Hard Shrink was chosen as a counter
to ReLU activation to see how well it compares to the use of ReLU since it
allows for zeroing out activation is both positive and negative directions. It
turned out that for the simple multi-layer perceptron that the Hard Shrink
produced better results than the ReLU, this could be due to simple nature of the
model.

### Gated Models

The gated models are very similar to the multi-layer perceptron though use ReLU
as the activation functions and have a simple gating network such that each input
feature network is multiplied by the gating network before being summed. This
better allows for the combination of input feature vectors through weighting the
contribution of each parameter of the input vector. This in turn produces a
better result than the simple multi-layer perceptron model.

### Cosine Gated Models

This final model is a variation of the gated models, but uses the cosine
similarity between the input and n number of target vectors as the transform
between layers rather than a linear layer. This was chosen since the loss for
the model is the cosine similarity which maximizes cases where the outcome of
the match is a win and minimizes when the match outcomes is a loss.

<a name="pretrained"/>
## Pretrained Models

The pretrained networks included in this project were trained on 80% of the
data, with 10% reserved for validation per epoch, and 10% reserved for final
testing to verify generalization. Below is the testing loss for each of the
pretrained networks.

| Network       | Loss  |
| ------------- | ----- |
| mlp384        | 25.32 |
| cosine384     | 25.48 |
| gated384      | 25.53 |
| gated768      | 25.57 |


<a name="cv"/>
## Cross Validation

The cross validation results were generated using a manual seed of 123 and 30% of the
dataset for testing. The data was limited to only data pulled from version 6.12
of the game. These are the cross validation estimates of the loss given a
10-fold cross validation.

| Network       | Cross Validation Estimate |
| ------------- | ------------------------- |
| mlp384        | 25.63                     |
| cosine384     | 25.48                     |
| gated384      | 25.43                     |
| gated768      | 25.39                     |


What follows is the individual results for each fold for the given model.

### mlp384

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

### cosine384

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

### gated384

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

### gated768

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

<a name="caveats"/>
## Caveats

The results are likely skewed and have more to do with the particular order in
which the data is trained on and/or the validation and test set selection (which
is based on the initial seed). There are a number of reasons this is likely
true. The cross validation results do not translate well to the actual training,
which makes sense due to the validation and test sets differing between training
of each individual model. Theoretically that should have limited impact overall
if the data is independently and identically distributed, but by nature of the
data that was pulled from the servers this is almost certainly not the case
(for example, most of the data is from Master and Challenger leagues while a
tiny portion is from the rest of the leagues as those cannot be obtained as
easily from the League of Legends API).

Since the cross validation used the same split for the validation and test sets
as well as the same data ordering for comparing the various models it should
paint a more accurate picture of which models would be most effective given data
that is independently and identically distributed.

As such there is more that can be done to make this project robust to these
issues, but in the interest of not spending too much time on this one project
(especially due to the lengthy training time on a 2013 Macbook Pro) little
effort has been put in to rectify these issues.
