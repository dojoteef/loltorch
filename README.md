# loltorch - A build optimizer based on Torch7 neural networks

  * [What does loltorch do?](#intro)
  * [Requirements](#requirements)
  * [Installation](#install)
  * [Trying loltorch out](#try)
  * [Training your own networks](#train)
  * [How does loltorch work?](#approach)
  * [Neural Network Details](#details)
  * [Caveats](#caveats)
  * [Disclosure](#disclosure)

## What does loltorch do?
<a name="intro"/>

loltorch pulls match data from the [League of Legends
API](http://developer.leagueoflegends.com) and runs the data through a neural
network which tries to make the Masteries, Runes, Spells, and Items look more
like what the winning team used and less like the losing team. This predicts
what the optimal winning build is for the combination of 1) Champion 2) Role 3)
Lane 4) Region and 5) League of Legends version. Even if given a combination of
those attributes not seen in the data it will make a prediction as to what an
optimal build might look like. For example, if you wanted to know what a support
character like Blitzcrank might need in order to be an effective jungler, this
project aims to do just that.

## Requirements
<a name="requirements"/>

This project depends on:
 * [Penlight](https://github.com/stevedonovan/Penlight)
(pl)
 * [lua-cjson](https://github.com/mpx/lua-cjson)
 * [lualol](https://github.com/dojoteef/lualol)
 * [torch7](http://torch.ch)
 * [nn](https://github.com/torch/nn)
 * [nngraph](https://github.com/torch/nngraph)
 * [tds](https://github.com/torch/tds)
 * [threads](https://github.com/torch/threads)

### Installation
<a name="install"/>

1. [Install Torch](http://torch.ch/docs/getting-started.html).
2. Install the following additional Lua libraries:

```bash
luarocks install lualol
luarocks install nn
luarocks install nngraph
luarocks install tds
luarocks install threads
```

Note if you have any trouble are trying to install loltorch on OSX from within
torch's LuaRocks installation you may need to use following command if you get
an error about not having [OpenSSL](https://www.openssl.org) installed.

    > luarocks install lualol OPENSSL_DIR=/usr/local/opt/openssl/

## Trying loltorch out
<a name="try"/>
Once you have loltorch installed the quickest way to get some predictions is to
use one of the pretrained models. Note that the pretrained models were trained
on data from versions 6.11, 6.12, and 6.13. So first you need to get the static
data for version 6.13:

```bash
outdir=pretrained version=6.13.1 ./bin/getstaticdata
```

Now that you have the static data downloaded you can use the following command

```bash
modeldir=pretrained ./bin/sample mlp384 -datadir pretrained
```

to get a preset list of examples comparing various roles for a subset of
champions. Using the following command

```bash
modeldir=pretrained ./bin/sample_odd mlp384 -datadir pretrained
```

will give some examples of weird combinations such as having Blitzcrank as an
adc.

For trying out your own combinations simply use the command:

```bash
th sample.lua -help
```

to see a list of options.

### Training your own networks
<a name="train"/>
In order to train your own network on new data you need to first get the match
data and process it:

```bash
./bin/getstaticdata
./bin/getallmatches
./bin/processmatches
```

Then simply train a model using the data. Use the following command to get a
list of training options (and their defaults):

```bash
./bin/trainbuild
```

Note that when actually training, the first parameter to the `trainbuild` script
needs to be the model type, so the minimal command to train the `gated384` model
would be:

```bash
./bin/trainbuild gated384
```

## How does loltorch work?
<a name="approach"/>

loltorch first pulls all the ranked match data from the League of Legends API
for the Master and Challenger leagues then takes all the features of the match
(Champion, Role, Runes, Spells, etc) for each participant and puts them into
a corpus.txt which is then run through word2vec in order to generate vectors
which makes closely related features more similar to each other. Then when
training the neural network tries to make the cosine similarity of the features
of the winning team large and those of the lossing team look smaller. This has
the effect of making the final champion builds that this project predicts to be
what is optimal for winning a match.

When actually sampling predictions from the neural network it generates an
output of vectors that are then compared to the feature vectors to see which
ones are closest in similarity. This similarity value is then able to be output
(along with filtering it based on a threshold), such that you can see how
confident the neural network is in predicting that particular feature.


### Neural Network Details
<a name="details"/>
A number of architectures were tried in regards to what would be an effective
neural network. The final neural network models that were chosen for inclusion
are visible in the 'models.lua' file. Through the use of cross validation the
best neural network results were obtained by using the 'gated768' model which
uses a separate MLP (multi-layer perceptron) per input feature and then combines
them based on a separate neural network which is the 'gate'.

The results of the final k-fold cross validation are listed in the
pretrained/README.md file, showing the parameters used for training and
cross validation scores of each network tested.


### Caveats
<a name="caveats"/>
Since I only have a 2013 model Macbook Pro with a Core i7 I only implemented the
neural network using the threading library in Torch7. While the laptop I have
has a dedicated GPU, in initial testing it turned out to be much slower than
running threads on the Core i7, so no CUDA support is built into the training
scripts.

Since the Core i7 is very slow in generating some of these results there may be
much better networks (that have many more parameters) than the ones I chose. For
example, the 'gated768' model has better cross validation results than the
'gated384' model, but since it took over three days to obtain the cross
validation results on only a subset of the total data there was only so much
time I was willing to commit to determining better models. If someone is so
inclined and has powerful enough GPU(s) they could add support for CUDA and try
out larger models to see if they produce better results.

The approach of using word2vec to generate the vectors was a decision based on
performance. Initially an approach using Torch7 was used, but this turned out to
be much slower. For that reason word2vec with a text corpus was chosen as a
close analogue to generating vectors based on the input features using Torch7.


## Disclosure
<a name="disclosure"/>

loltorch isn't endorsed by Riot Games and doesn't reflect the views or opinions
of Riot Games or anyone officially involved in producing or managing League of
Legends. League of Legends and Riot Games are trademarks or registered
trademarks of Riot Games, Inc. League of Legends © Riot Games, Inc.
