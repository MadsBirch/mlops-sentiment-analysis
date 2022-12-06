# Amazon Review Sentiment Analysis using Transformers

## Project decription & goal

This is a project on Sentiment Analysis using [Transformers](https://github.com/huggingface/transformers) for the Machine Learning Operations course at DTU. 

The goal of the project is interpreting customer __feedback__ through product reviews by categorizing the feedback provided by a customer into __positive__, __negative__, and __neutral__ based on the written review. __BERT__ pretrained Natural Language Processing (NLP) model from Google was used. It easily understands the context of a word in a sentence based on previous words in the sentences due to its bi-directional approach.

## Data
The dataset of Amazon reviews within the "Automotive" category was used. It conists of 20,473 samples and can be found at:
http://jmcauley.ucsd.edu/data/amazon/links.html

Download link:
http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Automotive_5.json.gz

## Running the project
The following commands can be run in the directory. In the conf_train_sweep.yaml file, the hyper parameter ranges for the sweep is declared. In the conf_train.yaml the best hyper parameters from the sweep is declared.

Create dataset:
`make data`

Perform a sweep over hyper parameters:
`make sweep`

Train model:
`make train`

Test model:
`make test`

## Results
Using Weights and Biases we performed a hyperparameter sweep to identify the importance of the various parameters. We made a sweep over suitable value ranges for the learning rate, weight decay, dropout and the choice of optimizer. The results are summarized in the figures below.

![Figure 1](https://github.com/MadsBirch/mlops-sentiment-analysis/blob/master/reports/figures/sweep.png?raw=true) <br />
*Figure 1: Hyper parameter sweep*

![Figure 2](https://github.com/MadsBirch/mlops-sentiment-analysis/blob/master/reports/figures/param_importance.png?raw=true) <br />
*Figure 2: Table of hyper parameter importance*



Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
