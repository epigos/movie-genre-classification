# Machine Learning Engineering task - Philip Adzanoukpe

[Task](https://docs.google.com/document/d/1Gx2fIG2uDg1eLp-8q7mChcn6hy5FD_ObFyAsOBH_j1Q/edit)

[Jupyter notebook](https://colab.research.google.com/drive/1gGGBYXKv98w31PbHhdig4k7OO586t6UO?usp=sharing)

# Project Setup

## Using Docker

Build docker image

    make docker/build

Check available commands

    docker run truelayer_movie_classifier

Run movie_classifier

    docker run truelayer_movie_classifier movie_classifier --title "Toy story" \
        --description "Led by Woody, Andy's toys live happily in his room until Andy's birthday brings 
        Buzz Lightyear onto the scene. Afraid of losing his place in Andy's heart, Woody plots against Buzz"

Train model

    docker run truelayer_movie_classifier train

Run tests

    make docker/tests

## Using Python commands

**Use Python 3.8+**

The application is setup using Python3.8 or higher

Install dependencies:

    pip install -r requirements

Check available commands:

    python main.py

Run movie_classifier:

    python main.py movie_classifier --title "Toy story" \
        --description "Led by Woody, Andy's toys live happily in his room until Andy's birthday brings 
        Buzz Lightyear onto the scene. Afraid of losing his place in Andy's heart, Woody plots against Buzz"

Train model:

    python main.py train

Run tests:

    python -m pytest tests -s -vv -ra

# Notes

## Programming Language

I used Python programming language becuase it has a lot of libraries for Machine Learning (ML). Also, the dynamic type
nature of the language makes it easy to manupulate any data type.

## Libaries

Here are the main libraries used to train the model:

**Pandas**

I used Pandas manily to load and preprocess the data because it's a fast, powerful, flexible and easy to use open source
data analysis and manipulation tool built for Python. It has builtin functions to load, clean and extract features from
any dataset.

**Scikit-Learn**

Is a simple and efficient tool for predictive data analysis and scientific computation in Python. It has tons of machine
learning algorithms for supervised and unsupervised learning algorithms. We made use of the RandomForestClassifier
module to train the classification model. Also, it comes with features to split your dataset for trainin and testing,
encode your features and labels set into vector representation.

## The Workflow

This describes the stages involved in trainin the classification model

1. **Data preparation**:

   This is involves loading and cleaning data to remove any missing values, noisy or inconsistent values in the dataset.

2. **Feature engineering**

   This is involves transforming the natural language text into numbers/vectors suitable for ML algorithms.

3. **Training of the model**

   Make use of an ML classification algorithm to learn the partterns in the dataset in order to predict the genres of
   the movie.

4. **Evaluating the model**

   We evaluate the performance of the model using the holdout validation method.

5. **Exporting/deploying the model**

   The final model is serilised into a file which can be loaded later to predict movie genres.

## Algorithms

**Term Frequency — Inverse Document Frequency (TF-IDF)**

This is a technique to quantify a word in documents, we generally compute a weight to each word which signifies the
importance of the word in the document and corpus. This method is a widely used technique in Information Retrieval and
Text Mining. TF-IDF is a statistical measure that evaluates how relevant a word is to a document in a collection of
documents.

This algorithm was mainly used in the Feature engineering stage to convert the natural language text into numeric
formats. ML algorithms usually deal with numbers, so we needed to transform those text (title and description) into
numbers. Once we transformed words into numbers, it can then be easily fed into a Classification Algorithm.

**Random Forest**

Random Forests are an ensemble learning method for classification, regression and other tasks that operates by
constructing a multitude of decision trees at training time. For classification tasks, the output of the random forest
is the class selected by most trees. For regression tasks, the average prediction of the individual trees is returned.

The fundamental concept behind random forest is a simple but powerful one — the majority votes wins. The reason that the
random forest model works so well is: Many relatively uncorrelated models (trees) operating as a committee will
outperform any of the individual constituent models. This idea of ensemble learning influence the decision to choose
this algorithm for the classification task. In addition, its also prevents overfitting to the training set.



