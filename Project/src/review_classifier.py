__author__ = "Kevin Trejos Vargas"
__email__  = "kevin.trejosvargas@ucr.ac.cr"

"""
MIT License

Copyright (c) 2023 Kevin Trejos Vargas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


################################################################################
# --                          Necessary libraries                           -- #
################################################################################

# -- Libraries to record the date and time
from datetime import datetime
import pytz

# -- Libraries to use SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection         import train_test_split
from sklearn.svm                     import SVC
from sklearn.metrics                 import accuracy_score
from sklearn.metrics                 import precision_score
from sklearn.metrics                 import recall_score
from sklearn.metrics                 import f1_score
from sklearn.metrics                 import confusion_matrix

# -- Libraries to include inputs directory
import sys, os

# -- directories
outWD = os.getcwd().replace('src','output')
inWD  = os.getcwd().replace('src','input')

# -- Libraries to use word embeddings
from   keras.models  import Sequential
from   keras.layers  import Flatten
from   keras.layers  import Dense
from   gensim.models import Word2Vec
from   nltk.tokenize import word_tokenize
import numpy         as     np
import tensorflow    as     tf
import nltk
import ssl

# -- Library for hyperparameters optimization
from hyperopt                 import tpe, fmin

# -- Libraries for plotting the confusion matrix
import matplotlib.pyplot as plt
import seaborn           as sns

# -- Import the input files
sys.path.append(f"{inWD}")
from relevant_dataset import dataset
from optimization_search_space import svc_search_space  , nn_search_space
from optimization_search_space import svc_default_params, nn_default_params

# -- Enable nltk tokenization
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt', quiet = True, raise_on_error = True)  


################################################################################
# --                     Convert paragraphs to vectors                      -- #
################################################################################
def paragraph_to_vector(paragraph, model):
    vector = [model.wv[word] for word in paragraph if word in model.wv]
    return np.mean(vector, axis=0) if vector else np.zeros(model.vector_size)


################################################################################
# --          Pre-process the dataset to get the corpus and labels          -- #
################################################################################
def generate_corpus_and_labels(dataset):
    corpusData = []
    labelsData = []
    for line in dataset:
        revCount = line["user_data"]["review_count"]
        useful   = line["user_data"]["useful"]
        labelsData.append(1 if useful / ( revCount + 1) >= 0.9 else 0)
        corpusData.append(line["text"])
    return corpusData, labelsData


################################################################################
# --              Utilize Bigrams with Support Vector Machine               -- #
################################################################################
def svm_approach(corpusData, labelsData, params):

    # -- Set the parameters to configure the classifier
    kernel                  = params["kernel"]
    C                       = params["C"]
    gamma                   = params["gamma"]
    decision_function_shape = params["decision_function_shape"]
    degree                  = params["degree"]
    if 'opt' in approach:
        file = open(f"{outWD}/svc_optimization.log", "a")
        file.write(f"kernel: {kernel} - ")
        file.write(f"C: {C} - ")
        file.write(f"gamma: {gamma} - ")
        file.write(f"decision_function_shape: {decision_function_shape} - ")
        file.write(f"degree: {degree}\n")
        file.close()

    # -- Create the bigram features
    vectorizer   = CountVectorizer(ngram_range=(2, 2))
    sparseMatrix = vectorizer.fit_transform(corpusData)

    # -- Split the data intro training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        sparseMatrix, labelsData, test_size = 0.2, random_state = 42
    )

    # -- Train the classifier
    if kernel == 'linear':
        classifier = SVC(
            kernel                  = kernel,
            C                       = C,
            decision_function_shape = decision_function_shape
        )
    elif kernel == 'poly':
        classifier = SVC(kernel = kernel, C = C, gamma = gamma, degree = degree)
    elif kernel in ['rbf', 'sigmoid']:
        classifier = SVC(kernel = kernel, C = C, gamma = gamma)
    classifier.fit(X_train, y_train)

    # -- Make predictions
    y_pred = classifier.predict(X_test)
    
    # -- Compute the metrics
    accuracy   = accuracy_score(y_test, y_pred)
    precision  = precision_score(y_test, y_pred)
    recall     = recall_score(y_test, y_pred)
    f1         = f1_score(y_test, y_pred)
    confMatrix = confusion_matrix(y_test, y_pred)
    
    # -- Return the results
    metrics               = {}
    metrics["accuracy"]   = accuracy
    metrics["precision"]  = precision
    metrics["recall"]     = recall
    metrics["f1"]         = f1
    metrics["confMatrix"] = confMatrix
    if 'opt' in approach:
        file = open(f"{outWD}/svc_optimization.log", "a")
        file.write(f"    {metrics}\n\n")
        file.close()
    return metrics


################################################################################
# --             Utilize Word Embeddings with a Neural Network              -- #
################################################################################
def word2vec_approach(corpusData, labelsData, params):

    # -- Set the parameters to configure the classifier and data model
    vector_size         = params["vector_size"]
    window_size         = params["window_size"]
    sg                  = params["sg"]
    hidden_layers       = params["hidden_layers"]
    neurons             = params["neurons"]
    activation_function = params["activation_function"]
    batch_size          = params["batch_size"]
    optimizer           = params["optimizer"]
    epochs              = params["epochs"]
    if 'opt' in approach:
        file = open(f"{outWD}/word2vec_optimization.log", "a")
        file.write(f"vector_size: {vector_size} - ")
        file.write(f"window_size: {window_size} - ")
        file.write(f"sg: {sg} - ")
        file.write(f"hidden_layers: {hidden_layers} - ")
        file.write(f"neurons: {neurons} - ")
        file.write(f"activation_function: {activation_function} - ")
        file.write(f"batch_size: {batch_size} - ")
        file.write(f"optimizer: {optimizer} - ")
        file.write(f"epochs: {epochs}\n"   )
        file.close()

    # -- Tokenize the paragraphs
    tokenizedParagraphs = [word_tokenize(review) for review in corpusData]

    # -- Train Word2Vec model on the tokenized data
    dataModel = Word2Vec(tokenizedParagraphs,
        vector_size = vector_size,
        window      = window_size,
        min_count   = 1,
        sg          = sg
    )

    # -- Save the trained data model for future use
    dataModel.save(
        f"{inWD}/custom_word2vec.model"
    )
    
    # -- Convert the data model to vector
    vectorizedData = [
        paragraph_to_vector(review, dataModel) for review in tokenizedParagraphs
    ]

    # -- Split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        vectorizedData, labelsData, test_size = 0.2, random_state = 42
    )
    X_train = tf.convert_to_tensor(X_train, dtype = tf.float32)
    X_test  = tf.convert_to_tensor(X_test , dtype = tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype = tf.float32)
    y_test  = tf.convert_to_tensor(y_test , dtype = tf.float32)
    
    # -- Build a simple sequential neural network
    classifier = Sequential()
    classifier.add(Flatten(input_shape = (X_train.shape[1],)))                  # Input layer sized to the input vector size
    # -- Add hidden layers
    for layer in range(hidden_layers):
        classifier.add(Dense(neurons, activation = activation_function))
    classifier.add(Dense(1  , activation = 'sigmoid'))                          # Output layer, single unit, as the classification is binary, sigmoid is used for binary classification

    # -- Compile the classifier with appropriate settings
    classifier.compile(
        optimizer = optimizer,
        loss      = 'binary_crossentropy',
        metrics   = ['accuracy']
    )

    # -- Train the classifier
    classifier.fit(X_train, y_train,
        epochs          = epochs,
        batch_size      = batch_size,
        validation_data = (X_test, y_test)
    )

    # -- Compute the metrics
    loss, accuracy = classifier.evaluate(X_test, y_test)
    y_pred         = classifier.predict(X_test)
    precision      = precision_score(y_test, y_pred > 0.5)
    recall         = recall_score(y_test, y_pred > 0.5)
    f1             = f1_score(y_test, y_pred > 0.5)
    confMatrix     = confusion_matrix(y_test, y_pred > 0.5)
    
    # -- Return the results
    metrics               = {}
    metrics["accuracy"]   = accuracy
    metrics["precision"]  = precision
    metrics["recall"]     = recall
    metrics["f1"]         = f1
    metrics["confMatrix"] = confMatrix
    if 'opt' in approach:
        file = open(f"{outWD}/word2vec_optimization.log", "a")
        file.write(f"    {metrics}\n\n")
        file.close()
    return metrics


################################################################################
# --                            Show the results                            -- #
################################################################################
def report_results(metrics):
    tn = metrics["confMatrix"][0][0]
    fp = metrics["confMatrix"][0][1]
    tp = metrics["confMatrix"][1][1]
    fn = metrics["confMatrix"][1][0]

    # -- Get the current date and time for Costa Rica
    TZ      = pytz.timezone('America/Costa_Rica')
    utcNow  = datetime.utcnow()
    sCRTime = utcNow.replace(tzinfo = pytz.utc).astimezone(TZ)
    sCRTime = sCRTime.strftime('%Y-%m-%d %H:%M:%S %Z')
    
    # -- Record the accuracy in a log
    file = open(
        f"{outWD}/results_{approach}.log", 'a'
    )
    file.write(f"CR time: {sCRTime}\n")
    file.write( "RESULTS:\n"                                            )
    file.write( "----\n"                                                )
    file.write(f"    Accuracy  [(TP+TN)/ALL]: {metrics['accuracy']}\n"  )
    file.write(f"    Precision [TP/(TP+FP)] : {metrics['precision']}\n" )
    file.write(f"    Recall    [TP/(TP+FN)] : {metrics['recall']}\n"    )
    file.write(f"    F1        [2*P*R/(P+R)]: {metrics['f1']}\n"        )
    file.write( "----\n"                                                )
    file.write(f"    Confusion matrix:\n"                               )
    file.write(f"        Correctly selected (TP):       {tp}\n"         )
    file.write(f"        Incorrectly selected (FP):     {fp}\n"         )
    file.write(f"        Correctly not selected (TN):   {tn}\n"         )
    file.write(f"        Incorrectly not selected (FN): {fn}\n"         )
    file.close()


    # -- Plot the confusion matrix
    if printCM:
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            data        = [[tp,fp],[fn,tn]],
            annot       = True,
            fmt         = 'd',
            cmap        = 'Blues', 
            xticklabels = [
                'Reviewer is impactful',
                'Reviewer is not impactful'
            ],
            yticklabels = [
                'Reviewer is impactful',
                'Reviewer is not impactful'
            ]
        )

        # -- Assign the plot labels
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Confusion Matrix Heatmap')

        # -- Move the X-axis to the top
        ax = plt.gca()
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')

        # -- Show the plot
        plt.show()


################################################################################
# --                        Target function for SMBO                        -- #
################################################################################
def target_function(args):
    results = {}
    if "svc" in approach:
        results = svm_approach(corpus, labels, params = args)
    elif "word2vec" in approach:
        results = word2vec_approach(corpus, labels, params = args)
    return 1/(results["accuracy"] + 1)


################################################################################
# --                            Main of the code                            -- #
################################################################################
if __name__ == "__main__":
    # -- Select the approach to take
    approach = 'word2vec'                                                       # Format -> [svc or word2vec]{_printCM}{_opt_{# of trials}}
    printCM  = False
    if len(sys.argv) >= 2:
        approach = sys.argv[1].lower()
        if 'printcm' in approach:
            printCM = True
        if 'opt' in approach:
            optTrials = int(approach.split('opt_')[1])
        if 'svc' not in approach and 'word2vec' not in approach:
            print("Classifier not recognized")
            exit(1)

    corpus, labels = generate_corpus_and_labels(dataset)

    if 'svc' in approach:
        if 'opt' in approach:
            best = fmin(
                target_function,
                svc_search_space,
                algo = tpe.suggest,
                max_evals = optTrials
            )
            file = open(f"{outWD}/svc_optimization.log", 'a')
            file.write(f"Best params for SVC after {optTrials} iterations\n")
            file.write(f"{best}\n\n")
            file.close()
        else:
            file = open(f"{outWD}/results_{approach}.log", 'a')
            file.write(f"\n\n{svc_default_params}\n")
            file.close()
            metrics = svm_approach(corpus, labels, params = svc_default_params)
            report_results(metrics)
    elif 'word2vec' in approach:
        if 'opt' in approach:
            best = fmin(
                target_function,
                nn_search_space,
                algo = tpe.suggest,
                max_evals = optTrials
            )
            file = open(f"{outWD}/word2vec_optimization.log", 'a')
            file.write(
                f"Best params for word2vec after {optTrials} iterations\n"
            )
            file.write(f"{best}\n\n")
            file.close()
        else:
            file = open(f"{outWD}/results_{approach}.log", 'a')
            file.write(f"\n\n{nn_default_params}\n")
            file.close()
            metrics = word2vec_approach(
                corpus, labels, params = nn_default_params
            )
            report_results(metrics)