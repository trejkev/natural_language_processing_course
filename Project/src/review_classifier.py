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

# -- Libraries to use SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection         import train_test_split
from sklearn.svm                     import SVC
from sklearn.metrics                 import accuracy_score
from sklearn.metrics                 import precision_score
from sklearn.metrics                 import recall_score
from sklearn.metrics                 import f1_score
from sklearn.metrics                 import confusion_matrix

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

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt', quiet = True, raise_on_error = True)                     # Necessary for nltk tokenization

# -- Libraries for plotting the confusion matrix
import matplotlib.pyplot as plt
import seaborn           as sns

# -- Method to convert the pharagraphs to vectors
def paragraph_to_vector(paragraph, model):
    vector = [model.wv[word] for word in paragraph if word in model.wv]
    return np.mean(vector, axis=0) if vector else np.zeros(model.vector_size)


################################################################################
# --                           Import the dataset                           -- #
################################################################################
import sys, os
sys.path.append(f"{os.getcwd().replace('src','input')}")
from relevant_dataset import dataset


################################################################################
# --          Pre-process the dataset to get the corpus and labels          -- #
################################################################################
corpusData = []
labelsData = []
for line in dataset:
    revCount = line["user_data"]["review_count"]
    useful   = line["user_data"]["useful"]
    labelsData.append(1 if useful / ( revCount + 1) >= 0.9 else 0)
    corpusData.append(line["text"])


################################################################################
# --                  Utilize a Support Vector Classifier                   -- #
################################################################################
approach = 'SVC'
if len(sys.argv) == 2 and sys.argv[1].lower() == 'word2vec':
    print("Will use word embeddings with self-training")
    approach = 'word2vec'
else:
    print("Defaulted to SVC")


if approach == 'SVC':
    # -- Create the bigram features
    vectorizer   = CountVectorizer(ngram_range=(2, 2))
    sparseMatrix = vectorizer.fit_transform(corpusData)

    # -- Split the data intro training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        sparseMatrix, labelsData, test_size = 0.2, random_state = 42
    )

    # -- Train the classifier
    classifier = SVC(kernel = 'rbf', C = 10, gamma = 0.1)
    classifier.fit(X_train, y_train)

    # -- Make predictions
    y_pred = classifier.predict(X_test)
    
    # -- Compute the metrics
    accuracy   = accuracy_score(y_test, y_pred)
    precision  = precision_score(y_test, y_pred)
    recall     = recall_score(y_test, y_pred)
    f1         = f1_score(y_test, y_pred)
    confMatrix = confusion_matrix(y_test, y_pred)


################################################################################
# --             Utilize Word Embeddings with a Neural Network              -- #
################################################################################
elif approach == 'word2vec':
    # -- Tokenize the paragraphs
    tokenizedParagraphs = [word_tokenize(review) for review in corpusData]

    # -- Train Word2Vec model on the tokenized data
    dataModel = Word2Vec(
        tokenizedParagraphs, vector_size = 100, window = 5, min_count = 1, sg = 0
    )

    # -- Save the trained data model for future use
    dataModel.save(
        f"{os.getcwd().replace('src','input')}/custom_word2vec.model"
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
    
    # -- Build a simple feedforward neural network
    classifier = Sequential()
    classifier.add(Flatten(input_shape = (X_train.shape[1],)))                  # Input layer sized to the input vector size
    classifier.add(Dense(256, activation = 'relu'))                             # 300 hidden layers, relu helps to detect complex patterns
    classifier.add(Dense(1  , activation = 'sigmoid'))                          # Output layer, single unit, as the classification is binary, sigmoid is used for binary classification

    # -- Compile the classifier with appropriate settings
    classifier.compile(
        optimizer = 'adam',
        loss      = 'binary_crossentropy',
        metrics   = ['accuracy']
    )

    # -- Train the classifier
    classifier.fit(X_train, y_train,
        epochs          = 100,
        batch_size      = 20,
        validation_data = (X_test, y_test)
    )

    # -- Compute the metrics
    loss, accuracy = classifier.evaluate(X_test, y_test)
    y_pred         = classifier.predict(X_test)
    precision      = precision_score(y_test, y_pred > 0.5)
    recall         = recall_score(y_test, y_pred > 0.5)
    f1             = f1_score(y_test, y_pred > 0.5)
    confMatrix     = confusion_matrix(y_test, y_pred > 0.5)


################################################################################
# --                           Evaluate the model                           -- #
################################################################################
tn = confMatrix[0][0]
fp = confMatrix[0][1]
tp = confMatrix[1][1]
fn = confMatrix[1][0]


################################################################################
# --                            Show the results                            -- #
################################################################################
print( "RESULTS:"                                    )
print( "----"                                        )
print(f"    Total train samples: {len(y_train)}"     )
print(f"    Total test samples:  {len(y_test)}"      )
print( "----"                                        )
print(f"    Accuracy  [(TP+TN)/ALL]: {accuracy}"     )
print(f"    Precision [TP/(TP+FP)] : {precision}"    )
print(f"    Recall    [TP/(TP+FN)] : {recall}"       )
print(f"    F1        [2*P*R/(P+R)]: {f1}"           )
print( "----"                                        )
print(f"    Confusion matrix:"                       )
print(f"        Correctly selected (TP):       {tp}" )
print(f"        Incorrectly selected (FP):     {fp}" )
print(f"        Correctly not selected (TN):   {tn}" )
print(f"        Incorrectly not selected (FN): {fn}" )


################################################################################
# --                        Plot the confusion matrix                       -- #
################################################################################
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
