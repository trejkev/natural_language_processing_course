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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection         import train_test_split
from sklearn.svm                     import SVC
from sklearn.metrics                 import accuracy_score
from sklearn.metrics                 import precision_score
from sklearn.metrics                 import recall_score
from sklearn.metrics                 import f1_score
from sklearn.metrics                 import confusion_matrix

import matplotlib.pyplot as plt
import seaborn           as sns

# -- Include path for the input data
import sys, os
sys.path.append(f"{os.getcwd().replace('src','input')}")
from relevant_dataset import dataset


################################################################################
# --          Pre-process the dataset to get the corpus and labels          -- #
################################################################################
corpusData = []
labelsData = []
counter = 0
for line in dataset:
    corpusData.append(line["text"])
    revCount = line["user_data"]["review_count"]
    useful   = line["user_data"]["useful"]
    labelsData.append(1 if useful / ( revCount + 1) >= 0.5 else 0)


################################################################################
# --                      Create the bigram  features                       -- #
################################################################################
vectorizer   = CountVectorizer(ngram_range=(2, 2))
sparseMatrix = vectorizer.fit_transform(corpusData)


################################################################################
# --             Split the data intro training and testing sets             -- #
################################################################################
X_train, X_test, y_train, y_test = train_test_split(
    sparseMatrix, labelsData, test_size = 0.2, random_state = 42
)


################################################################################
# --                          Train the classifier                          -- #
################################################################################
classifier = SVC(kernel = 'rbf', C = 10, gamma = 0.1)
classifier.fit(X_train, y_train)


################################################################################
# --                            Make predictions                            -- #
################################################################################
y_pred = classifier.predict(X_test)


################################################################################
# --                           Evaluate the model                           -- #
################################################################################
accuracy       = accuracy_score(y_test, y_pred)
precision      = precision_score(y_test, y_pred)
recall         = recall_score(y_test, y_pred)
f1             = f1_score(y_test, y_pred)
confMatrix     = confusion_matrix(y_test, y_pred)
tn             = confMatrix[0][0]
fp             = confMatrix[0][1]
tp             = confMatrix[1][1]
fn             = confMatrix[1][0]


################################################################################
# --                            Show the results                            -- #
################################################################################
print("RESULTS:"                                     )
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
print(f"        Correctly not selected (FN):   {fn}" )
print(f"        Incorrectly not selected (TN): {tn}" )


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
plt.xlabel('Actual'                 )
plt.ylabel('Predicted'              )
plt.title('Confusion Matrix Heatmap')

# -- Move the X-axis to the top
ax = plt.gca()
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')

# -- Show the plot
plt.show()
