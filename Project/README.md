# Evaluation and Comparison of the Accuracy of Two Classification Approaches on the Usefulness of User Reviews


## (Short) Introduction

In the branch of NLP there are multiple investigations on topics like sentiment analysis, LLM, and pattern recognition, as part of the last one mentioned comes into play the authorship identification, which is a branch that searches for patterns in the text that each author follows along the way. This is a deeply investigated branch into NLP, however, there is another sub-topic around authorship identification, which may be taken by responding _Is there any way to detect patterns that a group of people having something in common follow?_.

There are multiple things groups of people may have in common, that can lead to similar speeches, like education level, place where they live, country of birth, and many others. Reason why an investigation question came up related to pattern recognition in speech, it aims to determine if it is possible to detect a pattern that people with insightful reviews have in common, based on a portion of Yelp dataset. However, this can be extended to many other areas, like detecting the level of education about a topic, based only on people's answers to certain questions related to that topic, these are specific areas of pattern recognition in NLP that may be interesting to explore.

## About the Project

The present project is focused on the natural language processing section of identifying authorship, with the difference that in this case, we are looking for identification of the impact of the review, regardless of the author, to see if the most impactful reviewers in the website follow a common pattern or not. This is an important approach, that may allow to identify potentially useless reviews, or even fake reviews, depending on the approach it is used for.

For the exercise I am using Yelp Dataset (see https://www.yelp.com/dataset/), which provides a high volume of reviews with their authors, and some relevant data about each of the authors, businesses, reviews, and so.

Since the data from the author impact and their reviews is split into two different datasets (user.json and review.json), I had to create a tailored dataset, containing the relevant data from the review, which is the review text, review_id, and user_id, and the relevant data from the user, which is yelping_since for longevity, review_count for experience, and useful for impact analysis.

The tailored dataset looks this way:


    dataset = [
        {
            'text'     : "This is the text of the first review",
            'review_id': "This is the review ID",
            'user_id'  : "This is the user ID",
            'user_data': {
                'yelping_since': "This is the yelping since date",
                'review_count' : Integer representing the total number of reviews,
                'useful'       : Integer representing the total useful reviews
            }
        },
        ...
        ...
        ...
        {
            'text'     : "This is the text of the Nth review",
            'review_id': "This is the review ID",
            'user_id'  : "This is the user ID",
            'user_data': {
                'yelping_since': "This is the yelping since date",
                'review_count' : Integer representing the total number of reviews,
                'useful'       : Integer representing the total useful reviews
            }
        }
    ]


Generating this tailored dataset was a huge challenge, since importing the review and user datasets to any Python script using an 8 GB RAM computer was not possible without absorbing the whole RAM, and needing even more memory (review.json weights 5.35 GB, while user.json weights 3.37 GB). Therefore, a script using grep and sed CLI commands was needed to overcome this issue, with the problem that internally they use system calls, generating a comparable slower process than having the datasets included in the script, but the way to go if you have a computer with a lack of resources to deal with the size of these datasets.

# Support Vector Machine Classifier with Bigrams Data Model

## Classifier Tuning

For the classifier tuning, multiple parameters were considered with the combination of all of them when applicable, these are the following.

### Parameters Applicable to All Kernels:
* __kernel__:
  
        Description: The kernel determines the way the data is split.
        Range: {Linear, Polynomial, RBF, Sigmoid}
* __C__:

        Description: The regularization parameter C is also one of
                     the most critical parameters to tune, since it
                     controls the trade-off between maximizing the
                     margin and minimizing classification error.
        Range:       { x | 0.1 ≤ x < 1   , ∃ k ∈ ℤ (x = 0.1k) } ∪
                     { x | 1   ≤ x < 100 , ∃ k ∈ ℤ (x = 10k ) } ∪
                     { x | 100 ≤ x ≤ 1000, ∃ k ∈ ℤ (x = 100k) }

### Special Parameters:
* __gamma__:

        Description: This parameter is applicable for RBF,
                     Polynomial, and Sigmoid kernels, it controls
                     the shape of the decision boundary.
        Range:       { x | 0.1 ≤ x < 1   , ∃ k ∈ ℤ (x = 0.1k) } ∪
                     { x | 1   ≤ x < 100 , ∃ k ∈ ℤ (x = 10k ) } ∪
                     { x | 100 ≤ x ≤ 1000, ∃ k ∈ ℤ (x = 100k) }
* __decision_function_shape__:

        Description: This parameter applies only to the Linear kernel,
                     specifically, it determines how the decision
                     function is calculated for multi-class
                     classification problems.
        Range:       {'ovr', 'ovo'}
* __degree__:

        Description: This parameter applies only to the Polynomial
                     kernel, it determines the degree of the polynomial
                     equation splitting the data.
        Range:       { x | x ∈ ℤ, 2 ≤ x ≤ 9 }

### Notes:
* OVR (one versus the rest) means that scikit-learn will train multiple binary classifiers, and then, each binary classifier is trained to distinguish one class from all the others.
* OVO (one versus one) means that scikit-learn will train binary classifiers for every pair of classes, and then, each binary classifier will be trained to distinguish between two specific classes.

## Results

There were taken a total of 22475, where 17980 reviews for training purposes, and 4495 for testing purposes, and with this very short sample, it was possible to create a classifier using a _Support Vector Classifier_, and after some characterization trials to detect that using _Radial Basis Function_, with C equals to 10 and gamma equals to 0.1, can guarantee an accuracy of about 60.8%, which means, there seems to be a pattern that all the impactful reviewers follow when adding a review, and the classifier is being capable to detect it correctly 60.8% of the times.

    Accuracy  [(TP+TN)/ALL]: 0.6075639599555062
    Precision [TP/(TP+FP)] : 0.558104550582926
    Recall    [TP/(TP+FN)] : 0.7158707187650748
    F1        [2*P*R/(P+R)]: 0.6272189349112426

See the confusion matrix below for reference.

<p align="center">
  <img src="https://github.com/trejkev/natural_language_processing_course/assets/18760154/9c359d76-36e1-4cc9-9594-28d1739f7cc9" width="600" />
</p>

# Sequential Neural Network Classifier with Word Embeddings Data Model

To test a second approach to represent the data, and to classify the data, a neural network with word vectors approach was used.

## Classifier Tuning

For the enhancement of the accuracy, both the sequential neural network and the word vectors were tuned together. There were multiple parameters considered along the way, these are mentioned below.

### Word Vector (Word Embeddings) Parameters Considered

* __vector_size__:

        Description: When using word embeddings, every set word is
                     converted into a vector. Therefore, the vector
                     size refers to the number of dimensions in the
                     numerical vector used to represent a word.
         Range:       {50, 100, 200}
* __window_size__:

        Description: It refers to the number of surrounding words or
                     context words considered when training the
                     embeddings for a specific target word. It
                     determines how much contextual information is
                     taken into account when creating word vectors.
         Range:      {5, 10, 15, 20}
* __sg__:

        Description: Skip-gram is an algorithm used in Word2Vec to
                     predict the surrounding words based on a given
                     target word. On the opposite, a continuous bag
                     of words (CBOW) algorithm guesses the target word
                     based on the surrounding words.
         Range:      {0,1} -> 1 = Skip-Gram, 0 = CBOW

### Sequential Neural Network Parameters Considered

* __Number of hidden layers__:

        Description: Number of hidden layers of the neural network.
        Range:       {1, 2, 3, 4, 5, 6}
* __Number of neurons per hidden layer__:

        Description: Number of neurons that each hidden layer will have.
        Range:       {64, 128, 256}
* __Activation function__:

        Description: Activation function for each of the neurons.
        Range:       {relu, tanh, sigmoid}
* __batch_size__:

        Description: It controls how many data samples are used to
                     calculate the gradient and update the model's
                     parameters in each training iteration.
        Range:       {32, 64, 128}
* __optimizer__:

        Description: The optimizer determines how the model's weights
                     and biases are updated during training to minimize
                     the specified loss function.
        Range:       {adam, sgd, rmsprop}
* __epochs__:

        Description: The epochs parameter determines the number of
                     iterations that will occur to train the model.
        Range:       {10, 50, 100, 200, 300, 600}

## Results

There were taken a total of 22475, where 17980 reviews for training purposes, and 4495 for testing purposes, and with this very short sample, it was possible to create a classifier using a _Sequential Neural Network_, by the hand of the use of word vectors (aka _Word Embeddings_), and after some characterization trials to detect that using 4 hidden layers of 128 neurons each, and with relu activation function for the neural network architecture, sgd optimizer, a batch size of 32, and 100 epochs for the compilation-training stages, and a vector size of 200, with a window size of 5, and skip-gram enabled for the word vectors, can guarantee an accuracy of about 65%, which means, there seems to be a pattern that all the impactful reviewers follow when adding a review, and the classifier is being capable to detect it correctly 65% of the times.

    Accuracy  [(TP+TN)/ALL]: 0.6509454846382141
    Precision [TP/(TP+FP)] : 0.6392265193370166
    Recall    [TP/(TP+FN)] : 0.5581283164495899
    F1        [2*P*R/(P+R)]: 0.595930981200103

See the confusion matrix below for reference.

<p align="center">
  <img src="https://github.com/trejkev/natural_language_processing_course/assets/18760154/a5a6c875-7848-4cde-98d8-9542f106a2ac" width="600" />
</p>

# How to Use the Code

## Using the input_data_processor.py File

To use the code, at first, you will have to create the dataset, or use the already provided by the repository. In case you want to create it you will have to use the _input_data_processor.py_, which receives two different parameters, the first is the action to perform, and the second is the number of reviews to consider, or the starting point to continue generating the reviews data, based on the relevant_data.py file, see the following guidance for more information.

     This script intends to pre-process the data that will be needed by
     the AI-ML classifier that will be used to predict data about the
     users based on their reviews.\n
     To use this script, two parameters must be sent:
         - [1] Action to perform:
             - formatRevToListDict:     Formats the review json file to
                                        be a list of dictionaries. It
                                        will generate a _mod.json file
                                        that you must rename after
                                        verifying its correctness.
             - formatUsrListToDict:     Formats the user json file to be
                                        a list of dictionaries. It
                                        will generate a _mod.json file
                                        that you must rename after
                                        verifying its correctness.
             - generateRevRelevantData: Generates a rev_relevant_data.py
                                        file containing the review
                                        relevant data.
             - generateRelDataWithUsr:  Once generateRevRelevantData has
                                        been done, creates a
                                        relevant_data.py file containing
                                        a list of dictionaries including
                                        user data.
         - [2] Number of reviews to consider, 'all' considers all the
               reviews. Available only when using
               generateRevRelevantData.
         - [2] Add user data starting from this line number, based on
               the relevant_data.py line number - 1. Available only when
               using generateRelDataWithUsr.

## Using the review_classifier.py file

Once you have your dataset, you need to rename it to _relevant_dataset.py_ and place it into _input_ directory, with this, you are ready to use the script. There are three different things you may consider, first, the classifier you want to use, second, if you want to plot the confusion matrix, and third, if you want to run a simple iteration, or if you want to run an optimization iteration.

The command is quite simple, just use {svc or nn}{_printCM}{\_opt\_{# of trials}}, consider that none of the parameters are mandatory, there is a default scenario, running a simple run with word embeddings and without plotting the confusion matrix. If you desire to run a different scenario, see below the control inputs you are allowed to use.
1. {svc or nn}: This directive serves as a classification approach selector, it lets you choose between a Support Vector Machine (with bigrams data model) or a Neural Network (with word vectors data model).
2. {_printCM}: Plotting the confusion matrix is a blocking task, which means that while the plot is open, the program is frozen, this option is disabled by default, therefore, if you want to enable it, you will need to add it to the input param string.
3. {\_opt\_{# of trials}}: When you want to run an optimization instead of a simple run, you will have to add 'opt' to the input param, followed by the number of iterations you want. Since you need to run the script multiple times, it is recommended to disable the use of plotting the confusion matrix, otherwise the program will freeze for every iteration.

In the end, some examples of usage may be the ones shown below.

        python3 review_classifier.py             # Will run the default scenario
        python3 review_classifier.py svc         # Will run a simple iteration with a SVM
        python3 review_classifier.py nn_printCM  # Will run a simple iteration with a NN and printing the confusion matrix
        python3 review_classifier.py svc_opt_500 # Will run an optimization routine for 500 iterations using a SVM

