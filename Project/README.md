# Review Impact Classifier

The present project is focused on the natural language processing section of identifying authorship, with the difference that in this case, we are looking for identification of the impact of the review, regardless of the author, to see if the most impactful reviewers in the website follow a common pattern or not.

For the exercise I am using Yelp Dataset (see https://www.yelp.com/dataset/), which provides a high volume of reviews with their authors, and some relevant data about each of the authors, businesses, reviews, and so.

Since the data from the author impact and their reviews is split into two different datasets (user.json and review.json), I had to create a tailored dataset, containing the relevant data from the review, which is the review text, review_id, and user_id, and the relevant data from the user, which is yelping_since for longevity, review_count for experience, and useful for impact analysis.

The tailored dataset looks this way:

___
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
___

Classifier Tuning
----
For the classifier tuning, multiple parameters were considered with the combination of all of them when applicable, these are the following.

Parameters applicable to all kernels:
* __kernel__: The kernel function was varied between Linear, Polynomial, RBF, and Sigmoid, since these determine the way the data is split.
* __C__: The regularization parameter C is also one of the most critical parameters to tune, since it controls the trade-off between maximizing the margin and minimizing classification error. It was varied from very short numbers to considerably large numbers, to be more precise this is a simple representation of the tested values: [0.1 to 1, step of 0.1] + [1 to 100, step of 10] + [100 to 1000, step of 100]

Special parameters:
* __gamma__: This parameter is applicable for RBF, Polynomial, and Sigmoid kernels, it controls the shape of the decision boundary. It was varied from very short numbers to considerably large numbers, to be more precise this is a simple representation of the tested values: [0.1 to 1, step of 0.1] + [1 to 100, step of 10] + [100 to 1000, step of 100]
* __decision_function_shape__: This parameter applies only to the Linear kernel, specifically it determines how the decision function is calculated for multi-class classification problems. It was varied between 'ovr' and 'ovo'.
* __degree__: This parameter applies only to the Polynomial kernel, it determines the degree of the polynomial equation splitting the data. It was varied from 2 to 9, in steps of 1.

Notes:
* OVR (one versus the rest) means that scikit-learn will train multiple binary classifiers, and then, each binary classifier is trained to distinguish one class from all the others.
* OVO (one versus one) means that scikit-learn will train binary classifiers for every pair of classes, and then, each binary classifier will be trained to distinguish between two specific classes.
  
I have taken a total of 10361 reviews for training purposes, and 2591 for testing purposes, and with this very short sample, I was able to create a classifier using a _Support Vector Classifier_, and after some characterization trials to detect that using _Radial Basis Function_, with C equals to 10 and gamma equals to 0.1, can guarantee an accuracy of about 72%, which means, there is a pattern that all the impactful reviewers follow when adding a review, and the classifier is being capable to detect it 72% of the times.

Results
----
    Accuracy  [(TP+TN)/ALL]: 0.7186414511771517
    Precision [TP/(TP+FP)] : 0.7239070500196928
    Recall    [TP/(TP+FN)] : 0.984994640943194
    F1        [2*P*R/(P+R)]: 0.8345062429057889

See the confusion matrix below for reference.


<p align="center">
  <img src="https://github.com/trejkev/natural_language_processing_course/assets/18760154/ae69241d-f22a-42a1-afa9-a1fdcf3923bf" width="800" />
</p>

## How to Use the Code

### Using the input_data_processor.py file

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

### Using the review_classifier.py file

Once you have your dataset, you need to rename it to _relevant_dataset.py_ and place it into _input_ directory, and finally run the command _python3 review_classifier.py_ to get the classification results.
