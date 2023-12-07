# Evaluation and Comparison of the Accuracy of Two Classification Approaches on the Usefulness of User Reviews


## Introduction

In the realm of Natural Language Processing (NLP), various studies focus on significant topics such as sentiment analysis, Large Language Models (LLM), and pattern recognition. Within the realm of pattern recognition, a notable subfield is authorship identification. This area involves investigating patterns within text that are distinctive to individual authors.

While authorship identification is a well-explored aspect of NLP, there is an intriguing sub-topic within this field. This sub-topic revolves around discerning patterns shared by a collective group of individuals. The central question guiding this exploration is: _"Is there a method to identify patterns common to a group of people who share certain characteristics?"_.

This line of inquiry extends beyond individual authorship to explore shared linguistic traits or patterns among a defined group. It broadens the scope of traditional authorship identification, opening avenues for understanding language dynamics within specific communities or demographic groups.

In the investigation into pattern recognition within group speech, particularly focusing on usefulness of reviews and datasets like Yelp, the research aims to determine if identifiable patterns exist among individuals consistently providing insightful reviews. This line of inquiry provides valuable insights into shared characteristics within specific groups.

Furthermore, the applicability of this research extends beyond Yelp reviews to various domains. For instance, it could be applied to assess the level of education on a specific topic based solely on individuals' responses to related questions. This expansion highlights the versatility of pattern recognition in NLP, offering valuable insights into educational levels, interests, or expertise within distinct communities.

Exploring these specific areas of pattern recognition enhances the understanding of language dynamics and offers practical applications for tailoring content, services, or educational resources to the specific needs of diverse audiences. This aligns with the broader goal of leveraging NLP for more nuanced and personalized interactions within varied communities.

## About the Project

The current project zeroes in on the natural language processing aspect of authorship identification. However, the unique twist here is that we are not focused on identifying the specific authors; instead, we are aiming to discern patterns that indicate the impact of reviews, regardless of the author. The goal is to determine if the most impactful reviewers on the website share common patterns in their language use.

This approach is crucial as it has the potential to identify reviews that may lack impact or, in extreme cases, might be fake. By examining common patterns among impactful reviewers, we can develop a tool to assess the usefulness and authenticity of reviews on the website. This approach holds significant value in improving the reliability and quality of the information available to users.

Using the Yelp Dataset for this exercise is a strategic choice, given its wealth of reviews along with associated author and business data. The dataset, available at https://www.yelp.com/dataset/, provides a substantial volume of reviews, each attributed to specific authors. Additionally, it includes pertinent information about authors, businesses, and reviews, offering a comprehensive foundation for your natural language processing and authorship identification project.

By leveraging this dataset, you can explore patterns in language usage to identify impactful reviewers, transcending the need to pinpoint individual authors. The dataset's richness allows for a nuanced analysis that goes beyond mere author identification, enabling you to assess the broader impact of reviews on the platform. This approach has the potential to uncover valuable insights, ranging from the reliability of reviews to the detection of potentially misleading or fake feedback.

Creating a tailored dataset that combines relevant information from the `review.json` and `user.json` datasets is a thoughtful approach. Combining the necessary data, such as review text, review ID, user ID, yelping_since, review_count, and useful for impact analysis, provides a consolidated dataset that can yield comprehensive insights into reviewer impact.

By considering the longevity of users on Yelp (`yelping_since`), their experience level (`review_count`), and the impact of their reviews (`useful`), you're constructing a dataset that allows for a nuanced analysis. The temporal aspect of Yelping since provides insights into the reviewer's long-term engagement, while the review_count gives an indication of their overall experience. The parameter useful provides a metric for understanding the impact of their reviews.

This tailored dataset appears well-suited to facilitate a detailed examination of patterns related to the impact of reviews, potentially uncovering commonalities among impactful reviewers on Yelp. This can be instrumental in identifying trends and patterns that contribute to the effectiveness of reviews on the platform.

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


Handling large datasets, especially when memory constraints are present, can indeed be a challenging task. Your approach of using CLI commands like `grep` and `sed` to preprocess and filter the data is a pragmatic solution given the limitations of available memory. Although it might introduce some computational overhead due to system calls, it's a practical workaround to efficiently manage and process large datasets without overwhelming the system's memory.

Optimizing resource usage becomes crucial when dealing with datasets of this magnitude, and your use of command-line tools showcases adaptability in addressing these challenges. The balance between available resources and the computational efficiency required for dataset manipulation is a common consideration in data science and analysis.

This approach allows you to work with the data without having to load the entire dataset into memory, making it feasible to perform the necessary preprocessing and filtering steps for your tailored dataset. It's a pragmatic solution that demonstrates resourceful problem-solving in the face of hardware constraints.

# Support Vector Machine Classifier with Bigrams Data Model

## Classifier Tuning

For the classifier tuning, multiple parameters were considered with the combination of all of them when applicable, these are the following.

### Parameters Applicable to All Kernels:
* __kernel__:
  
        Description: The kernel determines the way the data is split.
        Range:       {Linear, Polynomial, RBF, Sigmoid}
* __C__:

        Description: The regularization parameter C is also one of
                     the most critical parameters to tune, since it
                     controls the trade-off between maximizing the
                     margin and minimizing classification error.
        Range:       {10/n | 1 ≤ n < 1000, n ∈ Z}

### Special Parameters:
* __gamma__:

        Description: This parameter is applicable for RBF,
                     Polynomial, and Sigmoid kernels, it controls
                     the shape of the decision boundary.
        Range:       {10/n | 1 ≤ n < 1000, n ∈ Z} U {’scale’}
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

Creating a classifier with a Support Vector Classifier (SVC) based on a sample of 22,475 reviews is an impressive feat, especially given the constraints you faced with large datasets. The division into 17,980 training reviews and 4,495 testing reviews allows for a robust evaluation of the model's performance.

The use of a Radial Basis Function (RBF) kernel with specific parameters (C=10, gamma=0.1) showcases the effectiveness of the classifier. Achieving an accuracy of approximately 60.8% suggests that there is a discernible pattern among impactful reviewers when adding a review, and the classifier is successfully identifying this pattern more than half of the time.

It's worth noting that a 60.8% accuracy rate is meaningful, especially considering the complexity and variability in language use. The classifier's ability to capture patterns in impactful reviews is promising, and further optimizations or feature engineering could potentially enhance its performance.

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
         Range:      {5 ≤ n < 21, n ∈ Z}
* __sg__:

        Description: Training algorithm, 1 for skip-gram; otherwise
                     CBOW.
         Range:      {0,1}

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
        Range:       {32 ≤ n < 129, n ∈ Z}
* __optimizer__:

        Description: The optimizer determines how the model's weights
                     and biases are updated during training to minimize
                     the specified loss function.
        Range:       {adam, sgd, rmsprop}
* __epochs__:

        Description: The epochs parameter determines the number of
                     iterations that will occur to train the model.
        Range:       {10 ≤ n < 200, n ∈ Z}

## Results

Creating a classifier using a Sequential Neural Network (SNN) with word vectors (Word Embeddings) is a sophisticated approach that leverages the power of deep learning for your analysis. Working with a total of 22,475 reviews and dividing them into 17,980 for training and 4,495 for testing demonstrates a robust methodology for model evaluation.

The specific architecture of the neural network, with four hidden layers of 128 neurons each, using the relu activation function, showcases a thoughtful design. Additionally, employing the stochastic gradient descent (sgd) optimizer, a batch size of 32, and 100 epochs during the compilation-training stages represents a well-considered set of hyperparameters.

The use of word vectors with a vector size of 200, a window size of 5, and skip-gram enabled for word vectors adds a level of semantic understanding to the model. Achieving an accuracy of around 65% implies that the neural network is effectively capturing patterns in impactful reviews, outperforming the Support Vector Classifier.

This work not only demonstrates the power of neural networks for natural language processing but also suggests that there are discernible patterns among impactful reviewers, and the classifier is successfully identifying them with a commendable accuracy of 65%. Further experimentation or fine-tuning could potentially improve the model's performance.

    Accuracy  [(TP+TN)/ALL]: 0.6509454846382141
    Precision [TP/(TP+FP)] : 0.6392265193370166
    Recall    [TP/(TP+FN)] : 0.5581283164495899
    F1        [2*P*R/(P+R)]: 0.595930981200103

See the confusion matrix below for reference.

<p align="center">
  <img src="https://github.com/trejkev/natural_language_processing_course/assets/18760154/a5a6c875-7848-4cde-98d8-9542f106a2ac" width="600" />
</p>

### Comparison



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

## Using the statistical_analyzer.py file

Once you have collected the results from each run into _results_NN.log_ and _results_SVC.log_ you will realize that the format obtained is not so useful for inferential analysis, therefore, the use of grep from shell may be the choice to take to get only the lines with the accuracy results, this can be done with the shell command _grep "Accuracy" results_SVC.log > accuracies_SVC.log_, with this, you will create a file named _accuracies_SVC.log_ containing only the accuracy results, and then, you can simply execute the file, adding the number of samples to average, if the data is not behaving normally at first (taking advantage of the central limit theorem), just like _python3 statistical_analyzer.py 7_ to average 7 samples.
