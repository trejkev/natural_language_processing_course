# Relevant data from review is:
#     text
#     review_id
#     user_id
# Relevant data from user is:
#     yelping_since
#     review_count
#     useful
#     user_id
#
# The idea is to guess if the review is useful or not with
# this data. Therefore, a dictionary for training and
# validation will be created.

import json, os, nltk, ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')

# -- Global variables
relevantInputData = []
inputDirectory    = os.getcwd().replace('src','input')

# datasetFile   = open(f"{inputDirectory}/yelp_academic_dataset_review.json")
datasetFile   = open(f"{inputDirectory}/TEST_review.json")
reviewDataset = json.load(datasetFile)

# -- Collect relevant data from the review dataset
for data in reviewDataset:
    # Generate bigrams without None values 
    tokens = nltk.word_tokenize(data["text"].lower())
    bigrams = list(
        nltk.ngrams(tokens, 2,
            pad_left         = True,
            pad_right        = True,
            left_pad_symbol  = None,
            right_pad_symbol = None
        )
    )[1:-1]
    relevantInputData.append(
        {
            "text"     : data["text"].lower(),
            "bigrams"  : bigrams,
            "review_id": data["review_id"],
            "user_id"  : data["user_id"]
        }
    )
del reviewDataset                                                               # Must be removed because it is extremely large
datasetFile.close()

# -- Collect all users relevant data
