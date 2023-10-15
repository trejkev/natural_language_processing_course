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

import json       # To deal with reviews dataset preprocessing
import os         # To get the current working directory
import sys        # To get input parameters
import nltk       # To generate bigrams
import ssl        # To install nltk dependencies
import subprocess # To generate the output dataset

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt')


################################################################################
# --                            Global variables                            -- #
################################################################################
relevantInputData = []
inputDirectory    = os.getcwd().replace('src','input')
revFileName       = "yelp_academic_dataset_review.json"
usrFileName       = "yelp_academic_dataset_user.json"


################################################################################
# --                              User manual                               -- #
################################################################################
if len(sys.argv) == 2 and sys.argv[1].lower() in ['man', 'manual']:
    print("This script intends to pre-process the data that will be needed by" )
    print("the AI-ML classifier that will be used to predict data about the"   )
    print("users based on their reviews.\n"                                    )
    print("To use this script, two parameters must be sent:"                   )
    print("    - [1] Action to perform:"                                       )
    print("        - formatRevToListDict:     Formats the review json file to" )
    print("                                   be a list of dictionaries. It"   )
    print("                                   will generate a _mod.json file"  )
    print("                                   that you must rename after"      )
    print("                                   verifying its correctness."      )
    print("        - formatUsrListToDict:     Formats the user json file to be")
    print("                                   a list of dictionaries. It"      )
    print("                                   will generate a _mod.json file"  )
    print("                                   that you must rename after"      )
    print("                                   verifying its correctness."      )
    print("        - generateRevRelevantData: Generates a rev_relevant_data.py")
    print("                                   file containing the review"      )
    print("                                   relevant data."                  )
    print("        - generateRelDataWithUsr:  Once generateRevRelevantData has")
    print("                                   been done, creates a"            )
    print("                                   relevant_data.py file containing")
    print("                                   a list of dictionaries including")
    print("                                   user data."                      )
    print("    - [2] Number of reviews to consider, 'all' considers all the"   )
    print("          reviews. Available only when using"                       )
    print("          generateRevRelevantData."                                 )
    print("    - [2] Add user data starting from this line number, based on"   )
    print("          the relevant_data.py line number - 1. Available only when")
    print("          using generateRelDataWithUsr."                            )
    exit(1)


################################################################################
# --                           Capture parameters                           -- #
################################################################################
if sys.argv[1] not in ["formatRevToListDict"    ,
                       "formatUsrListToDict"    ,
                       "generateRevRelevantData",
                       "generateRevRelevantData"]:
    print(f"{sys.argv[1]} not in the list of actions to perform.")
    exit(1)
else:
    actionToPerform = sys.argv[1]
if actionToPerform == 'generateRevRelevantData':
    # -- Capture the number of samples to consider
    if sys.argv[2].lower() != 'all':
        reviewsQty = int(sys.argv[2])
    else:
        reviewsQty = sys.argv[2].lower()
elif actionToPerform == "generateRelDataWithUsr":
    usrStartLine = sys.argv[2]


################################################################################
# --              Format JSON files as a list of dictionaries               -- #
################################################################################
if actionToPerform in ["formatRevToListDict", "formatUsrListToDict"]:
    if actionToPerform == "formatRevToListDict":
        jsonFile = revFileName 
    elif actionToPerform == "formatUsrToListDict":
        jsonFile = usrFileName

    with open(f"{inputDirectory}/{jsonFile}", 'r') as inputFile:
        content = inputFile.read()

    if "},\n" not in content:
        print("File not formated yet")
        modifiedContent = content.replace('}', '},')
        del content
        del inputFile

        modifiedContent = "[\n" + modifiedContent + "\n]\n"
        with open(
            f"{inputDirectory}/{jsonFile.replace('.json', '_mod.json')}", 'w'
        ) as output_file:
            output_file.write(modifiedContent)

        del modifiedContent
        del output_file
    else:
        print("File already formated")

    exit(1)


################################################################################
# --              Generate relevant data from reviews dataset               -- #
################################################################################
if actionToPerform == "generateRevRelevantData":
    # -- Open the input dataset
    datasetFile   = open(f"{inputDirectory}/{revFileName}")
    reviewDataset = json.load(datasetFile)
    relDataFile   = open(f"{inputDirectory}/relevant_data_rev.py", "w")

    relDataFile.write("dataset = [\n")

    # -- Collect relevant data from the review dataset
    iteration = 0
    for data in reviewDataset:
        # -- Generate bigrams without None values 
        # tokens = nltk.word_tokenize(data["text"].lower())
        # bigrams = list(
        #     nltk.ngrams(tokens, 2,
        #         pad_left         = True,
        #         pad_right        = True,
        #         left_pad_symbol  = None,
        #         right_pad_symbol = None
        #     )
        # )[1:-1]
        relDataFile.write(
            str(
                {
                    "text"     : data["text"].lower(),
                    # "bigrams"  : bigrams,
                    "review_id": data["review_id"],
                    "user_id"  : data["user_id"]
                }
            )
        )
        relDataFile.write(",\n")
        iteration += 1
        if type(reviewsQty) == int:
            if iteration == reviewsQty:
                break

    relDataFile.write("]\n")
    datasetFile.close()
    exit(1)


################################################################################
# --                        Generate output dataset                         -- #
################################################################################
if actionToPerform == "generateRelDataWithUsr":
    outputFile = open(f"{inputDirectory}/relevant_data.py", 'a')
    outputFile.write("dataset = [\n")
    outputFile.close()
    del outputFile

    lineNumber = usrStartLine
    while True:
        inputDatasetLine = subprocess.getoutput(
            f"sed -n '{lineNumber}p' '{inputDirectory}/relevant_data_rev.py'"
        )

        # -- Detecting if it is an invalid line
        if "]" in inputDatasetLine and len(inputDatasetLine) < 5:
            break
        else:
            lineNumber += 1

        # -- Get the user data in the usr dataset as per review data line
        userID       = inputDatasetLine.split("'user_id': '")[1].split("'")[0]
        lookfor      = f'"user_id":"{userID}"'
        command      = f"grep -rE '{lookfor}' '{inputDirectory}/{usrFileName}'"
        userDataLine = subprocess.getoutput(command)

        # -- Generate relevant data from user
        lineToAppend  = ", 'user_data': {"
        yelping       = userDataLine.split('yelping_since":"')[1].split('"')[0]
        lineToAppend += f"'yelping_since': '{yelping}', "
        reviewCount   = int(
            userDataLine.split('review_count":')[1].split(",")[0]
        )
        lineToAppend += f"'review_count': {reviewCount}, "
        useful        = int(userDataLine.split('useful":')[1].split(",")[0])
        lineToAppend += f"'useful': {useful}"
        lineToAppend += "}},\n"
        
        # -- Append relevant user data to output dataset
        outputFile = open(f"{inputDirectory}/relevant_data.py", 'a')
        outputFile.write(inputDatasetLine.replace('},', lineToAppend))
        outputFile.close()

    # -- When processed all the users, close the output dataset list
    outputFile = open(f"{inputDirectory}/relevant_data.py", 'a')
    outputFile.write("]\n")
    outputFile.close()
