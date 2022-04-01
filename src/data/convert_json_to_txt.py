# Convert json predictions file into 
# contexts.txt, gt_questions.txt and gen_questions.txt

import json
import sys
import argparse
sys.path.append('../')
import pandas as pd
from nltk.tokenize import word_tokenize

def run(args):
    folder_path = args.folder_path

    file_name = "predictions.json"
    file_path = folder_path + "/" + file_name

    with open(file_path, encoding='utf-8') as file:
        predictions_json = json.load(file)

    index_count = 0

    # Write to contexts.txt file
    #https://stackoverflow.com/questions/899103/writing-a-list-to-a-file-with-python/899176
    with open(folder_path + '/contexts.txt', 'w', encoding="utf-8") as f:
        for item in predictions_json:
            context = item['context']
            if '\n' in context:
                context = context.replace("\n", "") # hard coded fix! To be changed !!!!!!!!!!!!!!!!!!!
            words_tokenized = word_tokenize(context)
            words_tokenized_lowercase = [each_string.lower() for each_string in words_tokenized] # lowercase
            context = ' '.join(words_tokenized_lowercase)
            f.write("%s\n" % index_count)
            index_count = index_count + 1

    # Write to gt_questions.txt file
    with open(folder_path + '/gt_questions.txt', 'w', encoding="utf-8") as f:
        for item in predictions_json:
            words_tokenized = word_tokenize(item['gt_question'])
            words_tokenized_lowercase = [each_string.lower() for each_string in words_tokenized] # lowercase
            gt_question = ' '.join(words_tokenized_lowercase)
            f.write("%s\n" % gt_question)

    # Write to gen_questions.txt file
    with open(folder_path + '/gen_questions.txt', 'w', encoding="utf-8") as f:
        for item in predictions_json:
            words_tokenized = word_tokenize(item['gen_question'])
            words_tokenized_lowercase = [each_string.lower() for each_string in words_tokenized] # lowercase
            gen_question = ' '.join(words_tokenized_lowercase)
            f.write("%s\n" % gen_question)

    print("Predictions were saved in ", folder_path)

if __name__ == '__main__':
    # Initialize the Parser
    parser = argparse.ArgumentParser(description = 'Convert json predictions file into text files.')

    # Add arguments
    parser.add_argument('-fp','--folder_path', type=str, metavar='', required=False, default="../../predictions/2022-01-02_15-13-03", help='folder that contains predictions.json')
    
    # Parse arguments
    args = parser.parse_args()

    # Start conversion
    run(args)