# Convert json predictions file into 
# contexts.txt, gt_questions.txt and gen_questions.txt

import json
import sys
import argparse
sys.path.append('../')
import pandas as pd

def run(args):
    folder_path = args.folder_path

    file_name = "predictions.json"
    file_path = folder_path + "/" + file_name

    with open(file_path, encoding='utf-8') as file:
        predictions_json = json.load(file)

    # Write to contexts.txt file
    #https://stackoverflow.com/questions/899103/writing-a-list-to-a-file-with-python/899176
    with open(folder_path + '/contexts.txt', 'w', encoding="utf-8") as f:
        for item in predictions_json:
            context = item['context']
            if '\n' in context:
                context = context.replace("\n", "") # hard coded fix! To be changed !!!!!!!!!!!!!!!!!!!
            f.write("%s\n" % context)

    # Write to gt_questions.txt file
    with open(folder_path + '/gt_questions.txt', 'w', encoding="utf-8") as f:
        for item in predictions_json:
            f.write("%s\n" % item['gt_question'])

    # Write to gen_questions.txt file
    with open(folder_path + '/gen_questions.txt', 'w', encoding="utf-8") as f:
        for item in predictions_json:
            f.write("%s\n" % item['gen_question'])

    print("Predictions were saved in ", folder_path)

if __name__ == '__main__':
    # Initialize the Parser
    parser = argparse.ArgumentParser(description = 'Convert json predictions file into text files.')

    # Add arguments
    parser.add_argument('-fp','--folder_path', type=str, metavar='', required=True, help='folder that contains predictions.json')
    
    # Parse arguments
    args = parser.parse_args()

    # Start conversion
    run(args)