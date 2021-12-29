import pandas as pd

import json
import sys
sys.path.append('../')

 # Opening JSON file
with open('../../data/du_2017_split/processed/train.json') as train_json_file:
    train_data = json.load(train_json_file)

with open('../../data/du_2017_split/processed/dev.json') as dev_json_file:
    validation_data = json.load(dev_json_file)

with open('../../data/du_2017_split/processed/test.json') as test_json_file:
    test_data = json.load(test_json_file)