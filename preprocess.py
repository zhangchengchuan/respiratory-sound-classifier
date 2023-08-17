import os
import re
import shutil
import csv
from tqdm import tqdm

# Get dictionary of Index (3 digits) to 1/0 (train/test)
index_dictionary = {}
with open('data/train_test_split.txt', 'r') as f:
    for line in f.readlines():
        idx, ts = tuple(line.split('\t', maxsplit=1))
        index_dictionary[idx[:3]] = 1 if 'train' in ts else 0

label_dictionary = {}
with open('data/ground_truth.txt', 'r') as f:
    for line in f.readlines():
        idx, truth = tuple(line.split('\t', maxsplit=1))
        truth = truth.rstrip('\n')
        label_dictionary[idx] = truth


# Make folder for both train and test
if not os.path.exists('data/train'):
    os.mkdir('data/train')

if not os.path.exists('data/test'):
    os.mkdir('data/test')

# Iterate through database, storing each wav file in respective folder
files = sorted(os.listdir('data/ICBHI_final_database'))
final_csv = open('data/final.csv','w')
final_csv_writer = csv.writer(final_csv)
final_csv_writer.writerow(['patient_idx','path_to_wav','diagnosis','train_or_test'])
label_idx = 1
labels = {}

for file in tqdm(files):
    # ignore non .wav files
    if not re.search(r'\.wav$', file):
        continue

    # Check whether patient at current index is train or test
    patient_idx = file[:3]
    train = index_dictionary[file[:3]]
    label = label_dictionary[file[:3]]
    
    if label not in labels:
        labels[label] = label_idx
        label_idx += 1

    # Write to Final CSV file
    final_csv_writer.writerow([patient_idx, file, labels[label], train])

    # Copy wav file over to respective folders
    if train:
        shutil.copy(f'data/ICBHI_final_database/{file}', 'data/train')
    else:
        shutil.copy(f'data/ICBHI_final_database/{file}', 'data/test')
    
# Store all possible labels in another file
with open('data/labels.txt', 'w') as f:
    for label in tqdm(labels):
        f.write(f"{label} {labels[label]}\n")







