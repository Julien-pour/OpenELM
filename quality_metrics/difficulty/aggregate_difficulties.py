# simple script to aggregate results
import os
import numpy as np
import re
import json
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-f', '--file-prefix')
parser.add_argument('-d', '--directory', default='results')

args = parser.parse_args()

delete_previous = False

pattern_passatk = f'{args.file_prefix}_p([0-9]+)_passk.json'
pattern_puzzles = f'{args.file_prefix}_p([0-9]+)_puzzles_solved.json'

# compute number of partitions
all_files = os.listdir(args.directory)
partitions = [int(re.match(pattern_passatk, filename)[0]) for filename in all_files if re.match(pattern_passatk, filename)]

if not partitions:
    exit(0)

max_partition = max(partitions)

all_read_passatk = {}
all_read_puzzles = {}

print(f"Reading all jsons up to partition {max_partition}")
for partition in range(max_partition):
    d_passatk = json.load(open(f'{args.file_prefix}_p{partition}_passk.json', 'r'))
    d_puzzles = json.load(open(f'{args.file_prefix}_p{partition}_puzzles_solved.json', 'r'))

    all_read_passatk[partition] = d_passatk
    all_read_puzzles[partition] = d_puzzles

max_k = max(list(all_read_passatk[partition].keys()))

all_passatk = {}
all_puzzles = {}

for k in list(all_read_passatk[partition].keys()):
    all_passatk[k] = np.mean([np.mean([all_read_passatk[partition][k] for partition in range(max_partition)])])
    all_puzzles[k] = sum([all_read_puzzles[partition]])

all_passatk_path = f"{args.file_prefix}_passk.json"
all_puzzles_path = f"{args.file_prefix}_puzzles_solved.json"

print(f"Saving aggregate pass@k jsons at {all_passatk_path}")
json.dump(all_passatk, open(all_passatk_path, 'w'))
print(f"Saving aggregate puzzle jsons at {all_puzzles_path}")
json.dump(all_puzzles, open(all_puzzles_path, 'w'))

if delete_previous:
    print("Deleting previous runs")
    for partition in range(max_partition):
        os.remove(f'{args.file_prefix}_p{partition}_passk.json')
        os.remove(f'{args.file_prefix}_p{partition}_puzzles_solved.json')
