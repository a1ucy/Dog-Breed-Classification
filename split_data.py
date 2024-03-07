import os
import shutil
import random
import math

# Split size (80% training, 20% validation)
split_size = .8
min_len = math.inf
total_pics = 0

# Paths
base_dir = './Images/'
training_dir = './train/'
validation_dir = './val/'

# Create train and validation directories
if not os.path.exists(training_dir):
    os.mkdir(training_dir)
if not os.path.exists(validation_dir):
    os.mkdir(validation_dir)

# find min len breed
for breed in os.listdir(base_dir):
    if not breed.startswith('n02'):
        continue
    breed_source_dir = base_dir + breed + '/'
    len_breed = len(os.listdir(breed_source_dir))
    total_pics += len_breed
    if len_breed > 0 and len_breed < min_len:
        min_len = len_breed


def split_data(source, training, validation, split_size):
    files = os.listdir(source)

    training_length = int(min_len * split_size)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[:training_length]
    validation_set = shuffled_set[training_length:min_len]

    for filename in training_set:
        this_file = source + filename
        destination = training + filename
        shutil.copyfile(this_file, destination)

    for filename in validation_set:
        this_file = source + filename
        destination = validation + filename
        shutil.copyfile(this_file, destination)


for breed in os.listdir(base_dir):
    if not breed.startswith('n02'):
        continue

    # create breed folder name under train & val folders
    breed_source_dir = base_dir + breed + '/'
    _, breed_name = breed.split('-',1)
    breed_training_dir = training_dir + breed_name + '/'
    breed_validation_dir = validation_dir + breed_name + '/'

    if not os.path.exists(breed_training_dir):
        os.mkdir(breed_training_dir)
    if not os.path.exists(breed_validation_dir):
        os.mkdir(breed_validation_dir)

    # split data into train & val
    split_data(breed_source_dir, breed_training_dir, breed_validation_dir, split_size)
    print('Split', breed_name, 'complete.')


print("Data split complete.")
print('Total number of pictures:', total_pics)
