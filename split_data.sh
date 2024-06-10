#!/bin/bash

# Set the input and output directories
INPUT_FILE="./data/cal_housing.dat"
TRAIN_PREFIX="./data/train_"
TEST_PREFIX="./data/test_"
TRAIN_RATIO=0.7

# Create output directories if they don't exist
mkdir -p ./data

# Total number of lines in the input file
TOTAL_LINES=$(wc -l < "$INPUT_FILE")

# Calculate number of lines for training and testing
TRAIN_LINES=$(echo "$TOTAL_LINES * $TRAIN_RATIO" | bc | awk '{print int($1+0.5)}')
TEST_LINES=$(echo "$TOTAL_LINES - $TRAIN_LINES" | bc)

# Shuffle the input file
gshuf "$INPUT_FILE" > "./data/shuffled_train.dat"

# Split the shuffled file into training and testing sets
head -n "$TRAIN_LINES" "./data/shuffled_train.dat" > "./data/train_data.dat"
tail -n "$TEST_LINES" "./data/shuffled_train.dat" > "./data/test_data.dat"

# Split the training data into 6 files
split -l $(($TRAIN_LINES / 6)) -d -a 1 "./data/train_data.dat" "$TRAIN_PREFIX"

# Split the testing data into 6 files
split -l $(($TEST_LINES / 6)) -d -a 1 "./data/test_data.dat" "$TEST_PREFIX"

# Rename the split files for training and testing
for i in {0..5}; do
  mv "${TRAIN_PREFIX}${i}" "${TRAIN_PREFIX}${i}.dat"
  mv "${TEST_PREFIX}${i}" "${TEST_PREFIX}${i}.dat"
done

# Clean up temporary files
rm "./data/shuffled_train.dat"
rm "./data/train_data.dat"
rm "./data/test_data.dat"

echo "Data has been split into training and testing sets."
