# Improving-the-synonymization-model-test-task

## Project stucture

file [accuracy_utils.py](accuracy_utils.py) containes functions to measure model accuracy: HTS@10 and Mean Rank.

file [transe.py](transe.py) contains implementation of TransE model.

file [link_prediction.ipynb](link_prediction.ipynb) contains training proccess of TransE to solve link prediction task.

## Results

0.415 Hits@10 score was obtained with following hyperparameters:

- Dimension of embedding: 20

- Margin: 2

- Dissimilarity function: $L_1$ norm

- Learning rate: 0.05
