# Breast-Cancer-Histopathology
This repository contains a visual recognition artificial intelligence project that utilizes a convolutional neural network to detect the aggressiveness of Invasive Ductal Carcinoma (IDC) in magnified breast cancer histopathology images.
Aggressiveness scores are outputted as the percentage of IDC positive subpatches out of the total number of subpatches.

Dataset used to train the model: https://www.kaggle.com/paultimothymooney/breast-histopathology-images

====================================================================================

Report.pdf: PDF of final report
best_model.h5: File that contains our trained Convolutional Neural Network model
mean_image.txt: File that contains training data's mean value we used to preprocess
nnarch.png: Block diagram of our CNN's architecture 
project.ipynb: Notebook that demonstrates project (loads pre-trained model)
project.html: HTML of project notebook after all cells are run

====================================================================================

src: subdirectory that contains all of our code for the project
  - model_training.ipynb: Project notebook where we train and save our model
  - project_functions.py: helper functions used in project notebook
  - nnfigure.py: Someone else script from GitHub that draws your NN architecture 

====================================================================================

data: Directory containing sample data used to demonstrate project