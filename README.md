# sentiments_detection_cnn

A basic example of Convolutional Neural Network (CNN) developed in python.
The proyect is composed of two files:
- The first one will train the model and tokenizer and save them in the directory.
- The second (client) will use the two resources produced by the first file to process inputs introduced by the user through a html template.

The user must introduce a review about a film, and the model will analice and decide if the review is positive or negative, based on the input, and tokenizer.

To use it, the first file (trainerFile) must be launched, and once the model and tokenizer are created, the second file can be executed.
