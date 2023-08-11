# Hate_speech_classifier

## You can find the colab notebook here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uLemUD98PKwF9Y5tS4dPxN5MYczXR9N2?usp=sharing)

Model: The project involves training a sequence classification model for hate speech detection using the Hugging Face transformers library. The model architecture used is distilbert-base-uncased, which is a variant of the DistilBERT model.


Task: The task is to classify text into multiple hate speech categories. The script utilizes multi-label classification techniques, where a single input text can belong to multiple hate speech categories simultaneously.


Dataset: The dataset used is called "Measuring Hate Speech" and is loaded using the datasets library from Hugging Face. The dataset contains columns related to different target attributes (e.g., target_race, target_religion, target_origin, etc.). The code preprocesses the dataset by removing unnecessary columns and then performs a multi-label classification task.
Challanges : some data is biased, some categories have a higher samples then others, Finding the right hyperparameters for the model was quite difficult, the data set was a little old and cannot determine new gen words which are considered hate speech, 
