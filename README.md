# SDoH_prediction_using_NLP

This repo contains the code described in "Using NLP to predict the education level in a verbal autopsy dataset"

We use Bidirectional Encoder Representations from Transformers (BERT) models to create word representations of the contents of these reports. To implement, we use BertForSequenceClassification from Hugging Face, which is a BERT transformer with a sequence classification head that takes in the semantically-encoded sentences to predict potential variables of interest.  We predict highest education level which is given in the dataset as a ground truth label. Education level is split into four categories: no education, elementary, high school, and college and beyond. 

Our models can also be modified to predict number of people living at address, access to care, and more, which we plan to analyze in future work. We will only attempt to predict variables which are given in the data as categorical variables, such that we have ground truth labels. We realize that an extension of this work could be to predict other SDOH variables that are not encoded in the data, but that is out of the scope of this project. 

As a baseline, we implemented a simple TF-IDF encoding of the free responses and passed these into a multinomial logistic regression. We used only the open response text as input to the BertForSequenceClassification model. Finally, we input the open responses along with the remaining categorical and numerical variables into our multi-modal classification model
