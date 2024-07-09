# Q-learning Algorithm Application in Successful Song Lyrics Creation Process

**Authors**: Valentina Brivio, Enrico Cipolla Cipolla, Filip Juren, Arianna Zottoli

## Abstract

This paper presents a method to improve the success rate of song lyrics using a reinforcement learning inspired approach. Focusing on the decade 1990-2000, we developed a classifier to predict hit songs from “Billboard top 100” using a Random Forest with TF-IDF vectorization, genre, and sentiment scores. We integrated the classifier into a Q-Learning algorithm to modify lyrics, with the aid of Google LLM Gemma aiming to maximize hit potential, guided by a reward function composed of a linear combination of the probability given by our classifier and a coherence score. Despite a limited action space, this methodology presents a promising approach to support writers in their song creation process.

## Introduction

The idea behind our study is to provide a creative data-based support for emerging artists. The digital age has significantly lowered barriers to entry, causing market saturation. Figures from Alpha Data show that less than 10% of artists account for over 90% of all streams across major platforms. Being discovered has become significantly more difficult, but the effective use of social media and marketing can be a great launch pad. Virality has in fact become the goal of many artists on social media, and we provide a formula for it.

In this study, we analyzed a large dataset of songs from 1960 to 2022, focusing specifically on the 1990-2000 decade. We decided to focus on this decade given the trends of revival in the latest years, led by artists like TheWeeknd. Our initial goal was to develop a model to distinguish hit songs from non-hits. We developed a Random Forest classifier that uses TF-IDF vectorization of lyrics, sentiment scores, and genre information. We also appropriately weighted each class considering the imbalance between positive and negative samples.

We integrated this classifier into a Q-learning-inspired algorithm, a class of Reinforcement Learning Algorithm, where an agent (in this case, the song) learns the best actions to take to maximize a reward function (in this case, based on our classifier). We used Google’s LLM Gemma 2B, known for its efficiency and speed, suitable for environments with limited resources.

To start, we outline the steps involved in collecting data and constructing our models. Next, we present the outcomes of the two parts and discuss limitations and potential improvements of our approach.

## Method

### Data Collection and Cleaning

The data collection process began by identifying successful songs in the United States. The success was determined by Billboard based on radio airplay, online streaming, record sales, and YouTube views. Billboard publishes annually the top 100 songs of the year. We grouped them into decades, spanning from 1960 to 2022, obtaining a dataset of nearly 1000 successful songs per decade.

We collected the unsuccessful songs from a Kaggle dataset called “English Songs Lyrics”, which contains 5 million songs from the same period. We then merged the datasets, eliminating duplicates. Lastly, we obtained lyrics and genre of the songs. Using Selenium and Beautiful Soup, we scraped the lyrics from the Genius website, whereas we obtained the genres by scraping Last.fm and Wikipedia.

### Classifier Construction

In the classifier construction phase, we decided to focus on songs from 1990 to 2000. The dataset contained 256,805 songs, with title, genre, artist, year, and lyrics for each one. We assigned the label 1 to the 1,083 Billboard Top songs, while labeling the rest as 0. To manage the significant class imbalance, we performed stratified sampling by label and genre. We reduced the number of non-top songs to about 10,000 and kept all the top songs. The resulting subset has 11,080 songs and for each of them we calculated positive, negative, and neutral sentiment scores using RoBERTa for Sentiment Analysis.

Firstly, we conducted an exploratory analysis, then we built a classifier to predict the likelihood of a song being a “Billboard top song”. We used the labels indicating whether a song is top (1) or non-top (0) as the response variable. The song lyrics, genres, and the sentiment scores were the predictor variables. For the variable genre, we applied one-hot encoding, converting the categorical labels into binary vectors. To represent text data, we transformed the lyrics in vector representations testing 2 different methods: TF-IDF with 1-2 grams and Sentence Transformers. We combined these techniques with the following classification methods: Logistic Regression, Random Forest, and XGBoost. To handle the class imbalance, we added higher weights to top songs, to improve performance on them. We trained the models on the training set and assessed the final models’ performance on the test set. We selected the best hyperparameters using the 5-Fold Cross Validation.

**Model selection**: We compared the models using various metrics on the test set: the average precision, recall, F1-score, and accuracy. We focused on the F1-score because it balances precision and recall, providing a better measure of the classifier’s performance in the presence of class imbalances. The best classification model is Random Forest, which uses the genre, sentiment scores, and TF-IDF vector representation of the lyrics as predictors. We observed that including genre and sentiment scores as predictors improves performance compared to considering only the lyrics. Dimensionality reduction, instead, does not improve the model’s performance. Based on the results of all the different models, we decided to implement in the q-learning algorithm the TF-IDF Random Forest classifier.

### Q-learning Algorithm

To modify the song lyrics according to the Billboard 100 criteria, we implemented a simplified version of a Q-learning algorithm. A Q-learning algorithm is a type of reinforcement learning strategy. The technique allows an agent to know how to optimally act in an environment, maximizing a given reward function.

**Q-table States**: The features of a lyric define its state. The states are all the possible combinations of positive, negative, neutral sentiment scores, and number of lines. We compute the sentiment scores of the lyrics using RoBERTa for Sentiment analysis.

**Q-table actions**: The algorithm can either modify, delete, or add a new line to the lyrics. To execute the generative actions of the algorithm (add and modify), we adopted Gemma, a Large Language model developed by Google.

**Q-values**: For each state, the Q-values are the outputs of the reward function in response to an action of the algorithm.

**Reward function**: The reward function is a weighted average of the prediction of the Classifier (0 or 1) and the Coherence Score of the lyrics (0 to 1). The Coherence Score avoids that overly modified lyrics receive a high reward, as it measures the similarity of the modified lyrics to the original lyrics.

For further details, refer to the full project report .
