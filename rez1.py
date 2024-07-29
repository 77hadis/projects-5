#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


# In[2]:


# Assuming you have a DataFrame named 'df' with a column named 'Text' containing the sentences
# You should replace 'your_excel_file.xlsx' with the actual path to your Excel file
df = pd.read_excel('HW1_AUT_MLPR_4021-Ar-Fa.xlsx')


# In[3]:


df.head()


# In[4]:


# Extracting the sentences from the 'Text' column
sentences = df['جملات'].tolist()
sentences[:10]


# In[5]:


# A
# Creating a binary BoW feature vector using CountVectorizer
vectorizer = CountVectorizer(analyzer='char', binary=True)
X = vectorizer.fit_transform(sentences)
X


# In[7]:


# Converting the sparse matrix to a dense NumPy array
feature_matrix = X.toarray()


# In[8]:


# Displaying the feature matrix
print("Feature Matrix:")
print(feature_matrix)


# In[9]:


# B
# Creating a weighted BoW feature vector using CountVectorizer
vectorizer_b = CountVectorizer(analyzer='char', binary=False)
X_b = vectorizer_b.fit_transform(sentences)

# Converting the sparse matrix to a dense NumPy array
feature_matrix_b = X_b.toarray()

# Displaying the feature matrix
print("Weighted Feature Matrix:")
print(feature_matrix_b)


# In[10]:


# P
import numpy as np
normalized_feature_matrix = feature_matrix_b / feature_matrix_b.sum(axis=1)[:, np.newaxis]


# In[11]:


# Displaying the normalized feature matrix
print("Normalized Feature Matrix:")
print(normalized_feature_matrix)


# In[12]:


# T
# Calculate the mean and standard deviation for each feature
feature_means = np.mean(feature_matrix, axis=0)
feature_stds = np.std(feature_matrix, axis=0)

# Standardize the feature matrix
standardized_feature_matrix = (feature_matrix - feature_means) / feature_stds

# Displaying the standardized feature matrix
print("Standardized Feature Matrix:")
print(standardized_feature_matrix)


# In[14]:


# S
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


# In[15]:


# Assuming you have a DataFrame named 'df' with a column named 'Text' and a column named 'Label'
# 'Label' should contain the class labels for your samples
# You should replace 'your_excel_file.xlsx' with the actual path to your Excel file
#df = pd.read_excel('your_excel_file.xlsx')

# Extracting the sentences and labels from the DataFrame
#sentences = df['Text'].tolist()
labels = df['زبان'].tolist()

# Convert labels to numeric format using LabelEncoder
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
encoded_labels


# In[21]:


# Concatenate the training and testing data for feature extraction
all_data = X_train + X_test

# Function to create feature vectors based on the specified method
def create_feature_vector(data, method='binary'):
    vectorizer = CountVectorizer(analyzer='char', binary=(method == 'binary'))
    X = vectorizer.fit_transform(data)
    return X.toarray()

# Function to evaluate kNN algorithm with different distance metrics
def evaluate_knn(X_train, X_test, y_train, y_test, k_values, distance_metrics):
    results = []
    for k in k_values:
        for metric in distance_metrics:
            # Create and fit kNN classifier
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
            knn.fit(X_train, y_train)

            # Make predictions
            y_pred = knn.predict(X_test)

            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)

            # Store results
            results.append({
                'k': k,
                'metric': metric,
                'accuracy': accuracy
            })

    return results

# List of k values and distance metrics
k_values = [1, 3, 5]
distance_metrics = ['cosine', 'euclidean', 'correlation']

# Loop through feature vectors and evaluate kNN
feature_vectors = ['binary', 'weighted', 'normalized', 'standardized']
for method in feature_vectors:
    X_feature = create_feature_vector(all_data, method=method)
    X_train_feature = X_feature[:len(X_train)]
    X_test_feature = X_feature[len(X_train):]

    results = evaluate_knn(X_train_feature, X_test_feature, y_train, y_test, k_values, distance_metrics)

    # Display results
    print(f"\nResults for {method} feature vector:")
    results_df = pd.DataFrame(results)
    print(results_df)


# In[ ]:


#Certainly! I'll provide you with the code for two common feature extraction methods: 
#TF-IDF and Word Embeddings using Word2Vec. Note that for the Word Embeddings approach, 
#you'd need a pre-trained Word2Vec model. In this example, I'll use the gensim library for Word2Vec.


# In[23]:


# G
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier


# In[24]:


# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(analyzer='word', max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Run kNN with TF-IDF features
knn_tfidf = KNeighborsClassifier(n_neighbors=3, metric='cosine')
knn_tfidf.fit(X_train_tfidf, y_train)
y_pred_tfidf = knn_tfidf.predict(X_test_tfidf)

# Calculate accuracy
accuracy_tfidf = accuracy_score(y_test, y_pred_tfidf)
print(f'TF-IDF Accuracy: {accuracy_tfidf}')


# In[28]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec

# Tokenize sentences into words
tokenized_sentences = [sentence.split() for sentence in X_train]

# Train Word2Vec model
word2vec_model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)

# Average Word Embeddings for sentences
X_train_word2vec = [np.mean([word2vec_model.wv[word] for word in sentence.split() if word in word2vec_model.wv] or [np.zeros(100)], axis=0) for sentence in X_train]
X_test_word2vec = [np.mean([word2vec_model.wv[word] for word in sentence.split() if word in word2vec_model.wv] or [np.zeros(100)], axis=0) for sentence in X_test]

# Define k and distance metrics
k_values = [1, 3, 5]

# Results DataFrame
results_df = pd.DataFrame(columns=['Method', 'k', 'Distance Metric', 'Accuracy'])

# TF-IDF
tfidf_vectorizer = TfidfVectorizer(analyzer='word', max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

for k in k_values:
    for metric in ['cosine', 'euclidean']:  # Use valid metrics for sparse input
        # Run kNN with TF-IDF features
        knn_tfidf = KNeighborsClassifier(n_neighbors=k, metric=metric)
        knn_tfidf.fit(X_train_tfidf, y_train)
        y_pred_tfidf = knn_tfidf.predict(X_test_tfidf)

        # Calculate accuracy
        accuracy_tfidf = accuracy_score(y_test, y_pred_tfidf)

        # Append to results DataFrame
        results_df = results_df.append({'Method': 'TF-IDF', 'k': k, 'Distance Metric': metric, 'Accuracy': accuracy_tfidf}, ignore_index=True)


# In[25]:


from gensim.models import Word2Vec

# Tokenize sentences into words
tokenized_sentences = [sentence.split() for sentence in X_train]

# Train Word2Vec model
word2vec_model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)

# Average Word Embeddings for sentences
X_train_word2vec = [np.mean([word2vec_model.wv[word] for word in sentence.split() if word in word2vec_model.wv] or [np.zeros(100)], axis=0) for sentence in X_train]
X_test_word2vec = [np.mean([word2vec_model.wv[word] for word in sentence.split() if word in word2vec_model.wv] or [np.zeros(100)], axis=0) for sentence in X_test]

# Run kNN with Word2Vec features
knn_word2vec = KNeighborsClassifier(n_neighbors=3, metric='cosine')
knn_word2vec.fit(X_train_word2vec, y_train)
y_pred_word2vec = knn_word2vec.predict(X_test_word2vec)

# Calculate accuracy
accuracy_word2vec = accuracy_score(y_test, y_pred_word2vec)
print(f'Word2Vec Accuracy: {accuracy_word2vec}')


# In[29]:


# Word2Vec
for k in k_values:
    for metric in ['cosine', 'euclidean']:  # Use valid metrics for sparse input
        # Run kNN with Word2Vec features
        knn_word2vec = KNeighborsClassifier(n_neighbors=k, metric=metric)
        knn_word2vec.fit(X_train_word2vec, y_train)
        y_pred_word2vec = knn_word2vec.predict(X_test_word2vec)

        # Calculate accuracy
        accuracy_word2vec = accuracy_score(y_test, y_pred_word2vec)

        # Append to results DataFrame
        results_df = results_df.append({'Method': 'Word2Vec', 'k': k, 'Distance Metric': metric, 'Accuracy': accuracy_word2vec}, ignore_index=True)

# Display results
print(results_df)


# In[ ]:




