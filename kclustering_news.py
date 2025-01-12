#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 10:44:23 2024

@author: shaiyaan
"""

import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

API_URL = 'https://newsapi.org/v2/everything?q=warriors&apiKey=c65328e1f6d54cdfb325085118394cc2'

def get_data_json(endpoint):
    params = {
        "q" : "machine learning",
        "apiKey" : "c65328e1f6d54cdfb325085118394cc2",
        "pageSize": 100
        }
    response = requests.get(endpoint, params=params)
    return response.json()

def get_articles(data):
    clean_lst = []
    for article in data["articles"]:
        dct = {}
        dct["source"] = article.get("source", {}).get("name", "Unknown Source")
        dct['author'] = article.get("author", "Unknown Author")
        dct['title'] = article.get("title", "Unknown Title")
        dct['content'] = article.get("content", "Unknown Content")
        clean_lst.append(dct)
    return clean_lst

def words_to_values(DataFrame): 
    vectorizer = TfidfVectorizer(stop_words="english")
    vectorized_data = vectorizer.fit_transform(DataFrame['content'])
    pca = PCA(n_components=2)
    components = pca.fit_transform(vectorized_data.toarray())
    DataFrame[["x", "y"]] = components
    return DataFrame

def plot_intertia(df):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(df[['x', 'y']])
        wcss.append(kmeans.inertia_)  
    plt.plot(range(1, 11), wcss, marker = 'o')
    plt.title('elbow')
    plt.xlabel('# of clusters')
    plt.ylabel('wcss')
    plt.show()

def plot_clusters(df):
    plt.figure(figsize=(8, 6))
    for cluster in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster]
        plt.scatter(cluster_data['x'], cluster_data['y'], label=f'Cluster {cluster}', alpha=0.6)
    plt.title('Clusters of Articles (PCA-Reduced)')
    plt.xlabel('PCA Component 1 (x)')
    plt.ylabel('PCA Component 2 (y)')
    plt.legend(title='Cluster')
    plt.show()
        
def main():
    data = get_data_json(API_URL)
    data = get_articles(data)
    df = pd.DataFrame(data)
    df = words_to_values(df)
    plot_intertia(df)
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[['x', 'y']])
    plot_clusters(df)
    cluster_counts = df['cluster'].value_counts()
    print(cluster_counts)
    
if __name__ == "__main__":
    main()  



