import csv, sys
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import date
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer as count_vectorizer



LAST_ENTRY_FOR_SEP = 39 # magic number for bitcoin data matching news data's last entry (sep 30)

def cleanseData():
    bitcoindf = pd.read_csv('cryptocurrencypricehistory/bitcoin_price.csv', sep=",")
    refDate = date(2016, 1, 1)
    bitcoindf[pd.to_datetime(bitcoindf["Date"]) >= refDate].to_csv("recentbitcoin.csv", index=False)

    newsdf = pd.read_csv('million-headlines/abcnews-date-text.csv', sep=",")
    newsdf[pd.to_datetime(newsdf["publish_date"], format='%Y%m%d') >= refDate].to_csv("recentnews.csv", index=False)

def readData():
    bcreader = csv.reader(open('recentbitcoin.csv', "rt"), delimiter=",")
    bitcoins = list(bcreader)
    bitcoinMatrix = np.array(bitcoins)
    # btcseventyPercent = int(round(0.7 * len(bitcoinMatrix)))
    # testBitcoinMatrix = bitcoinMatrix[0:btcseventyPercent]
    # cvBitcoinMatrix = bitcoinMatrix[btcseventyPercent:]


    newsReaders = csv.reader(open('recentnews.csv', "rt"), delimiter=",")
    newsList = list(newsReaders)
    newsMatrix = np.array(newsList)
    #newsSeventyPercent = int(round(0.7 * len(newsMatrix)))
    #testNewsMatrix = newsMatrix[0:newsSeventyPercent]
    #cvNewsMatrix = newsMatrix[newsSeventyPercent:]
    return newsMatrix, bitcoinMatrix
    #return bitcoinMatrix, newsMatrix
    # print bitcoinMatrix
    # print newsMatrix

def readWordFreqs():
    wordFreqsReader = csv.reader(open('wordFrequencies.csv', "rt"), delimiter=",")
    wordFreqsList = list(wordFreqsReader)
    wordFreqs = np.array(wordFreqsList).astype(int)
    return wordFreqs

def generateWordsMatrix(newsMatrix, btMatrix):
    corpus = []
    # newsReaders = csv.reader(open('recentnews.csv', "rb"), delimiter=",")
    # newsList = list(newsReaders)
    # newsMatrix = np.array(newsList)
    wordFreqs = {}
    for row in newsMatrix[1:]:
        if pd.to_datetime(row[0], format='%Y%m%d') not in wordFreqs:
            wordFreqs[pd.to_datetime(row[0], format='%Y%m%d')] = {}
        for word in row[1].split(' '):
            if word not in corpus:
                corpus.append(word)
            if word not in wordFreqs[pd.to_datetime(row[0], format='%Y%m%d')]:
                wordFreqs[pd.to_datetime(row[0], format='%Y%m%d')][word] = 1
            else:
                wordFreqs[pd.to_datetime(row[0], format='%Y%m%d')][word] += 1
    dataMatrix = np.zeros((len(btMatrix[LAST_ENTRY_FOR_SEP:]), len(corpus) + 1))
    for i, row in enumerate(btMatrix[LAST_ENTRY_FOR_SEP:]):
        if pd.to_datetime(row[0]) in wordFreqs:
            print(pd.to_datetime(row[0]))
            sys.stdout.flush()
            currWords = wordFreqs[pd.to_datetime(row[0])]
            for j, word in enumerate(corpus):
                if word in currWords:
                    dataMatrix[i][j] = currWords[word]
                else:
                    dataMatrix[i][j] = 0
       # print float(btMatrix[i+1][4]) - float(btMatrix[i+1][1]), 1 if (float(btMatrix[i+1][4]) - float(btMatrix[i+1][1])) > 0 else 0
        dataMatrix[i][-1] = 1 if (float(btMatrix[i + LAST_ENTRY_FOR_SEP][4]) - float(btMatrix[i + LAST_ENTRY_FOR_SEP][1])) > 0 else 0
    np.savetxt("wordFrequencies.csv", dataMatrix.astype(int), fmt='%i', delimiter=",")
    
    return corpus, wordFreqs

# def generateWordFrequencies(corpus, cryptoPrices, news):
#     flag = True
#     for row in news[1:]:
#         freqs = {}

def SKlearnModel(wordFreqs):
    # print wordFreqs[:, :len(wordFreqs[0]) - 2].shape
    # print wordFreqs[:, -1]
    # sys.stdout.flush()
    X_train, X_test, Y_train, Y_test = train_test_split(wordFreqs[:, :len(wordFreqs[0]) - 2], wordFreqs[:,-1], test_size=0.2, random_state=2)
    logreg = linear_model.LogisticRegression()
    logreg.fit(X_train, Y_train)
    #logreg.fit(wordFreqs[:, :-2], wordFreqs[:, -1])
    # print wordFreqs[:, :-2], wordFreqs[:, -1]
    # print wordFreqs[:, :-2].shape, wordFreqs[:, -1].shape
    # sys.stdout.flush()
    #print logreg.score(X_train.astype(int), Y_train.astype(int))
    print(logreg.score(X_train, Y_train))
    print(logreg.score(X_test, Y_test))

def main():
    #cleanseData()
    # news, cryptoPrice = readData()
    # corpus, wordFreqs = generateWordsMatrix(news, cryptoPrice)
    wordFreqs = readWordFreqs()
    SKlearnModel(wordFreqs)

if __name__ == "__main__":
    main()