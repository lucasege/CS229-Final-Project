import csv, sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import scale
from textblob import TextBlob
from decimal import Decimal
rng = np.random

def importTrain(X, Y):
    feature_labels = ["word" + str(x) for x in range(len(X[-2]))]
    train_features = tf.convert_to_tensor(X[:400], dtype=tf.uint8)
    faetures = {}
    train_labels = Y[:400]
    return 
    print(train_features[4])

def readData():
    bcreader = csv.reader(open('recentbitcoin.csv', "rt"), delimiter=",")
    bitcoins = list(bcreader)
    bitcoinMatrix = np.array(bitcoins)
    newsReaders = csv.reader(open('recentnews.csv', "rt"), delimiter=",")
    newsList = list(newsReaders)
    newsMatrix = np.array(newsList)
    return newsMatrix, bitcoinMatrix

def generateWordsMatrix():
    newsMatrix, btMatrix = readData()
    #sentimentnp = np.array((len(btMatrix[39:]), 4))
    sentiments = {}
    for row in newsMatrix[1:]:
        currblob = TextBlob(row[1])
        if pd.to_datetime(row[0], format='%Y%m%d') not in sentiments:
            sentiments[pd.to_datetime(row[0], format='%Y%m%d')] = {}
            sentiments[pd.to_datetime(row[0], format='%Y%m%d')]['count'] = 1
            sentiments[pd.to_datetime(row[0], format='%Y%m%d')]['sentiment'] = currblob.sentiment.polarity
            sentiments[pd.to_datetime(row[0], format='%Y%m%d')]['subjectivity'] = currblob.sentiment.subjectivity
        else:
            sentiments[pd.to_datetime(row[0], format='%Y%m%d')]['count'] += 1
            sentiments[pd.to_datetime(row[0], format='%Y%m%d')]['sentiment'] += currblob.sentiment.polarity
            sentiments[pd.to_datetime(row[0], format='%Y%m%d')]['subjectivity'] += currblob.sentiment.subjectivity
    sentimentcsv = []
    writer = csv.writer(open("sentiment_averages.csv", "w"))
    for row in sentiments:
        sentiments[row]['sentiment_avg'] = (sentiments[row]['sentiment'] / sentiments[row]['count'])
        sentiments[row]['sentiment_weighted_avg'] = ((sentiments[row]['sentiment'] * sentiments[row]['subjectivity']) / sentiments[row]['count'])
        #print(sentiments[row]['sentiment'], sentiments[row]['count'], sentiments[row]['sentiment_avg'])
        writer.writerow([str(row), sentiments[row]['sentiment_avg'], sentiments[row]['sentiment_weighted_avg']])
        #sentimentcsv.append((str(row), sentiments[row]['sentiment_avg'], sentiments[row]['sentiment_weighted_avg']))

#     print(sentimentcsv)
    return np.array(sentimentcsv), btMatrix[39:]
#    # print(sentiments)
    
#     return None#corpus, wordFreqs


def readGoogleTrends():
    timelineReader = csv.reader(open('multiTimeline.csv', "rt"), delimiter=",")
    timeline = list(timelineReader)
    tMatrix = np.array(timeline)
    features = []
    features.append(0)
    features.append(0)
    for i, row in enumerate(tMatrix[1:]):
        for j in range(7):
            features.append(row[1])
            #print((i*7)+2 + j)
            # features[((i*7)+2) + j] = row[1]
    return features



def TFlearnModel(X, Y):
    print(X.shape[1])
    feature_columns = [tf.feature_column.numeric_column("x", shape=[X.shape[1]])]
    # classifier = tf.estimator.LinearClassifier(feature_columns=feature_columns,
    # n_classes=2,
    # model_dir="/tmp/crypto_model_log_reg")
    tf.set_random_seed(1)
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
    hidden_units=[1024, 512, 256],
    #hidden_units=[1000, 500, 250, 125],
   # hidden_units=[128, 64, 32],
    #hidden_units=[10, 20],
    n_classes=2,
    model_dir="/tmp/crypto_model")
    #classifier = tf.estimator.SVM(feature_columns=feature_columns)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": X[:500]},
        y=Y[:500],
        num_epochs=25,
        shuffle=True)
    
    #classifier.train(input_fn=train_input_fn, steps=2000)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": X[400:500]},
        y=Y[400:500],
        num_epochs=25,
        shuffle=False
    )

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": X[500:]},
        y=Y[500:],
        num_epochs=1,
        shuffle=False
    )
    
    
    classifier.train(input_fn=eval_input_fn)
    print(classifier.evaluate(input_fn=train_input_fn))
    accuracy_score_train = classifier.evaluate(input_fn=predict_input_fn)
    print("eval:", accuracy_score_train)
    # accuracy_score = classifier.evaluate(input_fn=predict_input_fn)
    # print(accuracy_score)
    
    # svmEstimator = tf.contrib.learn.SVM(
    #     example_id_column='example_id',
    #     feature_columns=feature_columns
    # )

    # svmEstimator.fit(input_fn=train_input_fn)
    # svmEstimator.evaluate(input_fn=eval_input_fn)
    # svmEstimator.predict(x=X[400:])

def TFLogits(X, Y):
    #x = tf.placeholder(dtype = tf.float32, shape = [639, 7])
    #y = tf.placeholder(dtype = tf.float32, shape = [639])
    x = tf.placeholder(dtype = tf.float32, shape=[None, 7])
    y = tf.placeholder(dtype = tf.int16, shape=[None])

    logits = tf.contrib.layers.fully_connected(x, 2)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = Y[:500], logits = logits))

    train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

    correct_pred = tf.argmax(logits, 1)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    tf.set_random_seed(1234)
    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    for i in range(201):
        print('EPOCH', i)
        _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: X[:500], y: Y[:500]})
        if i % 10 == 0:
            print("Loss: ", loss)
        print('DONE WITH EPOCH')

    predicted = sess.run([Y[500:]], feed_dict={x: X[500:]})
    print(predicted)




def readSentiment():
    sentReader = csv.reader(open('sentiment_averages.csv', "rt"), delimiter=",")
    sentList = list(sentReader)
    sentiments = np.array(sentList)

    btReader = csv.reader(open('recentbitcoin.csv', "rt"), delimiter=",")
    btlist = list(btReader)
    bt = np.array(btlist)
    return sentiments, bt[:640]

def readWordFreqs():
    wordFreqsReader = csv.reader(open('wordFrequencies.csv', "rt"), delimiter=",")
    wordFreqsList = list(wordFreqsReader)
    wordFreqs = np.array(wordFreqsList).astype(int)
    return wordFreqs

def featureCraft(sentimentcsv, btMatrix, googleTrendsData):
    #features = np.zeros((sentimentcsv.shape[0], 5))
    features = np.zeros((sentimentcsv.shape[0], 8))
    for i, row in enumerate(sentimentcsv[1:]):
        sent = row[1]
        sentAvg = row[2]
        prevDayClose = 0 if i == 0 else btMatrix[(i+1)-1][4]
        prevDayFluctuation = 0 if i == 0 else (float(btMatrix[(i+1)-1][4]) - float(btMatrix[(i+1)-1][1]))
        prevDaySent = 0 if i == 0 else (sentimentcsv[(i+1)-1][1])
        features[i][0] = row[1]
        features[i][1] = 0 if i == 0 else btMatrix[(i+1)-1][4]
        features[i][2] = 0 if i == 0 else (1 if (float(btMatrix[(i+1)-1][4]) - float(btMatrix[(i+1)-1][1])) > 0 else 0)
        features[i][3] = 0 if i == 0 else (sentimentcsv[(i+1)-1][1])
        features[i][4] = 0 if i == 0 else float((btMatrix[(i+1)-1][7]))
        features[i][5] = row[2]
        features[i][6] = googleTrendsData[i]
        features[i][7] = 0 if (float(btMatrix[i+1][4]) - float(btMatrix[i+1][1]) < 0) else 1
        
    print(features[40])
    np.random.shuffle(features)
    return features

def main():
    np.random.seed(100) # 9
    #sentimentcsv, btMatrix = generateWordsMatrix()
    st0 = np.random.get_state()
    #print(st0)
    #sys.stdout.flush()
    sentimentcsv, btMatrix = readSentiment()
    googleTrendsData = readGoogleTrends()

    features = featureCraft(sentimentcsv, btMatrix, googleTrendsData)
    #print(features.shape)
    #wordFreqs = readWordFreqs()
   # print(features[:, :-1])
    TFlearnModel(features[:, :-1], features[:, -1])
    # TFLogits(features[:, :-1].astype(np.float32), features[:, -1].astype(np.int))

if __name__ == "__main__":
    main()