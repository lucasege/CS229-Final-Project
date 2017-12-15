import tensorflow as tf
import numpy as np
import csv, sys

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 1
batch_size = 25

x = tf.placeholder('float32', [None, 7])
y = tf.placeholder('float32')

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

def readSentiment():
    sentReader = csv.reader(open('sentiment_averages.csv', "rt"), delimiter=",")
    sentList = list(sentReader)
    sentiments = np.array(sentList)

    btReader = csv.reader(open('recentbitcoin.csv', "rt"), delimiter=",")
    btlist = list(btReader)
    bt = np.array(btlist)
    return sentiments, bt[:640]

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
        features[i][2] = 0 if i == 0 else (float(btMatrix[(i+1)-1][4]) - float(btMatrix[(i+1)-1][1]))
        features[i][3] = 0 if i == 0 else (sentimentcsv[(i+1)-1][1])
        features[i][4] = 0 if i == 0 else float((btMatrix[(i+1)-1][7]))
        features[i][5] = row[2]
        features[i][6] = googleTrendsData[i]
        features[i][7] = 0 if (float(btMatrix[i+1][4]) - float(btMatrix[i+1][1]) < 0) else 1
        
    
    np.random.shuffle(features)
    return features

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([7, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}


    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output

def trainNetwork(realx, realy, testX, testY):
    prediction = neural_network_model(realx)

    cost = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y) )
    #cost = tf.reduce_mean( tf.losses.log_loss(prediction, y) )
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    hm_epochs = 10
    with tf.Session() as sess:
        # OLD:
        #sess.run(tf.initialize_all_variables())
        # NEW:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            #for i in range(20):
            #for _ in range(int(mnist.train.num_examples/batch_size)):
                #epoch_x, epoch_y = np.array(realx[i * batch_size : (i * batch_size) + batch_size]), np.array(realy[i * batch_size : (i * batch_size) + batch_size])
                #print(epoch_x.shape, epoch_y.shape)
                # epoch_x, epoch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={x: realx, y: realy})
                #_, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                #epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        print(prediction)
        correct = tf.equal(tf.argmax(prediction), tf.argmax(y))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x:testX, y:testY}))
        # print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


def main():
    sentiments, bt = readSentiment()
    googleTrends = readGoogleTrends()
    features = featureCraft(sentiments, bt, googleTrends)
    trainX, trainY = features[:500, :-1], features[:500, -1]
    testX, testY = features[500:, :-1], features[500:, -1]

    trainNetwork(trainX.astype(np.float32), trainY.astype(np.float32), testX.astype(np.float32), testY.astype(np.float32))

    # learn

if __name__ == "__main__":
    main()