import numpy as np
import math


### Assignment 4 ###

class MLP:
    def __init__(self, w1, b1, w2, b2, lr):
        self.l1 = FCLayer(w1, b1, lr)
        self.a1 = Sigmoid()
        self.l2 = FCLayer(w2, b2, lr)
        self.a2 = Sigmoid()

    def MSE(self, prediction, target):
        return np.square(target - prediction).sum()

    def MSEGrad(self, prediction, target):
        return - 2.0 * (target - prediction)

    def shuffle(self, X, y):
        idxs = np.arange(y.size)
        np.random.shuffle(idxs)
        return X[idxs], y[idxs]

    def train(self, X, y, steps):
        for s in range(steps):
            i = s % y.size
            if(i == 0):
                X, y = self.shuffle(X,y)
            xi = np.expand_dims(X[i], axis=0)
            yi = np.expand_dims(y[i], axis=0)

            pred = self.l1.forward(xi)
            pred = self.a1.forward(pred)
            pred = self.l2.forward(pred)
            pred = self.a2.forward(pred)
            loss = self.MSE(pred, yi)
            #print(loss)

            grad = self.MSEGrad(pred, yi)
            grad = self.a2.backward(grad)
            grad = self.l2.backward(grad)
            grad = self.a1.backward(grad)
            grad = self.l1.backward(grad)

    def predict(self, X):
        pred = self.l1.forward(X)
        pred = self.a1.forward(pred)
        pred = self.l2.forward(pred)
        pred = self.a2.forward(pred)
        pred = np.round(pred)
        return np.ravel(pred)

class FCLayer:

    def __init__(self, w, b, lr):
        self.lr = lr
        self.w = w  #Each column represents all the weights going into an output node
        self.b = b

    def forward(self, input):
        #Write forward pass here
        self.input = input
        return np.dot(self.input, self.w) + self.b

    def backward(self, gradients):
        #Write backward pass here
        temp_w = np.dot(np.transpose(self.input), gradients)
        temp_x = np.dot(gradients, np.transpose(self.w))
        self.w = self.w - self.lr * temp_w
        self.b = self.b - self.lr * gradients
        return temp_x

class Sigmoid:

    def __init__(self):
        None

    def forward(self, input):
        #Write forward pass here
        self.x = 1/(1+np.exp(-input))
        return self.x

    def backward(self, gradients):
        #Write backward pass here
        return gradients * self.x * (1- self.x)


class K_MEANS:

    def __init__(self, k, t):
        #k_means state here
        #Feel free to add methods
        # t is max number of iterations
        # k is the number of clusters
        self.k = k
        self.t = t

    def distance(self, centroids, datapoint):
        diffs = (centroids - datapoint)**2
        return np.sqrt(diffs.sum(axis=1))

    def train(self, X):
        #training logic here
        #input is array of features (no labels)
        k_random = np.random.choice(X.shape[0], size=self.k, replace=False)
        centroids = X[k_random]

        for i in range(self.t):
            self.cluster = []
            for j in range(len(X)):
                #find min distance between datapoint and each centroid
                dist = self.distance(X[j], centroids)
                min_dist = dist.argmin()
                #add min distance to cluster array
                self.cluster.append(min_dist)

            for k in range(self.k):
                #take average of all the points in the cluster to update centroid
                temp = []
                for i in range(len(self.cluster)):
                    if self.cluster[i] != k:
                        continue
                    else:
                        temp.append(X[i])
                average = np.average(temp)
                centroids[k] = average
        return self.cluster
        #return array with cluster id corresponding to each item in dataset


class AGNES:
    #Use single link method(distance between cluster a and b = distance between closest
    #members of clusters a and b
    def __init__(self, k):
        #agnes state here
        #Feel free to add methods
        # k is the number of clusters
        self.k = k

    def distance(self, a, b):
        diffs = (a - b)**2
        return np.sqrt(diffs.sum())

    def train(self, X):
        #training logic here
        #input is array of features (no labels)
        clusters = len(X)
        lst = []
        for i in range(clusters):
            lst.append(i)
        self.cluster = lst

        #1. calculate all of the distances between each pair of datapoints and store them
        distances = []
        for i in range(clusters):
            for j in range(0, i):
                dist = self.distance(X[i], X[j])
                distances.append((i, j, dist))
        #2. sort the list of distances in order from least to greatest
        distances.sort(key=lambda x: x[2])

        #6. Repeat steps 3-5 until #clusters = k
        while self.k != clusters:
            #3. pop off the top of the list to get the closest distance
            top = distances.pop()
            temp1 = self.cluster[top[0]]
            temp2 = self.cluster[top[1]]

            #4. check if they are already in the same cluster or in different clusters
            if temp1 == temp2:
                continue
            #5. If in different clusters, merge them into the same cluster
            # and overwrite all other members' labels in the cluster
            else:
                for i in range(len(self.cluster)):
                    if self.cluster[i] == temp1:
                        self.cluster[i] = temp2
                clusters -= 1

        return self.cluster
        #return array with cluster id corresponding to each item in dataset

