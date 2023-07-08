import numpy as np


### Assignment 3 ###

class KNN:
    def __init__(self, k):
        #KNN state here
        #Feel free to add methods
        self.k = k

    def distance(self, featureA, featureB):
        diffs = (featureA - featureB)**2
        return np.sqrt(diffs.sum())

    def train(self, X, y):
        #training logic here
        #input is an array of features and labels
        self.features = X
        self.labels = y
        None      
        
    def predict(self, X):
        #Run model here
        #Return array of predictions where there is one prediction for each set of features
        result = []
        for i in X:
            neighbors = self.neighbors(i)
            dic = {}
            for j in range(len(neighbors)):
                if neighbors[j][1] in dic:
                    dic[neighbors[j][1]] += 1
                else:
                    dic[neighbors[j][1]] = 1
            temp = max(dic,key=lambda x:dic[x])
            result.append(temp)
        return np.array(result)

    def neighbors(self,target):
        distance = []
        for i in range(len(self.features)):
            distance.append([self.distance(target,self.features[i]),self.labels[i]])
        distance.sort(key=lambda x:x[0])
        closest = self.k
        neighbors = []
        for i in range(closest):
            neighbors.append(distance[i])
        return neighbors


class Perceptron:
    def __init__(self, w, b, lr):
        #Perceptron state here, input initial weight matrix
        #Feel free to add methods
        self.lr = lr
        self.w = w
        self.b = b

    def train(self, X, y, steps):
        #training logic here
        #input is array of features and labels
        for i in range(len(X)):
            temp = np.dot(X[i], self.w) + self.b
            if temp > 0:
                predict = 1
            else:
                predict = 0
            self.w += (self.lr * (y[i] - predict) * X[i])
            self.b += (self.lr * (y[i] - predict))

    def predict(self, X):
        #Run model here
        #Return array of predictions where there is one prediction for each set of features
        result = []
        for i in range(len(X)):
            temp = np.dot(X[i], self.w) + self.b
            if temp > 0:
                result.append(1)
            else:
                result.append(0)
        return np.array(result)

    
class ID3:
    def __init__(self, nbins, data_range):
        #Decision tree state here
        #Feel free to add methods
        self.bin_size = nbins
        self.range = data_range
        self.root = None

    def preprocess(self, data):
        #Our dataset only has continuous data
        norm_data = np.clip((data - self.range[0]) / (self.range[1] - self.range[0]), 0, 1)
        categorical_data = np.floor(self.bin_size*norm_data).astype(int)
        return categorical_data

    def train(self, X, y):
        #training logic here
        #input is array of features and labels
        categorical_data = self.preprocess(X)
        tup = zip(categorical_data, y)
        examples = list(tup)
        attributes = categorical_data[0]
        parent_examples = None
        self.root = self.decision_tree_learning(examples, attributes, parent_examples)

    def predict(self, X):
        #Run model here
        #Return array of predictions where there is one prediction for each set of features
        categorical_data = self.preprocess(X)
        results = []
        for i in range(len(categorical_data)):
            while self.root.label is None:
                self.root = self.root.children[categorical_data[i][self.root.attribute]]
            results.append(self.root.label)
        return np.array(results)

    def decision_tree_learning(self, examples, attributes, parent_examples):
        temp = 0
        if len(examples) == 0:
            return parent_examples
        for i in examples:
            temp += i[1]
        if temp == 0:
            return Tree(0, None)
        elif temp == len(examples):
            return Tree(1, None)
        elif len(attributes) == 0:
            return self.plurality_value(examples)
        else:
            A = self.importance(attributes, examples)
            tree = Tree(None, A)
            for j in range(len(examples)):
                for k in examples:
                    exs = []
                    if k[0][A] == j:
                        exs.append(k)
                attributes_A = list(set(attributes) - set(list([A])))
                subtree = self.decision_tree_learning(exs, attributes_A, examples)
                tree.children.append(subtree)
            return tree

    def importance(self, attributes, examples):
        best_gain = -1
        best_attribute = None
        count_ones = 0
        count_zeros = 0
        info = 0
        #entropy
        if len(examples) != 0:
            for i in range(len(examples)):
                if examples[i][1] == 0:
                    count_zeros += 1
                else:
                    count_ones += 1
                probability_zeros = count_zeros/len(examples)
                probability_ones = count_ones/len(examples)
                if probability_zeros != 0 and probability_ones != 0:
                    info = -(probability_zeros * np.log2(probability_zeros) + probability_ones * np.log2(probability_ones))
            #attribute_entropy
            for j in range(len(attributes)):
                attribute_info = 0
                temp = 0
                for k in range(len(examples)):
                    count_ones = 0
                    count_zeros = 0
                    for n in range(len(examples)):
                        if examples[n][0][attributes[j]] == k:
                            if examples[n][1] == 0:
                                count_zeros += 1
                                probability_zeros = count_zeros/(count_zeros + count_ones)
                                temp = (probability_zeros * np.log2(probability_zeros))
                                attribute_info += temp
                            else:
                                count_ones += 1
                                probability_ones = count_ones/(count_zeros + count_ones)
                                temp = (probability_ones * np.log2(probability_ones))
                                attribute_info += temp
                #gain
                gain = info - attribute_info
                if gain > best_gain:
                    best_gain = gain
                    best_attribute = attributes[j]
        return best_attribute


    def plurality_value(self, examples):
        count_ones = 0
        count_zeros = 0
        for i in range(len(examples)):
            if examples[i][1] == 0:
                count_zeros += 1
            else:
                count_ones += 1
        if count_ones > count_zeros:
            return Tree(1, None)
        else:
            return Tree(0, None)

class Tree:
    def __init__(self, label, attribute):
        self.attribute = attribute
        self.label = label
        self.children = []

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
        return None

    def backward(self, gradients):
        #Write backward pass here
        return None


class Sigmoid:

    def __init__(self):
        None

    def forward(self, input):
        #Write forward pass here
        return None

    def backward(self, gradients):
        #Write backward pass here
        return None


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


        return self.cluster
        #return array with cluster id corresponding to each item in dataset

