import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
# scikit-learn package
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import pandas as pd

X, y = load_iris(return_X_y=True)

X, X_test, y, y_test = train_test_split(X, y, test_size= 0.20)

class Node:
    
    def __init__(self, gini, features_number, features_number_per_classes, prediction_class = None):
        self.gini = gini
        self.features_number = features_number
        self.features_number_per_classes = features_number_per_classes
        self.prediction_class = prediction_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

# Implement a decision tree classifier
class MyDecisionTreeClassifier:
    
    def __init__(self, max_depth):
        self.max_depth = max_depth
    
    def gini(self, classes) -> float:
        '''
        A Gini score gives an idea of how good a split is by how mixed the
        classes are in the two groups created by the split.
        
        A perfect separation results in a Gini score of 0,
        whereas the worst case split that results in 50/50
        classes in each group result in a Gini score of 0.5
        (for a 2 class problem).
        '''
        class_set = set(classes)
        class_number = len(classes)
        if type(classes) is not list:
            classes = classes.tolist()

        gini_result = 0
        for class_of_element in class_set:
            gini_result += (classes.count(class_of_element) / class_number) ** 2
        
        return 1 - gini_result

    
    def split_data(self, features, classes) -> tuple[int, int]:
        
        # test all the possible splits in O(N*F) where N in number of samples
        # and F is number of features

        # return index and threshold value
        if len(classes) < 2 :
            return None, None

        features_size = len(features[0])
        m = len(classes)

        df = pd.DataFrame(features, columns=[x for x in range(features_size)])
        df = df.reset_index(drop=True)
        df['class'] = classes

        best_gini = float('inf')
        best_threshold = None
        best_index = None
        
        for i in range(self.number_of_features_):
            df_copy = copy.deepcopy(df)

            df_copy = df_copy.sort_values(by=[i])

            for left_i in range(1,m):
                left_node_classes = df_copy['class'][:left_i].values.tolist()
                right_node_classes = df_copy['class'][left_i:].values.tolist()

                local_gini = i / m * self.gini(left_node_classes) + (m - i) / m * self.gini(right_node_classes)
                
                if df_copy[i][left_i-1] == df_copy[i][left_i]:
                    continue

                if local_gini < best_gini:
                    best_gini = local_gini
                    best_threshold = (df_copy[i][left_i-1] + df_copy[i][left_i]) / 2
                    best_index = i

        return best_index, best_threshold
        

    
    def build_tree(self, features, classes, depth = 0):
        # create a root node
        # recursively split until max depth is not exeeced

        features_number_per_classes = [np.sum(classes == i) for i in range(self.number_of_classes_)]
        prediction_class = np.argmax(features_number_per_classes)

        node = Node(gini = self.gini(classes),
                    features_number = len(classes),
                    features_number_per_classes = features_number_per_classes,
                    prediction_class = prediction_class
                    )

        if depth < self.max_depth:
            index, threshold = self.split_data(features, classes)
            if index is not None:
                left_index = features[:, index] < threshold
                left_features, left_classes = features[left_index], classes[left_index]
                right_features, right_classes = features[~left_index], classes[~left_index]
                
                # left_features, left_classes = [x for x in features if x[index] <= threshold], [y[i] for i, x in enumerate(features) if x[index] <= threshold]
                # right_features, right_classes = [x for x in features if x[index] > threshold], [y[i] for i, x in enumerate(features) if x[index] > threshold]

                if len(left_classes) == 0 or len(right_classes) == 0:
                    return node
                # print("inner node")
                node.feature_index = index
                node.threshold = threshold
                node.left = self.build_tree(left_features, left_classes, depth+1)
                node.right = self.build_tree(right_features, right_classes, depth+1)
                return node
        # print(f'leaf {prediction_class}')
        return node
    
    def fit(self, features, classes):
        # basically wrapper for build tree / train
        self.number_of_classes_ = len(set(classes))
        self.number_of_features_ = features.shape[1]
        self.tree = self.build_tree(features, classes)


    def predict(self, X):
        return [self._predict(inputs) for inputs in X]
    
    def _predict(self, feature_test):
        
        # traverse the tree while there is a child
        # and return the predicted class for it,
        # note that X_test can be a single sample or a batch
        
        node = self.tree
        while True:
            if feature_test[node.feature_index] <= node.threshold:
                if node.left:
                    node = node.left
                else:
                    return node.prediction_class
            else:
                if node.right:
                    node = node.right
                else:
                    return node.prediction_class

    def evaluate(self, X_test, y_test):
        # return accuracy
        return sum(self.predict(X_test) == y_test) / len(y_test)

classifier = MyDecisionTreeClassifier(10)
classifier.fit(X_test, y_test)
print(classifier.evaluate(X_test, y_test))