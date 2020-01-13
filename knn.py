'''
This code is an implementation of K-Nearest Neighbors on the MNIST data set

'''
import numpy as np
import heapq
from datetime import datetime

def read_images(filename, data_size):
    img_data = list()
    with open(filename, 'rb') as f:
        f.read(4)
        img_num = f.read(4)
        row_num = f.read(4)
        col_num = f.read(4)
        for i in range(data_size):
            a_img = np.empty((784, 1))
            for j in range(784):
                a_img[j] = int.from_bytes(f.read(1), 'big') / 255.0
            img_data.append(a_img)
    return img_data

def read_labels(filename, data_size):
    label_data = list()
    with open(filename, 'rb') as f:
        f.read(4)
        count = f.read(4)
        for i in range(data_size):
            a_label = int.from_bytes(f.read(1), 'big')
            label_data.append(a_label)
    return label_data

#Read in MNIST data in to numpy arrays with the corresponding label
def read_mnist(train_size, test_size):
    train_img_data = read_images('train-images', train_size)
    train_label_data = read_labels('train-labels', train_size)
    train_data = list(zip(train_img_data, train_label_data))

    test_img_data = read_images('test-images', test_size)
    test_label_data = read_labels('test-labels', test_size)
    test_data = list(zip(test_img_data, test_label_data))

    return [train_data, test_data]

#knn implementation
def knn(k, train_data, test_point):
    neighbors = []

    #compute the distance between each training image and the current point being classified
    #and maintain a list of the k nearest points
    for train_point in train_data:
        dist = np.sum(np.square(train_point[0] - test_point[0]))
        if len(neighbors) < k:
            neighbors.append((dist, train_point[1]))
            neighbors.sort()
        elif dist < neighbors[-1][0]:
            neighbors[-1] = (dist, train_point[1])
            neighbors.sort()
    return neighbors

#given a set of k nearest points, returns the most frequent classified point
#if there is a tie, choose the nearest neighbor
def classify(nearest_set):
    a_dict = {num: 0 for num in range(10)}
    for item in nearest_set:
        a_dict[item[1]] += 1
    
    max_val = max(a_dict.values())
    nearest = [key for key, val in a_dict.items() if val == max_val]
    if len(nearest) == 1:
        return nearest[0]
    else:
        return nearest_set[0][1]
    

if __name__ == '__main__':
    
    #10,000 training points, 1,000 test points
    [train_data, test_data] = read_mnist(10000, 1000)
    hit = 0
    miss = 0
    startTime = datetime.now()
    #compute the accuracy of knn
    for idx, test_point in enumerate(test_data):
        print('test #{}'.format(idx))
        nearest_set = knn(7, train_data, test_point)
        if classify(nearest_set) == test_point[1]:
            hit += 1
        else:
            miss += 1
    print(datetime.now() - startTime)
    print('Accuracy: {}%'.format(hit / (hit + miss) * 100 ))
    
        