#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 07:47:11 2018

@author: diego
"""
from scipy.io import arff
import pandas as pd
import math
import operator



def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2.iloc[x]), 2)
    return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet.iloc[x], length)
        distances.append((trainingSet.iloc[x], dist))
    distances.sort(key = operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors
    
def getResponse(neighbors):
    # Creating a list with all the possible neighbors
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def main():
    data = arff.loadarff('clasificacion-drug.arff') 
    df = pd.DataFrame(data[0])
    
    
    Age = float(input("Ingrese edad: "))
    Sex = float(input("Ingrese sexo: "))
    BP = float(input("Ingrese BP: "))
    Cholesterol = float(input("Ingrese colesterol: "))
    Na = float(input("Ingrese Na: "))
    K = float(input("Ingrese K: "))
    
    newInstance = [Age, Sex, BP, Cholesterol, Na, K]
    k = 3
    neighbors = getNeighbors(df, newInstance, k)
    print(neighbors)
    result = getResponse(neighbors)

    print('> predicted=' + repr(result) )


main()