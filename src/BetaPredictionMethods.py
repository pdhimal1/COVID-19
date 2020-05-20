"""
William Austin
Prakash Dhimal
George Mason University
CS 584 Theory and Applications of Data Mining
Semester project: Predicting the Impact of COVID-19
"""
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import numpy as np

# If memoryFactor = 0, then this is a simple linear prediction
def calculateSlopeHistoryList(betaValuesArray, memoryFactor=0.15, step=1):
    arraySize = betaValuesArray.shape[0]
    slopeHistoryList = []
    i = arraySize - 1
    
    weightSum = 0.0
    memRatio = 1 - memoryFactor
    memWeight = 1.0
        
    while i >= 1:
        start = betaValuesArray[i - 1]
        #middle = betaValuesArray[i - 1]
        end = betaValuesArray[i]
        
        # Divisor is probably 1
        slope = (end - start) / step
        #slope2 = (end - middle)
        
        slopeHistoryTuple = (slope, memWeight)
        slopeHistoryList.append(slopeHistoryTuple)
        i -= 1
        weightSum += memWeight
        memWeight *= memRatio
    
    return slopeHistoryList, weightSum

def getBetaPredictionBySlopeHistory(betaValuesArray, lastBetaValue, predictionDays, memoryFactor=0.15, step=1):
    slopeHistoryList, weightSum = calculateSlopeHistoryList(betaValuesArray, memoryFactor, step)
    
    weightedSlopeSum = 0.0
    
    for slopeHistory in slopeHistoryList:
        slope = slopeHistory[0]
        weight = slopeHistory[1]
        weightedSlopeSum += (slope * weight)
    
    return lastBetaValue + (weightedSlopeSum / weightSum) * predictionDays
    
def getRandomBetaPredictionsBySlopeHistory(betaValuesArray, memoryFactor=0.15, step=1):
    pass

# If memoryFactor = 0, then this is a simple prediction by averaging
def calculateValueHistoryList(betaValuesArray, memoryFactor=0.15):
    arraySize = betaValuesArray.shape[0]
    historyList = []
    i = arraySize - 1
    
    weightSum = 0.0
    memRatio = 1 - memoryFactor
    memWeight = 1.0
        
    while i >= 0:
        valueTuple = (betaValuesArray[i], memWeight)
        historyList.append(valueTuple)
        i -= 1
        weightSum += memWeight
        memWeight *= memRatio
    
    return historyList, weightSum

# Set the memory Factor to 0 for linear averaging
def getBetaPredictionByValueHistory(betaValuesArray, memoryFactor=0.15):
    historyList, sumOfWeights = calculateValueHistoryList(betaValuesArray, memoryFactor)
    sumOfWeightedValues = 0.0
    
    for historyTuple in historyList:
        value = historyTuple[0]
        weight = historyTuple[1]
        sumOfWeightedValues += (value * weight)
    
    return sumOfWeightedValues / sumOfWeights
   
# Must include at least 2 points. Last 2 points is trend line extension.
def getBetaPredictionByLinearRegression(betaValues, betaIndex, predictionIndex):
    regr = LinearRegression()
    betaIndex2d = np.reshape(betaIndex, newshape=(-1, 1))
    regr.fit(betaIndex2d, betaValues)
    slope = regr.coef_[0]
    #intercept = regr.intercept_
         
    predictionIndex2d = np.reshape(predictionIndex, newshape=(-1, 1))
    betaPredictionArray = regr.predict(predictionIndex2d)
    return betaPredictionArray
    
def getBetaPredictionByQuadraticRegression(betaValues, betaIndex, predictionIndex):
    quadModel = np.poly1d(np.polyfit(betaIndex, betaValues, 2))
    predictionValues = quadModel(predictionIndex)    
    #pipelineModel = Pipeline([('polyStep', PolynomialFeatures(degree=2)), ('linearStep', LinearRegression(fit_intercept=False))])
    #betaIndex2d = betaIndex[:, np.newaxis]
    #pipelineModel.fit(betaIndex2d, betaValues)
    #coefficients = pipelineModel.named_steps['linearStep'].coef_
    #predictionValues = pipelineModel.transform(predictionIndex[:, np.newaxis])
    return predictionValues
    