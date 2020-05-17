'''
Created on May 16, 2020

@author: William
'''
import Common as common
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from scipy import interpolate
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d


def newIntArray(size):
    return np.zeros(shape=(size,), dtype=np.int64)

def newFloatArray(size):
    return np.zeros(shape=(size,), dtype=np.float64)

def getSimpleFilter(size):
    x = np.ones(shape=(size,), dtype=np.float64)
    return x / size

def getSavitskyGovol(inputArray):
    return savgol_filter(inputArray, 7, 3)

def getGaussianAverage(inputArray, sigma):
    return gaussian_filter1d(inputArray, sigma, mode='nearest') 

def getSimpleMovingAverage(inputArray, smoothingSize):
    smoothingFilter = getSimpleFilter(smoothingSize) 
    smoothedArray = np.convolve(inputArray, smoothingFilter, 'same')
    return smoothedArray
    #f = interpolate.interp1d(xValues, inputArray, 'cubic')
    '''
    smoothed = newFloatArray(arraySize)
    for i in range(smoothSize - 1, arraySize):
        currentSum = 0
        for j in range(smoothSize):
            currentSum +=
    '''
    #return f(xValues)
def getPredictionsSIR(betaArray, gamma, sInit, iInit, rInit, countryPopulation):
    predictionSize = betaArray.shape[0]
    sPredict = newFloatArray(predictionSize)
    iPredict = newFloatArray(predictionSize)
    rPredict = newFloatArray(predictionSize)
    
    for i in range(predictionSize):
        if i == 0:
            sPredict[i] = sInit - (sInit * iInit) / countryPopulation
            iPredict[i] = iInit + (sInit * iInit) / countryPopulation - gamma * iInit
            rPredict[i] = rInit + gamma * iInit
          
        else:
            sPredict[i] = sPredict[i - 1] - (sInit * iInit) / countryPopulation
            iPredict[i] = iPredict[i - 1] + (sInit * iInit) / countryPopulation - gamma * iInit
            rPredict[i] = rPredict[i - 1] + gamma * iInit
     
    return sPredict, iPredict, rPredict

def getRandomBeta(meanValue):
    rv = np.random.normal(meanValue, meanValue / 3)
    return rv
    
def analyzeCountrySIR(tsData, countryName):
    tsSize = tsData.dateCount
    countryData = tsData.countryMap[countryName]
    
    countryPopulation = countryData.population
    
    confirmed = countryData.confirmed
    recovered = countryData.recovered
    deaths = countryData.deaths
    
    S = newIntArray(tsSize)
    I = newIntArray(tsSize)
    R = newIntArray(tsSize)
    sDelta = newIntArray(tsSize)
    iDelta = newIntArray(tsSize)
    rDelta = newIntArray(tsSize)
    betaObserved = newFloatArray(tsSize)
    gammaObserved = newFloatArray(tsSize)
    
    for i in range(tsSize):
        S[i] = countryPopulation - confirmed[i] - deaths[i] - recovered[i]
        I[i] = confirmed[i]
        R[i] = deaths[i] + recovered[i]
    
        if i == 0:
            sDelta[i] = 0
            iDelta[i] = 0
            rDelta[i] = 0
            betaObserved[i] = 0.0
            gammaObserved[i] = 0.0
        else:
            sDelta[i] = S[i] - S[i - 1]
            iDelta[i] = I[i] - I[i - 1]
            rDelta[i] = R[i] - R[i - 1]
            
            if I[i - 1] > 0:
                betaObserved[i] = (-1 * sDelta[i]) / ((S[i - 1] * I[i - 1]) / countryPopulation)
                gammaObserved[i] = rDelta[i] / I[i - 1]
            else:
                betaObserved[i] = 0
                gammaObserved[i] = 0
        
    #betaSmoothed = performInterpolation(betaObserved)
    betaSmoothed3 = getSimpleMovingAverage(betaObserved, 3)
    betaSmoothed7 = getSimpleMovingAverage(betaObserved, 7)
    #savitskyGovol = getSavitskyGovol(betaObserved)
    gaussianSmoothed3 = getGaussianAverage(betaObserved, 2.5)
    gaussianSmoothed2 = getGaussianAverage(betaObserved, 2)
    
    betaSevenDayAvg = np.sum(betaObserved[tsSize-7:]) / 7
    
    betaConstantTrend = newFloatArray(30)
    
    
    f1 = plt.figure(1, figsize=(10, 10))
    plt.plot(betaObserved)
    #plt.plot(betaSmoothed3)
    #plt.plot(betaSmoothed7)
    plt.plot(gaussianSmoothed3)
    #plt.plot(gaussianSmoothed2)
    
    
    f2 = plt.figure(2, figsize=(10, 10))
    
    
    plt.show()

if __name__ == '__main__':
    tsData = common.getTimeSeriesData()

    #for i in range(tsData.countryCount):
    #    countryName = tsData.countryIndex[i]
    #    countryData = tsData.countryMap[countryName]
    #    print("Country is: " + countryName + ", and first case was: " + str(tsData.dateIndex[countryData.firstIndex]))
    
    analyzeCountrySIR(tsData, "Brazil")
    
    print("Done")
    