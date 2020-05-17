'''
Created on May 16, 2020

@author: William
'''
import Common as common
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import seaborn as sb
from scipy import interpolate
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from sklearn import linear_model
import statistics


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
def getPredictionsSIR(betaArray, gammaArray, sInit, iInit, rInit, countryPopulation):
    predictionSize = betaArray.shape[0]
    sPredict = newFloatArray(predictionSize)
    iPredict = newFloatArray(predictionSize)
    rPredict = newFloatArray(predictionSize)
    
    for i in range(predictionSize):
        if i == 0:
            #sPredict[i] = sInit - (sInit * iInit) / countryPopulation
            #iPredict[i] = iInit + (sInit * iInit) / countryPopulation - gamma * iInit
            #rPredict[i] = rInit + gamma * iInit
            sPredict[i] = sInit
            iPredict[i] = iInit
            rPredict[i] = rInit
          
        else:
            sPredict[i] = sPredict[i - 1] - betaArray[i] * (sPredict[i - 1] * iPredict[i - 1]) / countryPopulation
            iPredict[i] = iPredict[i - 1] + betaArray[i] * (sPredict[i - 1] * iPredict[i - 1]) / countryPopulation - gammaArray[i] * iPredict[i - 1]
            rPredict[i] = rPredict[i - 1] + gammaArray[i] * iPredict[i - 1]
     
    return sPredict, iPredict, rPredict

def getRandomVariable(meanValue, stdDevValue):
    rv = np.random.normal(meanValue, meanValue / 11)
    return max(rv, 0.0)

def getRandomBeta(meanValue):
    rv = np.random.normal(meanValue, meanValue / 3.0)
    if rv < 0.0:
        return 0.0
    
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
    
    betaAvg, betaStdDev = np.mean(betaObserved[tsSize-7:]), np.std(betaObserved[tsSize-7:])
    gammaAvg, gammaStdDev = np.mean(gammaObserved[tsSize-7:]), np.std(gammaObserved[tsSize-7:])
    
    predictionDays = 61
    gammaSampleArray = newFloatArray(predictionDays)
    gammaSampleArray2 = newFloatArray(predictionDays)
    gammaSampleArray3 = newFloatArray(predictionDays)
    
    betaConstantTrend = newFloatArray(predictionDays)
    betaDownwardTrend = newFloatArray(predictionDays)
    betaContinueTrend = newFloatArray(predictionDays)
    
    regr = linear_model.LinearRegression()
    regr.fit(np.arange(tsSize - 7, tsSize).reshape(-1, 1), gaussianSmoothed3[-7:])
    betaLinearCoefficient = regr.coef_[0]
    betaLinearIntercept = regr.intercept_ 
    #betaContinueStart = tsSize * betaLinearCoefficient + betaLinearIntercept
    
    for i in range(predictionDays):
        betaConstantTrend[i] = getRandomVariable(betaAvg, betaStdDev)
        
        dayInMonthFraction = i / predictionDays
        betaDownwardMean = betaAvg - (betaAvg * dayInMonthFraction)
        betaDownwardStdDev = betaStdDev * (betaDownwardMean / betaAvg) 
        betaDownwardTrend[i] = getRandomVariable(betaDownwardMean, betaDownwardStdDev)
         
        betaContinueMean = max(betaLinearIntercept + (betaLinearCoefficient * (i + tsSize)), 0.0)
        betaContinueStdDev = betaStdDev * (betaContinueMean / betaAvg)
        betaContinueTrend[i] = getRandomVariable(betaContinueMean, betaContinueStdDev)
        
        gammaSampleArray[i] = getRandomVariable(gammaAvg, gammaStdDev)
        gammaSampleArray2[i] = getRandomVariable(gammaAvg, gammaStdDev)
        gammaSampleArray3[i] = getRandomVariable(gammaAvg, gammaStdDev)
         
    SP1, IP1, RP1 = getPredictionsSIR(betaConstantTrend, gammaSampleArray, S[-1], I[-1], R[-1], countryPopulation)
    SP2, IP2, RP2 = getPredictionsSIR(betaDownwardTrend, gammaSampleArray2, S[-1], I[-1], R[-1], countryPopulation)
    SP3, IP3, RP3 = getPredictionsSIR(betaContinueTrend, gammaSampleArray3, S[-1], I[-1], R[-1], countryPopulation)
    
    historicalRange = np.arange(tsSize)
    predictionRange = np.arange(predictionDays) + tsSize - 1
      
    
    figure1, axis1 = plt.subplots()
    figure1.set_size_inches(7.5, 7.5)
    axis1.plot(historicalRange, betaObserved, label="Observed beta", color="gray")
    #plt.plot(betaSmoothed3)
    #plt.plot(betaSmoothed7)
    axis1.plot(historicalRange, gaussianSmoothed3, label="Smoothed beta", color="blue")
    #plt.plot(gaussianSmoothed2)
    axis1.set_xlabel("Day Number")
    axis1.set_ylabel("Transmission Rate")
    #axis1.xticks(np.arange(0, tsSize, 10))
    axis1.xaxis.set_major_locator(plticker.MultipleLocator(base=10))
    axis1.legend()
    axis1.set_title("Historical Transmission Rate for " + countryName, fontsize="xx-large")
    #figure1.suptitle("Historical Transmission Rate for " + countryName)
    
    
    figure2, axis2 = plt.subplots()
    figure2.set_size_inches(7.5, 7.5)

    axis2.axvspan(tsSize - 1, tsSize + predictionDays - 2, alpha=0.3, color='gray')
    #plt.plot(historicalRange, S)
    axis2.plot(historicalRange, I, label="Infected (Confirmed Cases)", color="darkblue")
    axis2.plot(historicalRange, R, label="Recovered (Deaths + Recovered Cases)", color="mediumblue")
    
    #plt.plot(predictionRange, SP1)
    axis2.plot(predictionRange, IP1, label="Infected Prediction (constant beta)", color="firebrick")
    axis2.plot(predictionRange, RP1, label="Recovered Prediction (constant beta)", color="lightcoral")
    
    axis2.plot(predictionRange, IP2, label="Infected Prediction (decreasing beta)", color="forestgreen")
    axis2.plot(predictionRange, RP2, label="Recovered Prediction (decreasing beta)", color="limegreen")
    
    axis2.plot(predictionRange, IP3, label="Infected Prediction (continue beta trend)", color="royalblue")
    axis2.plot(predictionRange, RP3, label="Recovered Prediction (continue beta trend)", color="cornflowerblue")    
    
    axis2.set_xlabel("Day Number")
    axis2.set_ylabel("Number of Individuals")
    #axis2.xticks(np.arange(0, tsSize + predictionDays, 10))
    axis2.xaxis.set_major_locator(plticker.MultipleLocator(base=10))
    axis2.legend()
    axis2.set_title("60 Day Predictions for " + countryName + " (Infected and Recovered)", fontsize="xx-large")
    #figure2.suptitle("60 Day Predictions for " + countryName + " (Infected and Recovered)")
    
    figure3, axis3 = plt.subplots()
    figure3.set_size_inches(7.5, 7.5)
    axis3.plot(predictionRange, betaConstantTrend, label="Constant beta", color="firebrick")
    axis3.plot(predictionRange, betaDownwardTrend, label="Decreasing beta", color="forestgreen")
    axis3.plot(predictionRange, betaContinueTrend, label="Continue beta trend", color="royalblue")
    axis3.set_xlabel("Day Number")
    axis3.set_ylabel("Transmission Rate")
    #axis3.xticks(np.arange(tsSize - 1, tsSize + predictionDays, 5))
    axis3.xaxis.set_major_locator(plticker.MultipleLocator(base=5))
    axis3.legend()
    axis3.set_title("Prediction Beta Values for " + countryName, fontsize="xx-large")
    #figure3.suptitle("Prediction Beta Values for " + countryName)
    
    plt.show()

if __name__ == '__main__':
    tsData = common.getTimeSeriesData()

    #for i in range(tsData.countryCount):
    #    countryName = tsData.countryIndex[i]
    #    countryData = tsData.countryMap[countryName]
    #    print("Country is: " + countryName + ", and first case was: " + str(tsData.dateIndex[countryData.firstIndex]))
    
    analyzeCountrySIR(tsData, "US")
        
    print("Done")
    