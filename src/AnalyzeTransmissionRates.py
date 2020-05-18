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
from datetime import datetime, timedelta
import matplotlib.ticker as mticker

def getFullTickLabel(dayIndex):
    dayIndexInt = int(dayIndex)
    start = datetime(2020, 1, 22)
    offset = timedelta(days=dayIndex)
    tickDate = start + offset
    return tickDate.strftime("%b %d") + "  (" + str(dayIndexInt) + ")"
    
     

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
            infectionCount = (betaArray[i] * (sPredict[i - 1] * iPredict[i - 1])) / countryPopulation
            recoveryCount = gammaArray[i] * iPredict[i - 1]
            sPredict[i] = sPredict[i - 1] - infectionCount
            iPredict[i] = iPredict[i - 1] + infectionCount - recoveryCount
            rPredict[i] = rPredict[i - 1] + recoveryCount
            
            #print("Day = " + str(i) + ": (S, I, R) = (" + str(sPredict[i]) + ", " + str(iPredict[i]) + ", " + str(rPredict[i]) + ")")
            #print(" -> Infections = " + str(infectionCount) + ", Recoveries = " + str(recoveryCount))
     
    return [sPredict, iPredict, rPredict]

def getRandomVariable(meanValue):
    rv = np.random.normal(meanValue, meanValue / 8)
    return max(rv, 0.0)

def getRandomBeta(meanValue):
    rv = np.random.normal(meanValue, meanValue / 3.0)
    if rv < 0.0:
        return 0.0
    
    return rv

def getObservedModelValues(tsData, countryName):
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
        #rTemp[i] = deaths[i] + recovered[i]
        
        S[i] = countryPopulation - confirmed[i] #- deaths[i] - recovered[i]
        I[i] = confirmed[i] - deaths[i] - recovered[i]
        R[i] = deaths[i] + recovered[i]
    
        if i == 0:
            sDelta[i] = -1 * confirmed[i]
            iDelta[i] = confirmed[i] - deaths[i] - recovered[i]
            rDelta[i] = deaths[i] + recovered[i]
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
    #betaSmoothed3 = getSimpleMovingAverage(betaObserved, 3)
    #betaSmoothed7 = getSimpleMovingAverage(betaObserved, 7)
    #savitskyGovol = getSavitskyGovol(betaObserved)
    betaSmoothed = getGaussianAverage(betaObserved, 2.5)
    
    return {"S": S, "I": I, "R": R, "beta": betaObserved, "gamma": gammaObserved, "betaSmoothed": betaSmoothed}
    

def getStandardPredictions(tsData, countryName, rangeStart, rangeEnd, daysToPredict):
    countryData = tsData.countryMap[countryName]
    countryPopulation = countryData.population
    tsSizeSliced = rangeEnd - rangeStart + 1
    #elements = ["S", "I", "R", "beta", "gamma", "betaSmoothed"]
    #elementsIndex = {}
    #for i in range(len(elements)):
    #    elementsIndex[elements[i]] = i
    
    fullData = getObservedModelValues(tsData, countryName)
    #------------------- slicedData = [x[rangeStart:rangeEnd] for x in fullData]
    
    sSliced = fullData["S"][rangeStart:rangeEnd + 1]
    iSliced = fullData["I"][rangeStart:rangeEnd + 1]
    rSliced = fullData["R"][rangeStart:rangeEnd + 1]
    betaSliced = fullData["beta"][rangeStart:rangeEnd + 1]
    gammaSliced = fullData["gamma"][rangeStart:rangeEnd + 1]
    betaSmoothedSliced = fullData["betaSmoothed"][rangeStart:rangeEnd]
    
    betaAvg, betaStdDev = np.mean(betaSliced[tsSizeSliced-7:]), np.std(betaSliced[tsSizeSliced-7:])
    gammaAvg, gammaStdDev = np.mean(gammaSliced[tsSizeSliced-14:]), np.std(gammaSliced[tsSizeSliced-14:])
    
    # At index 0, this will be the same as the last value of the real data
    predictionDays = daysToPredict + 1 
    gammaSampleArray1 = newFloatArray(predictionDays)
    gammaSampleArray2 = newFloatArray(predictionDays)
    gammaSampleArray3 = newFloatArray(predictionDays)
    
    betaConstantTrend = newFloatArray(predictionDays)
    betaDownwardTrend = newFloatArray(predictionDays)
    betaContinueTrend = newFloatArray(predictionDays)
    
    regr = linear_model.LinearRegression()
    regr.fit(np.arange(tsSizeSliced - 7, tsSizeSliced).reshape(-1, 1), betaSmoothedSliced[-7:])
    betaLinearCoefficient = regr.coef_[0]
    betaLinearIntercept = regr.intercept_ 
    #betaContinueStart = tsSize * betaLinearCoefficient + betaLinearIntercept
    
    for i in range(predictionDays):
        if i == 0:
            betaConstantTrend[i] = betaAvg
            betaDownwardTrend[i] = betaAvg
            betaContinueTrend[i] = betaAvg
        else:
            betaConstantTrend[i] = getRandomVariable(betaAvg)
            
            predictionCompletionRatio = i / (predictionDays - 1)
            betaDownwardMean = max(betaAvg - (betaAvg * predictionCompletionRatio), betaAvg / 10.0)
            betaDownwardTrend[i] = getRandomVariable(betaDownwardMean)
             
            betaContinueMean = max(betaAvg + (betaLinearCoefficient * i), betaAvg / 10.0)
            betaContinueTrend[i] = getRandomVariable(betaContinueMean)
            
        gammaSampleArray1[i] = getRandomVariable(gammaAvg)
        gammaSampleArray2[i] = getRandomVariable(gammaAvg)
        gammaSampleArray3[i] = getRandomVariable(gammaAvg)
         
    predictionsConstant = getPredictionsSIR(betaConstantTrend, gammaSampleArray1, sSliced[-1], iSliced[-1], rSliced[-1], countryPopulation)
    predictionsDownward = getPredictionsSIR(betaDownwardTrend, gammaSampleArray2, sSliced[-1], iSliced[-1], rSliced[-1], countryPopulation)
    predictionsContinueTrend = getPredictionsSIR(betaContinueTrend, gammaSampleArray3, sSliced[-1], iSliced[-1], rSliced[-1], countryPopulation)

    return {"sirConstant": predictionsConstant, "sirDownward": predictionsDownward, \
            "sirContinueTrend": predictionsContinueTrend, "betaConstant": betaConstantTrend, \
            "betaDownward": betaDownwardTrend, "betaContinueTrend": betaContinueTrend}

def updateTicks(x, pos):
    return getFullTickLabel(x)   
    
def analyzeCountrySIR(tsData, countryName):
    '''
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
    #betaSmoothed3 = getSimpleMovingAverage(betaObserved, 3)
    #betaSmoothed7 = getSimpleMovingAverage(betaObserved, 7)
    #savitskyGovol = getSavitskyGovol(betaObserved)
    gaussianSmoothed3 = getGaussianAverage(betaObserved, 2.5)
    #gaussianSmoothed2 = getGaussianAverage(betaObserved, 2)
    
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
    
    '''
    
    tsSize = tsData.dateCount
    countryData = tsData.countryMap[countryName]
    countryPopulation = countryData.population
    futurePredictionDays = 60 # Arrays will be 31
    cvPredictionDays = 30
    
    fullData = getObservedModelValues(tsData, countryName)
    
    historicalRange = np.arange(tsSize)
    predictionRange = np.arange(tsSize - 1, tsSize + futurePredictionDays)
    
    futurePredictions = getStandardPredictions(tsData, countryName, 0, tsSize - 1, futurePredictionDays)
    cvPredictions = getStandardPredictions(tsData, countryName, 0, tsSize - cvPredictionDays - 1, cvPredictionDays)
    cvTrainRange = np.arange(0, tsSize - cvPredictionDays)
    cvTestRange = np.arange(tsSize - cvPredictionDays - 1, tsSize)
    cvFullRange = np.arange(0, tsSize)
    
    figure1, axis1 = plt.subplots()
    figure1.set_size_inches(7.5, 7.5)
    axis1.plot(historicalRange, fullData["beta"], label="Observed beta", color="gray")
    #plt.plot(betaSmoothed3)
    #plt.plot(betaSmoothed7)
    axis1.plot(historicalRange, fullData["betaSmoothed"], label="Smoothed beta", color="blue")
    #plt.plot(gaussianSmoothed2)
    axis1.set_xlabel("Day Number")
    axis1.set_ylabel("Transmission Rate")
    #axis1.xticks(np.arange(0, tsSize, 10))
    axis1.xaxis.set_major_locator(plticker.MultipleLocator(base=10))
    axis1.xaxis.set_tick_params(rotation=60)
    axis1.xaxis.set_major_formatter(mticker.FuncFormatter(updateTicks))
    axis1.legend()
    axis1.set_title("Historical Transmission Rate for " + countryName, fontsize="xx-large")
    #figure1.suptitle("Historical Transmission Rate for " + countryName)
    
    
    figure2, axis2 = plt.subplots()
    figure2.set_size_inches(7.5, 7.5)

    axis2.axvspan(tsSize - 1, tsSize + futurePredictionDays - 2, alpha=0.3, color='gray')
    #plt.plot(historicalRange, S)
    axis2.plot(historicalRange, fullData["I"], label="Infected (Confirmed Cases)", color="darkblue")
    axis2.plot(historicalRange, fullData["R"], label="Recovered (Deaths + Recovered Cases)", color="mediumblue")
    
    #plt.plot(predictionRange, SP1)
    axis2.plot(predictionRange, futurePredictions["sirConstant"][1], label="Infected Prediction (constant beta)", color="firebrick")
    axis2.plot(predictionRange, futurePredictions["sirConstant"][2], label="Recovered Prediction (constant beta)", color="lightcoral")
    
    axis2.plot(predictionRange, futurePredictions["sirDownward"][1], label="Infected Prediction (decreasing beta)", color="forestgreen")
    axis2.plot(predictionRange, futurePredictions["sirDownward"][2], label="Recovered Prediction (decreasing beta)", color="limegreen")
    
    axis2.plot(predictionRange, futurePredictions["sirContinueTrend"][1], label="Infected Prediction (continue beta trend)", color="royalblue")
    axis2.plot(predictionRange, futurePredictions["sirContinueTrend"][2], label="Recovered Prediction (continue beta trend)", color="cornflowerblue")    
    
    axis2.set_xlabel("Day Number")
    axis2.set_ylabel("Number of Individuals")
    #axis2.xticks(np.arange(0, tsSize + predictionDays, 10))
    axis2.xaxis.set_major_locator(plticker.MultipleLocator(base=10))
    axis2.xaxis.set_tick_params(rotation=60)
    axis2.xaxis.set_major_formatter(mticker.FuncFormatter(updateTicks))
    axis2.legend()
    axis2.set_title("60 Day Predictions for " + countryName + " (Infected and Recovered)", fontsize="xx-large")
    #figure2.suptitle("60 Day Predictions for " + countryName + " (Infected and Recovered)")
    
    figure3, axis3 = plt.subplots()
    figure3.set_size_inches(7.5, 7.5)
    axis3.plot(predictionRange, futurePredictions["betaConstant"], label="Constant beta", color="firebrick")
    axis3.plot(predictionRange, futurePredictions["betaDownward"], label="Decreasing beta", color="forestgreen")
    axis3.plot(predictionRange, futurePredictions["betaContinueTrend"], label="Continue beta trend", color="royalblue")
    axis3.set_xlabel("Day Number")
    axis3.set_ylabel("Transmission Rate")
    #axis3.xticks(np.arange(tsSize - 1, tsSize + predictionDays, 5))
    axis3.xaxis.set_major_locator(plticker.MultipleLocator(base=5))
    axis3.xaxis.set_tick_params(rotation=60)
    axis3.xaxis.set_major_formatter(mticker.FuncFormatter(updateTicks))
    axis3.legend()
    axis3.set_title("Prediction Beta Values for " + countryName, fontsize="xx-large")
    #figure3.suptitle("Prediction Beta Values for " + countryName)
    
    figure4, axis4 = plt.subplots()
    figure4.set_size_inches(7.5, 7.5)
    #figure4.canvas.draw()

    axis4.axvspan(tsSize - cvPredictionDays - 1, tsSize, alpha=0.3, color='gray')
    #plt.plot(historicalRange, S)
    axis4.plot(cvFullRange, fullData["I"], label="Infected (Confirmed Cases)", color="darkblue", linewidth=4)
    axis4.plot(cvFullRange, fullData["R"], label="Recovered (Deaths + Recovered Cases)", color="mediumblue", linewidth=4)
    
    #plt.plot(predictionRange, SP1)
    axis4.plot(cvTestRange, cvPredictions["sirConstant"][1], label="Infected Prediction (constant beta)", color="firebrick")
    axis4.plot(cvTestRange, cvPredictions["sirConstant"][2], label="Recovered Prediction (constant beta)", color="lightcoral")
    
    axis4.plot(cvTestRange, cvPredictions["sirDownward"][1], label="Infected Prediction (decreasing beta)", color="forestgreen")
    axis4.plot(cvTestRange, cvPredictions["sirDownward"][2], label="Recovered Prediction (decreasing beta)", color="limegreen")
    
    axis4.plot(cvTestRange, cvPredictions["sirContinueTrend"][1], label="Infected Prediction (continue beta trend)", color="royalblue")
    axis4.plot(cvTestRange, cvPredictions["sirContinueTrend"][2], label="Recovered Prediction (continue beta trend)", color="cornflowerblue")    
    
    axis4.set_xlabel("Day Number")
    axis4.set_ylabel("Number of Individuals")
    #axis2.xticks(np.arange(0, tsSize + predictionDays, 10))
    axis4.xaxis.set_major_locator(plticker.MultipleLocator(base=10))
    axis4.xaxis.set_tick_params(rotation=60)
    axis4.xaxis.set_major_formatter(mticker.FuncFormatter(updateTicks))
    #axis4.xaxis.
    #print("Tick Labels = " + str(axis4.get_xticklabels()))
    
    #axis_labels = [getFullTickLabel(int(q)) for q in axis4.get_xticks().tolist()]
    #print(axis_labels)
    #axis4.set_xlabels(axis_labels)
    #axis4.xticks(rotation=60)
     #$[getFullTickLabel(str(i)) for i in range(0, tsSize, 10)]
    #axis_labels = [item.get_text() for item in axis4.get_xticklabels()]
    #axis4.set_xticklabels(axis_labels)
    
    axis4.legend()
    axis4.set_title("Compare 30 Day Predictions vs. Actual for " + countryName + " (Infected and Recovered)", fontsize="xx-large")
    #figure2.suptitle("60 Day Predictions for " + countryName + " (Infected and Recovered)")
    
    plt.show()

if __name__ == '__main__':
    tsData = common.getTimeSeriesData()

    #for i in range(tsData.countryCount):
    #    countryName = tsData.countryIndex[i]
    #    countryData = tsData.countryMap[countryName]
    #    print("Country is: " + countryName + ", and first case was: " + str(tsData.dateIndex[countryData.firstIndex]))
    
    for c in ["China", "Spain", "Germany", "France", "Italy"]: #Brazil", "Russia", "Nigeria", "Mexico"]:
        analyzeCountrySIR(tsData, c)
    #analyzeCountrySIR(tsData, "US")
        
    print("Done")
    