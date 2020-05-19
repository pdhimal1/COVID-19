"""
William Austin
Prakash Dhimal
George Mason University
CS 584 Theory and Applications of Data Mining
Semester project: Predicting the Impact of COVID-19
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from sklearn import linear_model
import math


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
    # f = interpolate.interp1d(xValues, inputArray, 'cubic')
    '''
    smoothed = newFloatArray(arraySize)
    for i in range(smoothSize - 1, arraySize):
        currentSum = 0
        for j in range(smoothSize):
            currentSum +=
    '''
    # return f(xValues)
    
def boundAtZero(inArray):
    return np.array([max(x, 0) for x in inArray])
    
def getPredictionsSIR2(betaArray, gammaArray, sInit, iInit, rInit, countryPopulation):
    predictionSize = betaArray.shape[0]
    sPredict = newFloatArray(predictionSize)
    iPredict = newFloatArray(predictionSize)
    rPredict = newFloatArray(predictionSize)

    sPrevious = sInit
    iPrevious = iInit
    rPrevious = rInit

    for i in range(predictionSize):
        infectionCount = (betaArray[i] * (sPrevious * iPrevious)) / countryPopulation
        recoveryCount = gammaArray[i] * iPrevious
        sPredict[i] = sPrevious - infectionCount
        iPredict[i] = iPrevious + infectionCount - recoveryCount
        rPredict[i] = rPrevious + recoveryCount
        
        sPrevious = sPredict[i]
        iPrevious = iPredict[i]
        rPrevious = rPredict[i]

            # print("Day = " + str(i) + ": (S, I, R) = (" + str(sPredict[i]) + ", " + str(iPredict[i]) + ", " + str(rPredict[i]) + ")")
            # print(" -> Infections = " + str(infectionCount) + ", Recoveries = " + str(recoveryCount))

    return {"S": sPredict, "I": iPredict, "R": rPredict}


def getPredictionsSIR(betaArray, gammaArray, sInit, iInit, rInit, countryPopulation):
    predictionSize = betaArray.shape[0]
    sPredict = newFloatArray(predictionSize)
    iPredict = newFloatArray(predictionSize)
    rPredict = newFloatArray(predictionSize)

    for i in range(predictionSize):
        if i == 0:
            # sPredict[i] = sInit - (sInit * iInit) / countryPopulation
            # iPredict[i] = iInit + (sInit * iInit) / countryPopulation - gamma * iInit
            # rPredict[i] = rInit + gamma * iInit
            sPredict[i] = sInit
            iPredict[i] = iInit
            rPredict[i] = rInit
        else:
            infectionCount = (betaArray[i] * (sPredict[i - 1] * iPredict[i - 1])) / countryPopulation
            recoveryCount = gammaArray[i] * iPredict[i - 1]
            sPredict[i] = sPredict[i - 1] - infectionCount
            iPredict[i] = iPredict[i - 1] + infectionCount - recoveryCount
            rPredict[i] = rPredict[i - 1] + recoveryCount

            # print("Day = " + str(i) + ": (S, I, R) = (" + str(sPredict[i]) + ", " + str(iPredict[i]) + ", " + str(rPredict[i]) + ")")
            # print(" -> Infections = " + str(infectionCount) + ", Recoveries = " + str(recoveryCount))

    return [sPredict, iPredict, rPredict]


def getRandomVariable(meanValue, factor=8):
    rv = np.random.normal(meanValue, meanValue / factor)
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
        # rTemp[i] = deaths[i] + recovered[i]

        S[i] = countryPopulation - confirmed[i]  # - deaths[i] - recovered[i]
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

    # betaSmoothed = performInterpolation(betaObserved)
    # betaSmoothed3 = getSimpleMovingAverage(betaObserved, 3)
    # betaSmoothed7 = getSimpleMovingAverage(betaObserved, 7)
    # savitskyGovol = getSavitskyGovol(betaObserved)
    betaSmoothed = getGaussianAverage(betaObserved, 2.5)

    return {"S": S, "I": I, "R": R, "beta": betaObserved, "gamma": gammaObserved, "betaSmoothed": betaSmoothed}


def getStandardPredictions(tsData, countryName, rangeStart, rangeEnd, daysToPredict):
    countryData = tsData.countryMap[countryName]
    countryPopulation = countryData.population
    tsSizeSliced = rangeEnd - rangeStart + 1
    # elements = ["S", "I", "R", "beta", "gamma", "betaSmoothed"]
    # elementsIndex = {}
    # for i in range(len(elements)):
    #    elementsIndex[elements[i]] = i

    fullData = getObservedModelValues(tsData, countryName)
    # ------------------- slicedData = [x[rangeStart:rangeEnd] for x in fullData]

    sSliced = fullData["S"][rangeStart:rangeEnd + 1]
    iSliced = fullData["I"][rangeStart:rangeEnd + 1]
    rSliced = fullData["R"][rangeStart:rangeEnd + 1]
    betaSliced = fullData["beta"][rangeStart:rangeEnd + 1]
    gammaSliced = fullData["gamma"][rangeStart:rangeEnd + 1]
    betaSmoothedSliced = fullData["betaSmoothed"][rangeStart:rangeEnd]

    betaAvg, betaStdDev = np.mean(betaSliced[tsSizeSliced - 7:]), np.std(betaSliced[tsSizeSliced - 7:])
    gammaAvg, gammaStdDev = np.mean(gammaSliced[tsSizeSliced - 14:]), np.std(gammaSliced[tsSizeSliced - 14:])

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
    # betaContinueStart = tsSize * betaLinearCoefficient + betaLinearIntercept

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

    predictionsConstant = getPredictionsSIR(betaConstantTrend, gammaSampleArray1, sSliced[-1], iSliced[-1], rSliced[-1],
                                            countryPopulation)
    predictionsDownward = getPredictionsSIR(betaDownwardTrend, gammaSampleArray2, sSliced[-1], iSliced[-1], rSliced[-1],
                                            countryPopulation)
    predictionsContinueTrend = getPredictionsSIR(betaContinueTrend, gammaSampleArray3, sSliced[-1], iSliced[-1],
                                                 rSliced[-1], countryPopulation)

    return {"sirConstant": predictionsConstant, "sirDownward": predictionsDownward, \
            "sirContinueTrend": predictionsContinueTrend, "betaConstant": betaConstantTrend, \
            "betaDownward": betaDownwardTrend, "betaContinueTrend": betaContinueTrend}


# Note that the indices are sensitive here!
# Probably best for 3 week training and 1 week testing
def getTrainingSample(sSlice, iSlice, rSlice, betaSlice, trainDays, testDays):
    sample = np.zeros(shape=(1, 5), dtype=np.float64)
    # (1, 3, 7, 13, 21) => Predict a week or 2 weeks?
    # (1, 7, 14, 22, 31) => Predict a month 
    sample[0, 0] = betaSlice[trainDays - 1]
    sample[0, 1] = betaSlice[trainDays - 7]
    sample[0, 2] = betaSlice[trainDays - 14]
    sample[0, 3] = betaSlice[trainDays - 22]
    sample[0, 4] = betaSlice[trainDays - 31]

    label = betaSlice[-1]

    return sample, label


def buildSlidingWindowTrainingSet(tsData, trainDays, testDays):
    totalWindowSize = trainDays + testDays
    tsSize = tsData.dateCount
    countryMap = tsData.countryMap
    X = None
    y = None

    for countryName in tsData.countryIndex:
        countryData = countryMap[countryName]
        firstIndex = countryData.firstIndex
        dataMap = getObservedModelValues(tsData, countryName)
        windowStart = firstIndex
        i = 0

        countryTrainDataSize = tsSize - totalWindowSize - firstIndex + 1
        if countryTrainDataSize > 0:
            countryTrainSamples = np.zeros(shape=(countryTrainDataSize, 5), dtype=np.float64)
            countryTrainLabels = np.zeros(shape=(countryTrainDataSize,), dtype=np.float64)

            while windowStart + totalWindowSize <= tsSize:
                sSlice = dataMap["S"][windowStart:windowStart + totalWindowSize]
                iSlice = dataMap["I"][windowStart:windowStart + totalWindowSize]
                rSlice = dataMap["R"][windowStart:windowStart + totalWindowSize]
                betaSlice = dataMap["betaSmoothed"][windowStart:windowStart + totalWindowSize]

                sample, label = getTrainingSample(sSlice, iSlice, rSlice, betaSlice, trainDays, testDays)
                countryTrainSamples[i, :] = sample
                countryTrainLabels[i] = label

                windowStart += 1
                i += 1

            if X is None and y is None:
                X = countryTrainSamples
                y = countryTrainLabels
            else:
                X_tuple = (X, countryTrainSamples)
                X = np.concatenate(X_tuple, axis=0)

                y_tuple = (y, countryTrainLabels)
                y = np.concatenate(y_tuple)

    return X, y

def fillBetaTransitionQuadratic(startBeta, startBetaSlope, targetBeta, predictionDays):
    c = startBeta
    b = startBetaSlope
    a = (targetBeta - b * predictionDays) / (predictionDays * predictionDays)
    
    results = np.zeros(shape=(predictionDays), dtype=np.float64)
    
    for i in range(predictionDays):
        x = i + 1
        results[i] = (a * x * x) + (b * x) + c
    
    return results
        
def fillBetaTransitionLinear(startBeta, targetBeta, predictionDays):
    delta = targetBeta - startBeta
    
    results = np.zeros(shape=(predictionDays), dtype=np.float64)
    
    for i in range(predictionDays):
        results[i] = startBeta + (delta * ((i + 1) * 1.0 / predictionDays))
    
    return results

def addNoiseToArray(inputArray, factor=8):
    #arraySize = inputArray.shape[0]
    noisy = np.array([getRandomVariable(x, factor) for x in inputArray])
    return noisy

def computeMeanSquareError(trueValues, predictedValues):
    elementCount = 0
    sumSquaredErrors = 0.0 
    for yTrue, yPredicted in zip(trueValues, predictedValues):
        error = (yPredicted - yTrue)
        #currentWeight = index + 1
        sumSquaredErrors += (error * error) #currentWeight * (percentError * percentError)
        #sumOfWeights += currentWeight
        #index += 1
        elementCount += 1
    
    return math.sqrt(sumSquaredErrors / elementCount)
