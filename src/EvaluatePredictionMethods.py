"""
William Austin
Prakash Dhimal
George Mason University
CS 584 Theory and Applications of Data Mining
Semester project: Predicting the Impact of COVID-19
"""
import Common as common
import Utilities as util
import BetaPredictionMethods as bpm
import numpy as np

countryMinimumMap = {}
modelScoreMap = {}
modelTypeSampleSizeMap = {}
tupleList = []

def extractToMap(modelDescriptor):
    parts = modelDescriptor.split(", ")
    descriptorMap = {}
    for part in parts:
        keyValueList = part.split("=")
        descriptorMap[keyValueList[0]] = keyValueList[1]
    
    return descriptorMap
        

def updateMapping(countryName, modelDescriptor, scoreValue, sampleSize, combinedModelType):
    newTuple = (countryName, combinedModelType, scoreValue, sampleSize)
    tupleList.append(newTuple)
    
    # Update the minimum
    countryMinimumTuple = countryMinimumMap.get(countryName)
    
    if countryMinimumTuple is None or scoreValue < countryMinimumTuple[0]:
        countryMinimumMap[countryName] = (scoreValue, modelDescriptor)
    
    # Model score map
    countryScoreMap = modelScoreMap.get(modelDescriptor)
    if countryScoreMap is None:
        newCountryScoreMap = {}
        newCountryScoreMap[countryName] = scoreValue
        modelScoreMap[modelDescriptor] = newCountryScoreMap
    else:
        countryScoreMap[countryName] = scoreValue
    
    # Model Type -> Sample Size Maps
    '''
    sampleSizeDetails = modelTypeSampleSizeMap.get(combinedModelType)
    
    if sampleSizeDetails is None:
        newSampleSizeDetails = {}
        newSampleSizeDetails[str(sampleSize)] = (scoreValue, 1)
        modelTypeSampleSizeMap[combinedModelType] = newSampleSizeDetails
    else:
        sampleSizeSummaryTuple = sampleSizeDetails.get(str(sampleSize))
        if sampleSizeSummaryTuple is None:
            newSampleSizeSummaryTuple = (scoreValue, 1)
            sampleSizeDetails[str(sampleSize)] = newSampleSizeSummaryTuple
        else:
            oldScoreSum = sampleSizeSummaryTuple[0]
            oldCount = sampleSizeSummaryTuple[1]
            sampleSizeDetails[str(sampleSize)] = (oldScoreSum + scoreValue, oldCount + 1)
    '''
if __name__ == '__main__':
    timeSeriesData = common.getTimeSeriesData()
    timeSeriesSize = timeSeriesData.dateCount
    
    countryIndex = timeSeriesData.countryIndex
    countryMap = timeSeriesData.countryMap
    predictionDays = 14
    
    for countryName in countryIndex:
        countryData = countryMap[countryName]
        countryFirstIndex = countryData.firstIndex
        countryPopulation = countryData.population
        countryArraySize = timeSeriesSize - countryFirstIndex
        
        if countryArraySize > predictionDays:
            resultsMap = util.getObservedModelValues(timeSeriesData, countryName)
            
            sObserved = resultsMap["S"][countryFirstIndex:]
            iObserved = resultsMap["I"][countryFirstIndex:]
            rObserved = resultsMap["R"][countryFirstIndex:]
            gamma = resultsMap["gamma"][countryFirstIndex:]
            
            # We aren't evaluating gamma, so just get the best value
            avgGamma = np.sum(gamma) / gamma.shape[0]
            
            beta = resultsMap["beta"][countryFirstIndex:]
            
            gammaArray = np.full(shape=(predictionDays,), fill_value=avgGamma)
            
            # Add a day because we want to prefill the 0 index with the info from 
            # the last day of the observed period 
            observedDaysCount = countryArraySize - predictionDays
            observedRange = np.arange(0, observedDaysCount)
            predictionRange = np.arange(observedDaysCount, countryArraySize)
            
            sInit = sObserved[observedDaysCount - 1]
            iInit = iObserved[observedDaysCount - 1]
            rInit = rObserved[observedDaysCount - 1]
            sTrueValues = sObserved[predictionRange]
            
            countryBestMethod = None
            countryBestSSE = None
            
            smoothingValues = [1.5, 2.0, 2.5, 3.0, 3.5]
            for smoothingValue in smoothingValues:
                smoothedBetaObserved = util.getGaussianAverage(beta[observedRange], smoothingValue)
                
                sampleSizes = [1, 2, 3, 5, 7, 10, 14, 21, 31, 44, 60]
                
                for sampleSize in sampleSizes:
                    if sampleSize <= observedDaysCount:
                        sampleRange = np.arange(observedDaysCount-sampleSize, observedDaysCount)
                        modelDescriptor = ""
                        
                        if sampleSize >= 2:
                            betaPredictions = bpm.getBetaPredictionByLinearRegression(smoothedBetaObserved[sampleRange], sampleRange, predictionRange)
                            betaPredictions = util.boundAtZero(betaPredictions)
                            predictionsMap = util.getPredictionsSIR2(betaPredictions, gammaArray, sInit, iInit, rInit, countryPopulation)
                            relativeSSE = util.computeMeanSquareError(sTrueValues, predictionsMap["S"])
                            modelDescriptor = "Type=LinearRegression, SampleSize=" + str(sampleSize) + ", SmoothingValue = " + str(smoothingValue)
                            updateMapping(countryName, modelDescriptor, relativeSSE, sampleSize, "LinearRegression")
                        
                        if sampleSize >= 3:
                            betaPredictions = bpm.getBetaPredictionByQuadraticRegression(smoothedBetaObserved[sampleRange], sampleRange, predictionRange)
                            #betaPredictions = bpm.getBetaPredictionByLinearRegression(smoothedBetaObserved[sampleRange], sampleRange, predictionRange)
                            betaPredictions = util.boundAtZero(betaPredictions)
                            predictionsMap = util.getPredictionsSIR2(betaPredictions, gammaArray, sInit, iInit, rInit, countryPopulation)
                            relativeSSE = util.computeMeanSquareError(sTrueValues, predictionsMap["S"])
                            modelDescriptor = "Type=QuadraticRegression, SampleSize=" + str(sampleSize) + ", SmoothingValue=" + str(smoothingValue)
                            updateMapping(countryName, modelDescriptor, relativeSSE, sampleSize, "QuadraticRegression")
                        
                            lastBeta = smoothedBetaObserved[observedDaysCount-1]
                            lastBetaSlope = smoothedBetaObserved[observedDaysCount-1] - smoothedBetaObserved[observedDaysCount-2]
                            memoryValues = [0.0, 0.001, 0.05, 0.01, 0.05, 0.1, 0.15]
                            
                            for mv in memoryValues:
                                historyTypes = ["SlopeHistory", "ValueHistory"]
                                for historyType in  historyTypes:
                                    targetBetaValue = None
                                    if(historyType == "SlopeHistory"):
                                        targetBetaValue = bpm.getBetaPredictionBySlopeHistory(smoothedBetaObserved[sampleRange], \
                                            lastBeta, predictionDays, mv, 1)
                                    elif historyType == "ValueHistory":
                                        targetBetaValue = bpm.getBetaPredictionByValueHistory(smoothedBetaObserved[sampleRange], mv)
                                
                                    fillTypes = ["Constant", "Linear", "Quadratic"]
                                    for fillType in fillTypes:
                                        betaPredictions = None
                                        if fillType == "Constant":
                                            betaPredictions = np.full(shape=(predictionDays,), fill_value=targetBetaValue)
                                        elif fillType == "Linear":
                                            betaPredictions = util.fillBetaTransitionLinear(lastBeta, targetBetaValue, predictionDays)
                                        elif fillType == "Quadratic":
                                            betaPredictions = util.fillBetaTransitionQuadratic(lastBeta, lastBetaSlope, targetBetaValue, predictionDays)
                                        
                                        betaPredictions = util.boundAtZero(betaPredictions)
                                        predictionsMap = util.getPredictionsSIR2(betaPredictions, gammaArray, sInit, iInit, rInit, countryPopulation)
                                        relativeSSE = util.computeMeanSquareError(sTrueValues, predictionsMap["S"])
                                        modelDescriptor = "Type=" + historyType +", SampleSize=" + str(sampleSize) + ", SmoothingValue=" + str(smoothingValue) \
                                            + ", MemoryFactor=" + str(mv) + ", FillType="  + fillType
                                        updateMapping(countryName, modelDescriptor, relativeSSE, sampleSize, historyType + "/" + fillType)
                                        
                                '''
                                #Linear fill
                                betaValueFromSlope = bpm.getBetaPredictionBySlopeHistory(smoothedBetaObserved[sampleRange], \
                                                    lastBeta, predictionDays, mv, 1)
                                betaPredictions2 = util.fillBetaTransitionLinear(lastBeta, betaValueFromSlope, predictionDays)
                                betaPredictions2 = util.boundAtZero(betaPredictions2)
                                predictionsMap2 = util.getPredictionsSIR2(betaPredictions2, gammaArray, sInit, iInit, rInit, countryPopulation)
                                relativeSSE2 = util.computeMeanSquareError(sTrueValues, predictionsMap2["S"])
                                modelDescriptor = "Type=SlopeHistory, SampleSize=" + str(sampleSize) + ", SmoothingValue=" + str(smoothingValue) \
                                    + ", MemoryFactor=" + str(mv) + ", FillType=Linear"
                                updateMapping(countryName, modelDescriptor, relativeSSE2)
                                
                                # Quadratic Fill
                                betaPredictions2 = util.fillBetaTransitionQuadratic(lastBeta, lastBetaSlope, betaValueFromSlope, predictionDays)
                                betaPredictions2 = util.boundAtZero(betaPredictions2)
                                predictionsMap2 = util.getPredictionsSIR2(betaPredictions2, gammaArray, sInit, iInit, rInit, countryPopulation)
                                relativeSSE2 = util.computeMeanSquareError(sTrueValues, predictionsMap2["S"])
                                modelDescriptor = "Type=SlopeHistory, SampleSize=" + str(sampleSize) + ", SmoothingValue=" + str(smoothingValue) \
                                    + ", MemoryFactor=" + str(mv) + ", FillType=Quadratic"
                                updateMapping(countryName, modelDescriptor, relativeSSE2)
                                
                                # Constant
                                betaPredictions2 = np.full(shape=(predictionDays,), fill_value=betaValueFromSlope)
                                betaPredictions2 = util.boundAtZero(betaPredictions2)
                                predictionsMap2 = util.getPredictionsSIR2(betaPredictions2, gammaArray, sInit, iInit, rInit, countryPopulation)
                                relativeSSE2 = util.computeMeanSquareError(sTrueValues, predictionsMap2["S"])
                                modelDescriptor = "Type=SlopeHistory, SampleSize=" + str(sampleSize) + ", SmoothingValue=" + str(smoothingValue) \
                                    + ", MemoryFactor=" + str(mv) + ", FillType=Constant"
                                updateMapping(countryName, modelDescriptor, relativeSSE2)
                                '''
        
        print("Completed models for country: " + countryName)
    
    print("\n===================")
    print("Summarizing Results")
    print("===================")
    
    countTypeMap = {}
    for key, value in countryMinimumMap.items():
        rmseScore = value[0]
        modelDescriptor = value[1]
        
        print("Minimum Score for " + key + " is: " + str(value))
        if rmseScore > 0.0:
            minMap = extractToMap(modelDescriptor)
            modelString = minMap["Type"]
            if modelString == "SlopeHistory" or modelString == "ValueHistory":
                modelString = modelString + "/" + minMap["FillType"]
            
            current = countTypeMap.get(modelString)
            if current is None:
                countTypeMap[modelString] = 1
            else:
                countTypeMap[modelString] = current + 1
    
    print("======")
    print("Min map by type = " + str(countTypeMap))
        
    
    finalModelScores = []
    for modelKey, scoreMap in modelScoreMap.items():
        normalizedModelScore = 0.0
        countryCount = 0
        for countryName, countryScore in scoreMap.items():
            minimumCountryTuple = countryMinimumMap.get(countryName)
            if minimumCountryTuple is not None and minimumCountryTuple[0] > 0.0:
                relativeScore = countryScore / minimumCountryTuple[0]
                normalizedModelScore += relativeScore
                countryCount += 1
        
        modelScore = normalizedModelScore / countryCount
        modelScoreTuple = (modelScore, modelKey, countryCount)
        finalModelScores.append(modelScoreTuple)
    
    finalModelScores.sort(key=lambda x: x[0])
    
    print("\n================")
    print("Printing top models.")
    print("====================")
    totalModelCount = len(finalModelScores)
    print("Total model count = " + str(totalModelCount))
    
    for i in range(min(totalModelCount, 50)):
        modelDetails = finalModelScores[i]
        print(str(i + 1) + ". Score = " + str(modelDetails[0]) + ", Country Count = " + str(modelDetails[2]) + ", Model Key = " + modelDetails[1])
    
    print("\n==================")
    print("Printing worst models.")
    print("======================")
    
    for i in range(min(totalModelCount, 50)):
        index = totalModelCount - i - 1
        modelDetails = finalModelScores[index]
        print(str(index + 1) + ". Score = " + str(modelDetails[0]) + ", Country Count = " + str(modelDetails[2]) + ", Model Key = " + modelDetails[1])
        
    #print("Country Best for " + countryName + " was SSE = " + str(countryBestSSE) + ", and method is: " + countryBestMethod)
    
    print("\n==================")
    print("Printing trends of averge model performance vs. sample size.")
    print("======================")
    '''
    for combinedModelType, ssDetailsMap in modelTypeSampleSizeMap.items():
        avgScoresBySS = []
        for ss, ssSummaryTuple in ssDetailsMap.items():
            ssTotalSum = ssSummaryTuple[0]
            ssCount = ssSummaryTuple[1]
            ssAvgTuple = (int(ss), ssTotalSum / ssCount)
            avgScoresBySS.append()
    '''        
    
    typeSampleSizeScoresMap = {}
    
    for tuple in tupleList:
        countryName = tuple[0]
        combinedModel = tuple[1]
        scoreValue = tuple[2]
        sampleSize = tuple[3]
        
        minimumCountryTuple = countryMinimumMap.get(countryName)
        if minimumCountryTuple is not None and minimumCountryTuple[0] > 0.0:
            relativeScore = scoreValue / minimumCountryTuple[0]
            key = combinedModel + ":" + str(sampleSize)
            currentTuple = typeSampleSizeScoresMap.get(key)
            
            if currentTuple is None:
                typeSampleSizeScoresMap[key] = (relativeScore, 1)
            else:
                newTuple = (currentTuple[0] + relativeScore, currentTuple[1] + 1)
                typeSampleSizeScoresMap[key] = newTuple
    
    sortingThing = []
    for k, v in typeSampleSizeScoresMap.items():
        avgScore = v[0] / v[1]
        parts = k.split(":")
        t = (parts[0], int(parts[1]), avgScore)
        sortingThing.append(t)
    
    sortingThing.sort(key=lambda tup: (tup[0], tup[1]))
    
    for s in sortingThing:
        print("Model average for combo: " + str(s))
