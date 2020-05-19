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

countryMinimumScore = {}
modelScoreMap = {}

def updateMapping(countryName, modelDescriptor, scoreValue):
    minimumScore = countryMinimumScore.get(countryName)
    
    if minimumScore is None or scoreValue < minimumScore:
        countryMinimumScore[countryName] = scoreValue
    
    countryScoreMap = modelScoreMap.get(modelDescriptor)
    if countryScoreMap is None:
        newCountryScoreMap = {}
        newCountryScoreMap[countryName] = scoreValue
        modelScoreMap[modelDescriptor] = newCountryScoreMap
    else:
        countryScoreMap[countryName] = scoreValue

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
                            updateMapping(countryName, modelDescriptor, relativeSSE)
                        
                        if sampleSize >= 3:
                            betaPredictions = bpm.getBetaPredictionByQuadraticRegression(smoothedBetaObserved[sampleRange], sampleRange, predictionRange)
                            #betaPredictions = bpm.getBetaPredictionByLinearRegression(smoothedBetaObserved[sampleRange], sampleRange, predictionRange)
                            betaPredictions = util.boundAtZero(betaPredictions)
                            predictionsMap = util.getPredictionsSIR2(betaPredictions, gammaArray, sInit, iInit, rInit, countryPopulation)
                            relativeSSE = util.computeMeanSquareError(sTrueValues, predictionsMap["S"])
                            modelDescriptor = "Type=QuadraticRegression, SampleSize=" + str(sampleSize) + ", SmoothingValue=" + str(smoothingValue)
                            updateMapping(countryName, modelDescriptor, relativeSSE)
                        
                            lastBeta = smoothedBetaObserved[observedDaysCount-1]
                            lastBetaSlope = smoothedBetaObserved[observedDaysCount-1] - smoothedBetaObserved[observedDaysCount-2]
                            memoryValues = [0.0, 0.001, 0.05, 0.01, 0.05, 0.1, 0.15]
                            
                            for mv in memoryValues:
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
        
        print("Completed models for country: " + countryName)
    
    print("\n===================")
    print("Summarizing Results")
    print("===================")
    
    for key, value in countryMinimumScore.items():
        print("Minimum Score for " + key + " is: " + str(value))
    
    finalModelScores = []
    for modelKey, scoreMap in modelScoreMap.items():
        normalizedModelScore = 0.0
        countryCount = 0
        for countryName, countryScore in scoreMap.items():
            minimumCountryScore = countryMinimumScore.get(countryName)
            if minimumCountryScore is not None and minimumCountryScore > 0.0:
                relativeScore = countryScore / minimumCountryScore
                normalizedModelScore += relativeScore
                countryCount += 1
        
        modelScore = normalizedModelScore / countryCount
        modelScoreTuple = (modelScore, modelKey, countryCount)
        finalModelScores.append(modelScoreTuple)
    
    finalModelScores.sort(key=lambda x: x[0])
    
    print("\n================")
    print("Printing top models.")
    print("====================")
    print("Total model count = " + str(len(finalModelScores)))
    
    for i in range(min(len(finalModelScores), 400)):
        modelDetails = finalModelScores[i]
        print(str(i + 1) + ". Score = " + str(modelDetails[0]) + ", Country Count = " + str(modelDetails[2]) + ", Model Key = " + modelDetails[1])
        
    #print("Country Best for " + countryName + " was SSE = " + str(countryBestSSE) + ", and method is: " + countryBestMethod)