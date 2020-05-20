"""
William Austin
Prakash Dhimal
George Mason University
CS 584 Theory and Applications of Data Mining
Semester project: Predicting the Impact of COVID-19
"""
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.ticker as plticker
import numpy as np

import Common as common
import Utilities as util


def getFullTickLabel(dayIndex):
    dayIndexInt = int(dayIndex)
    start = datetime(2020, 1, 22)
    offset = timedelta(days=dayIndex)
    tickDate = start + offset
    return tickDate.strftime("%b %d") + "  (" + str(dayIndexInt) + ")"


def updateTicks(x, pos):
    return getFullTickLabel(x)


def createHistoricalBetaChart(historicalRange, fullData, countryName):
    betaSmoothed15 = util.getGaussianAverage(fullData["beta"], 1.5)
    betaSmoothed4 = util.getGaussianAverage(fullData["beta"], 4)
    
    figure1, axis1 = plt.subplots()
    figure1.set_size_inches(7.5, 7.5)
    figure1.subplots_adjust(bottom=0.16)
    axis1.plot(historicalRange, fullData["beta"], label="Observed Beta", color="gray")
    # plt.plot(betaSmoothed3)
    # plt.plot(betaSmoothed7)
    axis1.plot(historicalRange, betaSmoothed15, label="Smoothed Beta, Sigma = 1.5", color="blue")
    axis1.plot(historicalRange, fullData["betaSmoothed"], label="Smoothed Beta, Sigma = 2.5", color="deepskyblue")
    axis1.plot(historicalRange, betaSmoothed4, label="Smoothed Beta, Sigma = 4", color="limegreen")
    # plt.plot(gaussianSmoothed2)
    axis1.set_xlabel("Day Number")
    axis1.set_ylabel("Transmission Rate")
    # axis1.xticks(np.arange(0, tsSize, 10))
    axis1.xaxis.set_major_locator(plticker.MultipleLocator(base=10))
    axis1.xaxis.set_tick_params(rotation=60)
    axis1.xaxis.set_major_formatter(mticker.FuncFormatter(updateTicks))
    axis1.legend()
    axis1.set_title("Historical Transmission Rate for " + countryName, fontsize="xx-large")
    # figure1.suptitle("Historical Transmission Rate for " + countryName)


def createPredictionChart(tsSize, futurePredictionDays, historicalRange, predictionRange, fullData, futurePredictions,
                          countryName):
    figure2, axis2 = plt.subplots()
    figure2.set_size_inches(7.5, 7.5)

    axis2.axvspan(tsSize - 1, tsSize + futurePredictionDays - 2, alpha=0.3, color='gray')
    # plt.plot(historicalRange, S)
    axis2.plot(historicalRange, fullData["I"], label="Infected (Confirmed Cases)", color="black", linestyle='solid')
    axis2.plot(historicalRange, fullData["R"], label="Recovered (Deaths + Recovered Cases)", color="black",
               linestyle='dotted')

    # plt.plot(predictionRange, SP1)
    axis2.plot(predictionRange, futurePredictions["sirConstant"][1], label="Infected Prediction (constant beta)",
               color="red", linestyle='solid')
    axis2.plot(predictionRange, futurePredictions["sirConstant"][2], label="Recovered Prediction (constant beta)",
               color="red", linestyle='dotted')

    axis2.plot(predictionRange, futurePredictions["sirDownward"][1], label="Infected Prediction (decreasing beta)",
               color="green", linestyle='solid')
    axis2.plot(predictionRange, futurePredictions["sirDownward"][2], label="Recovered Prediction (decreasing beta)",
               color="green", linestyle='dotted')

    axis2.plot(predictionRange, futurePredictions["sirContinueTrend"][1],
               label="Infected Prediction (continue beta trend)", color="blue", linestyle='solid')
    axis2.plot(predictionRange, futurePredictions["sirContinueTrend"][2],
               label="Recovered Prediction (continue beta trend)", color="blue", linestyle='dotted')

    axis2.set_xlabel("Day Number")
    axis2.set_ylabel("Number of Individuals")
    # axis2.xticks(np.arange(0, tsSize + predictionDays, 10))
    axis2.xaxis.set_major_locator(plticker.MultipleLocator(base=10))
    axis2.xaxis.set_tick_params(rotation=60)
    axis2.xaxis.set_major_formatter(mticker.FuncFormatter(updateTicks))
    axis2.legend()
    axis2.set_title("60 Day Predictions for " + countryName + " (Infected and Recovered)", fontsize="xx-large")
    # figure2.suptitle("60 Day Predictions for " + countryName + " (Infected and Recovered)")


def createPredictionsBetaChart(predictionRange, futurePredictions, countryName):
    figure3, axis3 = plt.subplots()
    figure3.set_size_inches(7.5, 7.5)
    axis3.plot(predictionRange, futurePredictions["betaConstant"], label="Constant beta", color="red")
    axis3.plot(predictionRange, futurePredictions["betaDownward"], label="Decreasing beta", color="green")
    axis3.plot(predictionRange, futurePredictions["betaContinueTrend"], label="Continue beta trend", color="blue")
    axis3.set_xlabel("Day Number")
    axis3.set_ylabel("Transmission Rate")
    # axis3.xticks(np.arange(tsSize - 1, tsSize + predictionDays, 5))
    axis3.xaxis.set_major_locator(plticker.MultipleLocator(base=5))
    axis3.xaxis.set_tick_params(rotation=60)
    axis3.xaxis.set_major_formatter(mticker.FuncFormatter(updateTicks))
    axis3.legend()
    axis3.set_title("Prediction Beta Values for " + countryName, fontsize="xx-large")
    # figure3.suptitle("Prediction Beta Values for " + countryName)


def createValidationChart(tsSize, fullData, cvPredictionDays, cvFullRange, cvTestRange, cvPredictions, countryName):
    figure4, axis4 = plt.subplots()
    figure4.set_size_inches(7.5, 7.5)
    # figure4.canvas.draw()

    axis4.axvspan(tsSize - cvPredictionDays - 1, tsSize, alpha=0.3, color='silver')
    # plt.plot(historicalRange, S)
    axis4.plot(cvFullRange, fullData["I"], label="Infected (Confirmed Cases)", color="black", linestyle='solid',
               linewidth=3.5)
    axis4.plot(cvFullRange, fullData["R"], label="Recovered (Deaths + Recovered Cases)", color="black",
               linestyle='dotted', linewidth=3.5)

    # plt.plot(predictionRange, SP1)
    axis4.plot(cvTestRange, cvPredictions["sirConstant"][1], label="Infected Prediction (constant beta)", color="red",
               linestyle='solid')
    axis4.plot(cvTestRange, cvPredictions["sirConstant"][2], label="Recovered Prediction (constant beta)", color="red",
               linestyle='dotted')

    axis4.plot(cvTestRange, cvPredictions["sirDownward"][1], label="Infected Prediction (decreasing beta)",
               color="green", linestyle='solid')
    axis4.plot(cvTestRange, cvPredictions["sirDownward"][2], label="Recovered Prediction (decreasing beta)",
               color="green", linestyle='dotted')

    axis4.plot(cvTestRange, cvPredictions["sirContinueTrend"][1], label="Infected Prediction (continue beta trend)",
               color="blue", linestyle='solid')
    axis4.plot(cvTestRange, cvPredictions["sirContinueTrend"][2], label="Recovered Prediction (continue beta trend)",
               color="blue", linestyle='dotted')

    axis4.set_xlabel("Day Number")
    axis4.set_ylabel("Number of Individuals")
    # axis2.xticks(np.arange(0, tsSize + predictionDays, 10))
    axis4.xaxis.set_major_locator(plticker.MultipleLocator(base=10))
    axis4.xaxis.set_tick_params(rotation=60)
    axis4.xaxis.set_major_formatter(mticker.FuncFormatter(updateTicks))
    # axis4.xaxis.
    # print("Tick Labels = " + str(axis4.get_xticklabels()))

    # axis_labels = [getFullTickLabel(int(q)) for q in axis4.get_xticks().tolist()]
    # print(axis_labels)
    # axis4.set_xlabels(axis_labels)
    # axis4.xticks(rotation=60)
    # $[getFullTickLabel(str(i)) for i in range(0, tsSize, 10)]
    # axis_labels = [item.get_text() for item in axis4.get_xticklabels()]
    # axis4.set_xticklabels(axis_labels)

    axis4.legend()
    axis4.set_title("Compare 30 Day Predictions vs. Actual for " + countryName + " (Infected and Recovered)",
                    fontsize="xx-large")
    # figure2.suptitle("60 Day Predictions for " + countryName + " (Infected and Recovered)")


def analyzeCountrySIR(tsData, countryName, chartSet, callShowPlot=True):
    tsSize = tsData.dateCount
    countryData = tsData.countryMap[countryName]
    countryPopulation = countryData.population
    futurePredictionDays = 60  # Arrays will be 31
    cvPredictionDays = 30

    fullData = util.getObservedModelValues(tsData, countryName)

    historicalRange = np.arange(tsSize)
    predictionRange = np.arange(tsSize - 1, tsSize + futurePredictionDays)

    futurePredictions = util.getStandardPredictions(tsData, countryName, 0, tsSize - 1, futurePredictionDays)
    cvPredictions = util.getStandardPredictions(tsData, countryName, 0, tsSize - cvPredictionDays - 1, cvPredictionDays)
    cvTrainRange = np.arange(0, tsSize - cvPredictionDays)
    cvTestRange = np.arange(tsSize - cvPredictionDays - 1, tsSize)
    cvFullRange = np.arange(0, tsSize)

    if "HistoricalBetaChart" in chartSet:
        createHistoricalBetaChart(historicalRange, fullData, countryName)

    if "PredictionChart" in chartSet:
        createPredictionChart(tsSize, futurePredictionDays, historicalRange, predictionRange, fullData,
                              futurePredictions, countryName)

    if "PredictionsBetaChart" in chartSet:
        createPredictionsBetaChart(predictionRange, futurePredictions, countryName)

    if "ValidationChart" in chartSet:
        createValidationChart(tsSize, fullData, cvPredictionDays, cvFullRange, cvTestRange, cvPredictions, countryName)

    if callShowPlot:
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    tsData = common.getTimeSeriesData()

    # for i in range(tsData.countryCount):
    #    countryName = tsData.countryIndex[i]
    #    countryData = tsData.countryMap[countryName]
    #    print("Country is: " + countryName + ", and first case was: " + str(tsData.dateIndex[countryData.firstIndex]))

    allCharts = {"HistoricalBetaChart", "PredictionChart", "PredictionsBetaChart", "ValidationChart"}

    for c in ["US", "Canada", "China", "Spain", "Germany", "France", "Italy", "Brazil", "Russia", "Nigeria", "Mexico"]:
        analyzeCountrySIR(tsData, c, {"HistoricalBetaChart"}, False)

    plt.show()

    # analyzeCountrySIR(tsData, "France")

    print("Done")
