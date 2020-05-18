'''
Created on May 16, 2020

@author: William
'''
from os.path import dirname, realpath, join

import pandas as pd

from data_objects import CountryData
from data_objects import TimeSeriesData

BASE_URL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"


def getPopulationDataframe():
    rootDirectory = getProjectRootDirectory()
    populationFilePath = join(rootDirectory, "resources", "population_by_country_2020.csv")
    return pd.read_csv(populationFilePath)


def getProjectRootDirectory():
    dirPath = dirname(realpath(__file__))
    projectDirPath = dirname(dirPath)
    return projectDirPath


def getSummarizedDataframe(sourceUrl):
    rawData = pd.read_csv(sourceUrl)
    simplified = rawData.drop(["Province/State", "Lat", "Long"], axis=1, inplace=False)
    summarized = simplified.groupby(["Country/Region"]).sum()
    return summarized


def getSummarizedDataframes():
    confirmed = getSummarizedDataframe(BASE_URL + "time_series_covid19_confirmed_global.csv")
    recovered = getSummarizedDataframe(BASE_URL + "time_series_covid19_recovered_global.csv")
    deaths = getSummarizedDataframe(BASE_URL + "time_series_covid19_deaths_global.csv")
    return confirmed, recovered, deaths


def getTimeSeriesArray(dataframe, countryName):
    countryData = dataframe.loc[countryName]
    return countryData.values.flatten()


def getTimeSeriesData():
    populationData = getPopulationDataframe()

    countryMap = {}
    confirmedDF, recoveredDF, deathsDF = getSummarizedDataframes()

    countryIndex = confirmedDF.index
    dateIndex = confirmedDF.columns

    for country in countryIndex:
        populationDataSeries = populationData.loc[populationData["Country/Region"] == country]
        population = populationDataSeries.iloc[0]["Population"]

        countryConfirmed = getTimeSeriesArray(confirmedDF, country)
        countryRecovered = getTimeSeriesArray(recoveredDF, country)
        countryDeaths = getTimeSeriesArray(deathsDF, country)

        foundFirstIndex = False
        firstIndex = -1
        i = 0
        while not foundFirstIndex and i < len(dateIndex):
            if countryConfirmed[i] > 0 or countryRecovered[i] > 0 or countryDeaths[i] > 0:
                firstIndex = i
                foundFirstIndex = True
            else:
                i += 1

        countryData = CountryData(country, population, countryConfirmed, countryRecovered, countryDeaths, firstIndex)
        countryMap[country] = countryData

    timeSeriesData = TimeSeriesData(countryMap, countryIndex, dateIndex)
    return timeSeriesData
