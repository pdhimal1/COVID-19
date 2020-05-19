"""
William Austin
Prakash Dhimal
George Mason University
CS 584 Theory and Applications of Data Mining
Semester project: Predicting the Impact of COVID-19
"""


class CountryData(object):
    '''
    classdocs
    '''

    def __init__(self, name, population, confirmed, recovered, deaths, firstIndex):
        '''
        Constructor
        '''
        self.name = name
        self.population = population
        self.confirmed = confirmed
        self.recovered = recovered
        self.deaths = deaths
        self.firstIndex = firstIndex


class TimeSeriesData(object):
    '''
    classdocs
    '''

    def __init__(self, countryMap, countryIndex, dateIndex):
        '''
        Constructor
        '''
        self.countryMap = countryMap
        self.countryIndex = countryIndex
        self.dateIndex = dateIndex
        self.countryCount = len(countryIndex)
        self.dateCount = len(dateIndex)
