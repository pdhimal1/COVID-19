"""
William Austin
Prakash Dhimal
George Mason University
CS 584 Theory and Applications of Data Mining
Semester project: Predicting the Impact of COVID-19
"""

import pandas as pd


def get_raw_data():
    confirmed_df = pd.read_csv(
        'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    recovered_df = pd.read_csv(
        'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
    death_df = pd.read_csv(
        'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
    return confirmed_df, recovered_df, death_df


def get_country_data_running_total(data, columnName, countryName):
    country_data = data.loc[data['Country/Region'] == countryName]
    country_time_series_only = country_data.drop(["Province/State", "Country/Region", "Lat", "Long"], axis=1)
    transposed = country_time_series_only.transpose()
    transposed = pd.DataFrame({
        columnName: transposed.sum(axis=1)
    })
    return transposed


'''
This dataset doesnot have daily new case number
'''


def get_country_data_daily_number(data, columnName, countryName):
    country_data = data.loc[data['Country/Region'] == countryName]
    country_time_series_only = country_data.drop(["Province/State", "Country/Region", "Lat", "Long"], axis=1)
    transposed = country_time_series_only.transpose()
    transposed.columns = [columnName]
    return transposed


def get_country_confirmed_recovered_death_running_total_data(confirmed_df, recovered_df, death_df, countryName):
    country_confirmed = get_country_data_running_total(confirmed_df, "Confirmed", countryName)
    country_recovered = get_country_data_running_total(recovered_df, "Recovered", countryName)
    country_death = get_country_data_running_total(death_df, "Death", countryName)
    country_all = country_confirmed
    country_all["Recovered"] = country_recovered.Recovered
    country_all["Death"] = country_death.Death
    return country_all


def get_country_confirmed_recovered_death_daily_data(confirmed_df, recovered_df, death_df, countryName):
    country_confirmed = get_country_data_daily_number(confirmed_df, "Confirmed", countryName)
    country_recovered = get_country_data_daily_number(recovered_df, "Recovered", countryName)
    country_death = get_country_data_daily_number(death_df, "Death", countryName)
    country_all = country_confirmed
    country_all["Recovered"] = country_recovered.Recovered
    country_all["Death"] = country_death.Death
    return country_all


def get_US_daily(confirmed_df, recovered_df, death_df):
    us_data = get_country_confirmed_recovered_death_daily_data(
        confirmed_df,
        recovered_df,
        death_df,
        "US")
    return us_data


def get_US_running_total(confirmed_df, recovered_df, death_df):
    us_data = get_country_confirmed_recovered_death_running_total_data(
        confirmed_df,
        recovered_df,
        death_df,
        "US")
    return us_data


def main():
    confirmed_df, recovered_df, death_df = get_raw_data()
    countries = list()
    data = {}
    countries.append("US")
    countries.append("Italy")
    countries.append("China")
    countries.append("France")
    countries.append("Spain")
    combined_data = pd.DataFrame()
    for country in countries:
        country_data = get_country_confirmed_recovered_death_running_total_data(confirmed_df, recovered_df, death_df,
                                                                                country)
        data[country] = combined_data
        combined_data[country + "Confirmed"] = country_data.Confirmed
        combined_data[country + "Recovered"] = country_data.Recovered
        combined_data[country + "Death"] = country_data.Death
    print(data)
    print(combined_data)


if __name__ == '__main__':
    main()
