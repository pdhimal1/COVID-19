```
William Austin
Prakash Dhimal
George Mason University
CS 584 Theory and Applications of Data Mining
Semester project: Predicting the Impact of COVID-19
```

# Predicting the Impact of COVID-19

#### Why Investigate COVID-19?
 * COVID-19 is a now a major global pandemic and is having a major impact on people’s lives around the globe.
 * There is significant uncertainty about what the effects the pandemic will be going forward.
 * Large amounts of data have been collected as the spread of Covid-19 has progressed. Using data mining techniques to attempt to predict likely outcomes will support the task of building an appropriate and effective response strategy.

#### Primary Project Goal:
For this project, we will construct a model showing how cases of COVID-19 will spread around the globe. Our main focus will be on accurately predicting the number of infections, fatalities, and recoveries we should expect, along with what the time frame is for these events to unfold.

#### COVID-19 Data Sources
##### Primary Data Source:
Our primary data source for mapping the global spread of COVID-19 will be the data provided by the Johns Hopkins Center for Systems Science and Engineering (CSSE). This contains day-by-day time series data, broken down by country and region.
 * Source: https://github.com/CSSEGISandData/COVID-19
 * Kaggle Page: https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset
 * Visualization: https://coronavirus.jhu.edu/map.html

#### Project source code:
The source code for this project is located in the `src` directory. Jupyter notebooks are in the `src/jupyter-notebook directory`.

#### Project report
Please refer to: `report/report.pdf` for the project report

#### Dependency:
This python program depends on the following modules:
  * datetime
  * matplotlib
  * numpy
  * os
  * pandas
  * scipy
  * sklearn
  * scipy

####
To run the SIR model and generate COVID-19 forecast plots, run:
  * `src/AnalyzeTransmissionRates.py`
For validation work, run:
  * `src/PredictFutureBetaValues.py`

Contributors:<br/>
Prakash Dhimal<br/>
William Austin<br/>
George Mason University
