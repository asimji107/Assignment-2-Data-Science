# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 20:42:43 2023

@author: 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Function to read data from Excel
def read_data(excel_url, sheet_name, new_cols, countries):
    """
   Read and process data from an Excel file.

   Parameters:
   excel_url (str): URL of the Excel file.
   sheet_name (str): Name of the sheet in the Excel file.
   new_cols (list): List of columns to keep.
   countries (list): List of countries for which data is required.

   Returns:
   tuple: A tuple containing the processed DataFrame and its transpose.
   """
    data = pd.read_excel(excel_url, sheet_name=sheet_name, skiprows=3)
    data = data[new_cols].set_index('Country Name').loc[countries]
    return data, data.T

# URLs for World Bank data
excel_urls = {
    'CO2': 'https://api.worldbank.org/v2/en/indicator/EN.ATM.CO2E.PC?downloadformat=excel',
    'forest': 'https://api.worldbank.org/v2/en/indicator/AG.LND.FRST.ZS?downloadformat=excel',
     'GDP': 'https://api.worldbank.org/v2/en/indicator/NY.GDP.MKTP.KD.ZG?downloadformat=excel',
   'urban' : 'https://api.worldbank.org/v2/en/indicator/SP.URB.GROW?downloadformat=excel' ,
   'electricity' : 'https://api.worldbank.org/v2/en/indicator/EG.ELC.FOSL.ZS?downloadformat=excel',
   'agriculture' : 'https://api.worldbank.org/v2/en/indicator/NV.AGR.TOTL.ZS?downloadformat=excel'
}

# Parameters for data extraction
sheet_name = 'Data'
new_cols = ['Country Name', '1997', '2000', '2003', '2006', '2009', '2012', '2015']
countries = ['Germany', 'United States', 'United Kingdom', 'Nigeria', 'China', 'Brazil', 'Australia']

# Reading and processing the data
data_frames = {}
for key, url in excel_urls.items():
    data, data_transpose = read_data(url, sheet_name, new_cols, countries)
    data_frames[key] = {'data': data, 'transpose': data_transpose}
# Print the data for urban growth
print("Urban Growth Data:\n", data_frames['urban']['data'])
print("Transposed Urban Growth Data:\n", data_frames['urban']['transpose'])

print("GDP Statistics:\n", data_frames['GDP']['transpose'].describe())
print("Forest Statistics:\n", data_frames['forest']['transpose'].describe())


# Function to plot time series data
def plot_time_series(data, title, xlabel, ylabel):
    """
   Plot time series data.

   Parameters:
   data (DataFrame): Data to be plotted.
   title (str): Title of the plot.
   xlabel (str): Label for the X-axis.
   ylabel (str): Label for the Y-axis.
   """
    plt.figure(figsize=(12, 6))
    for country in data.columns:
        plt.plot(data.index, data[country], label=country)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

# Plotting CO2 Emissions Over Time
plot_time_series(data_frames['CO2']['transpose'], "CO2 Emissions Over Time", "Year", "Emissions")

# Function to create a bar plot
def bar_plot(data, title, xlabel, ylabel):
    """
   Create a bar plot for the given data.

   Parameters:
   data (DataFrame): Data to be plotted.
   title (str): Title of the plot.
   xlabel (str): Label for the X-axis.
   ylabel (str): Label for the Y-axis.
   """
    data.plot(kind='bar', figsize=(12, 6))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# Bar plot for the year 2015 across all countries
bar_plot(data_frames['CO2']['data']['2015'], "CO2 Emissions in 2015", "Country", "Emissions")

# Function for correlation analysis
def correlation_analysis(data, title):
    """
 Perform correlation analysis and display a heatmap.

 Parameters:
 data (DataFrame): Data for which correlation is to be computed.
 title (str): Title of the heatmap.
 """
    corr_matrix = data.corr()
    plt.figure(figsize=(8, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title(title)
    plt.show()

# Correlation analysis for CO2 dataset
correlation_analysis(data_frames['CO2']['data'], "Correlation Matrix for CO2 Data")

def grouped_bar_plot(labels_array, width, y_data, y_label, labels, title):
    """
  Create a grouped bar plot for given data.

  Parameters:
  labels_array (list): List of labels for the X-axis.
  width (float): Width of each bar.
  y_data (list): List of data series to plot.
  y_label (str): Label for the Y-axis.
  labels (list): List of labels for each data series.
  title (str): Title of the plot.
  """
    x = np.arange(len(labels_array))
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, (year_data, year_label) in enumerate(zip(y_data, labels)):
        ax.bar(x + i * width, year_data, width, label=year_label)

    ax.set_title(title)
    ax.set_xlabel('Country')
    ax.set_ylabel(y_label)
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels_array)
    ax.legend()

    plt.show()

# Urban population growth plot
labels_array = ['Germany', 'USA', 'UK', 'Nigeria', 'China', 'Brazil', 'Australia']
width = 0.2
y_data_urban = [data_frames['urban']['data']['1997'], 
                data_frames['urban']['data']['2003'], 
                data_frames['urban']['data']['2009'], 
                data_frames['urban']['data']['2015']]
y_label_urban = 'Urban growth'
label_years_urban = ['Year 1997', 'Year 2003', 'Year 2009', 'Year 2015']
title_urban = 'Urban population growth (annual %)'

grouped_bar_plot(labels_array, width, y_data_urban, y_label_urban, label_years_urban, title_urban)

# Agriculture, forestry, and fishing plot
y_data_agri = [data_frames['agriculture']['data']['1997'], 
               data_frames['agriculture']['data']['2003'], 
               data_frames['agriculture']['data']['2009'], 
               data_frames['agriculture']['data']['2015']]
y_label_agri = '% of GDP'
label_years_agri = ['Year 1997', 'Year 2003', 'Year 2009', 'Year 2015']
title_agri = 'Agriculture, forestry, and fishing, value added (% of GDP)'

grouped_bar_plot(labels_array, width, y_data_agri, y_label_agri, label_years_agri, title_agri)

# Creating a DataFrame for China's data
data_China = {
    'Urban pop. growth': data_frames['urban']['transpose']['China'],
    'Electricity production': data_frames['electricity']['transpose']['China'],
    'Agric. forestry and Fisheries': data_frames['agriculture']['transpose']['China'],
    'CO2 Emissions': data_frames['CO2']['transpose']['China'],
    'Forest Area': data_frames['forest']['transpose']['China'],
    'GDP Annual Growth': data_frames['GDP']['transpose']['China']        
}

df_China = pd.DataFrame(data_China)
print(df_China)

def correlation_pvalues(data_x, data_y):
    """
Calculate Pearson correlation coefficients and p-values.

Parameters:
data_x (Series): Data series to correlate with others.
data_y (DataFrame): DataFrame containing multiple series for correlation.

Returns:
DataFrame: A DataFrame containing correlation coefficients and p-values.
"""
    corr_dataframe = pd.DataFrame(columns=['r', 'p'])
    for col in data_y:
        if pd.api.types.is_numeric_dtype(data_y[col]):
            r, p = stats.pearsonr(data_x, data_y[col])
            corr_dataframe.loc[col] = [round(r, 3), round(p, 3)]
    return corr_dataframe

# Function to create a heatmap for correlation data
def correlation_heatmap(data, corr, title):
    """
  Create a heatmap for the correlation matrix.

  Parameters:
  data (DataFrame): Data used for correlation.
  corr (DataFrame): Correlation matrix.
  title (str): Title of the heatmap.
  """
    plt.figure(figsize=(8, 8), dpi=200)
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(title, fontsize=20, fontweight='bold')
    plt.show()
data_x = df_China['Forest Area']
data_y = df_China
forest_area_corr = correlation_pvalues(data_x, data_y)
print("Correlation with Forest Area:\n", forest_area_corr)

# Creating and displaying the heatmap
corr_China = df_China.corr()
print("Correlation Matrix:\n", corr_China)
correlation_heatmap(df_China, corr_China, 'Correlation Heatmap for China')

def perform_regression_analysis(x_data, y_data):
    """
    Perform linear regression analysis.

    Parameters:
    x_data (DataFrame): Independent variable.
    y_data (DataFrame): Dependent variable.

    Returns:
    LinearRegression: A fitted regression model.
    """
    model = LinearRegression()
    model.fit(x_data, y_data)
    return model

 # Regression between GDP and CO2 emissions
gdp_data = data_frames['GDP']['data']['2015'].values.reshape(-1, 1)
co2_data = data_frames['CO2']['data']['2015'].values
regression_model = perform_regression_analysis(gdp_data, co2_data)

developed_countries = ['Germany', 'United States', 'United Kingdom']
developing_countries = ['Nigeria','Australia',  'Brazil']

co2_developed = data_frames['CO2']['data'].loc[developed_countries].mean()
co2_developing = data_frames['CO2']['data'].loc[developing_countries].mean()

plt.plot(co2_developed, label='Developed Countries')
plt.plot(co2_developing, label='Developing Countries')
plt.legend()
plt.title("Average CO2 Emissions Over Years")
plt.xlabel("Year")
plt.ylabel("CO2 Emissions")
plt.show()