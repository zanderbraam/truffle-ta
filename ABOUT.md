## Overview

This app allows users to visualize stock price data and detect local tops and bottoms within a specified window. 
Users can select different tickers, window lengths, and order for detecting tops and bottoms.

## Methodology

### Local Tops and Bottoms

The app identifies local tops and bottoms in the stock price data using a rolling window approach. The parameters involved are:

- **Order**: Determines the number of surrounding data points to consider when identifying local tops and bottoms.
- **Tolerance**: A value between 0 and 1 that determines how close the detected tops and bottoms need to be to form horizontal lines (triple tops and bottoms).
- **Consecutive Only**: A boolean parameter that controls whether horizontal lines are drawn based on any three tops/bottoms or only three consecutive tops/bottoms.

### Parameters

- **Ticker**: The stock ticker symbol to analyze.
- **Window Length**: The number of days in the window of data to analyze. Options are 252 days (approximately one trading year) and 504 days (approximately two trading years).
- **Order**: The number of surrounding days to consider when identifying local tops and bottoms. Options range from 1 to 7.
- **Tolerance**: A value between 0 and 1 that determines the acceptable difference between the prices to consider them as forming a horizontal line.
- **Consecutive Only**: When set to True, horizontal lines are only drawn if three consecutive tops or bottoms form a horizontal line. When False, lines are drawn if any three tops or bottoms in the window form a horizontal line.
- **Log Price**: A boolean parameter that, when set to True, uses the logarithm of the closing prices instead of the raw closing prices for all calculations and plots.

## Usage

1. Select a ticker from the dropdown menu.
2. Choose the window length and order.
3. Enter a tolerance value between 0 and 1.
4. Set the Consecutive Only parameter as desired.
5. Set the Log Price parameter as desired.
6. Use the navigation buttons to move through the data by day or month.
7. Adjust the start index using the slider if needed.

The plot will display the closing prices with detected tops and bottoms, along with any identified triple tops and bottoms.
