## Overview

This app allows users to visualize stock price data and detect local tops and bottoms within a specified window. 
Users can select different tickers, window lengths, and order for detecting tops and bottoms. The app also supports 
overlaying an Exponential Moving Average (EMA) and invalidating crossed lines for triple tops and bottoms. 
Additionally, the app provides a detailed walk-forward analysis of support and resistance levels, including breakout 
accuracy and various statistical measures.

## Methodology

### Local Tops and Bottoms

The app identifies local tops and bottoms in the stock price data using a rolling window approach. The parameters involved are:

- **Order**: Determines the number of surrounding data points to consider when identifying local tops and bottoms.
- **Tolerance**: A value between 0 and 1 that determines how close the detected tops and bottoms need to be to form horizontal lines (triple tops and bottoms).
- **Consecutive Only**: A boolean parameter that controls whether horizontal lines are drawn based on any three tops/bottoms or only three consecutive tops/bottoms.

### Parameters

- **Ticker**: The stock ticker symbol to analyze.
- **Window Length**: The number of days in the window of data to analyze. Options are 252 days (approximately one trading year) and 504 days (approximately two trading years).
- **Order**: The number of surrounding days to consider when identifying local tops and bottoms. Options range from 4 to 10.
- **Tolerance**: A value between 0 and 1 that determines the acceptable difference between the prices to consider them as forming a horizontal line.
- **Consecutive Only**: When set to True, horizontal lines are only drawn if three consecutive tops or bottoms form a horizontal line. When False, lines are drawn if any three tops or bottoms in the window form a horizontal line.
- **Log Price**: A boolean parameter that, when set to True, uses the logarithm of the closing prices instead of the raw closing prices for all calculations and plots.
- **Overlay EMA**: A boolean parameter that, when set to True, overlays an Exponential Moving Average (EMA) over the window of data. Users can specify the EMA span.
- **Invalidate Crossed Lines**: A boolean parameter that, when set to True, changes the color of the lines for triple tops and bottoms to grey if they are crossed by the price within the window, indicating they are no longer valid.
- **Breakout Threshold (days)**: The number of days to look ahead to determine if a breakout is true.
- **Breakout Threshold (percentage)**: The percentage change in price required to confirm a breakout.

### Statistics

The app calculates several statistics based on the walk-forward analysis:

- **Support Break Accuracy**: The ratio of true support breaks to total support breaks.
- **Resistance Break Accuracy**: The ratio of true resistance breaks to total resistance breaks.
- **Total Support Breaks**: The number of support breaks identified in the analysis.
- **Total Resistance Breaks**: The number of resistance breaks identified in the analysis.
- **Average Support Level**: The average value of identified support levels.
- **Average Resistance Level**: The average value of identified resistance levels.
- **Maximum Support Level**: The maximum value of identified support levels.
- **Minimum Support Level**: The minimum value of identified support levels.
- **Maximum Resistance Level**: The maximum value of identified resistance levels.
- **Minimum Resistance Level**: The minimum value of identified resistance levels.
- **False Support Break Rate**: The ratio of false support breaks to total support breaks.
- **False Resistance Break Rate**: The ratio of false resistance breaks to total resistance breaks.

## Usage

1. Select a ticker from the dropdown menu.
2. Choose the window length and order.
3. Enter a tolerance value between 0 and 1.
4. Set the Consecutive Only parameter as desired.
5. Set the Log Price parameter as desired.
6. Set the Overlay EMA parameter and specify the EMA span if needed.
7. Set the Invalidate Crossed Lines parameter as desired.
8. Use the navigation buttons to move through the data by day or month.
9. Adjust the start index using the slider if needed.
10. Enter the breakout threshold (days) and breakout threshold (percentage).
11. Click "Run Analysis" to perform the walk-forward analysis.

The plot will display the closing prices with detected tops and bottoms, along with any identified triple tops and 
bottoms. If the EMA overlay is enabled, it will also be displayed. If the Invalidate Crossed Lines parameter is enabled,
lines for triple tops and bottoms will turn grey if crossed within the window.

The app also provides an option to download the analysis results as a CSV file, which includes all detected tops, 
bottoms, support and resistance levels, and break information.
