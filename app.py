import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Function to load data
@st.cache_data
def load_data(ticker):
    file_path = f'data/{ticker.lower()}.csv'
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    return df


# Function to check for local tops
def rw_top(data, curr_index, order):
    if curr_index < order * 2 + 1:
        return False

    top = True
    k = curr_index - order
    v = data[k]
    for i in range(1, order + 1):
        if data[k + i] > v or data[k - i] > v:
            top = False
            break

    return top


# Function to check for local bottoms
def rw_bottom(data, curr_index, order):
    if curr_index < order * 2 + 1:
        return False

    bottom = True
    k = curr_index - order
    v = data[k]
    for i in range(1, order + 1):
        if data[k + i] < v or data[k - i] < v:
            bottom = False
            break

    return bottom


# Function to detect tops and bottoms
def rw_extremes(data, order):
    tops = []
    bottoms = []
    for i in range(len(data)):
        if rw_top(data, i, order):
            top = [i, i - order, data[i - order]]
            tops.append(top)

        if rw_bottom(data, i, order):
            bottom = [i, i - order, data[i - order]]
            bottoms.append(bottom)

    return tops, bottoms


# Function to detect horizontal lines
def detect_horizontal_lines(points, tolerance=0.005):
    horizontal_lines = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            for k in range(j + 1, len(points)):
                if (abs(points[i][2] - points[j][2]) / points[i][2] < tolerance and
                        abs(points[i][2] - points[k][2]) / points[i][2] < tolerance):
                    horizontal_lines.append(points[i][2])
    return horizontal_lines


# Function to plot the window
def plot_window(df, start, order, window_length, tolerance):
    plt.figure(figsize=(10, 5))
    window = df.iloc[start:start + window_length]
    data = window["close"].to_numpy()
    idx = window.index

    # Detect tops and bottoms
    tops, bottoms = rw_extremes(data, order)

    # Detect horizontal lines for triple tops and bottoms
    triple_tops = detect_horizontal_lines(tops, tolerance)
    triple_bottoms = detect_horizontal_lines(bottoms, tolerance)

    # Plot the closing prices
    plt.plot(idx, data, label="Close")

    # Plot detected tops and bottoms
    for top in tops:
        plt.plot(idx[top[1]], top[2], marker="o", color="green")

    for bottom in bottoms:
        plt.plot(idx[bottom[1]], bottom[2], marker="o", color="red")

    # Plot horizontal lines for triple tops and bottoms
    for top in triple_tops:
        plt.axhline(y=top, color="g", linestyle="--", label="Triple Top")

    for bottom in triple_bottoms:
        plt.axhline(y=bottom, color="r", linestyle="--", label="Triple Bottom")

    plt.title(f"{window_length}-Day Window of Closing Prices")
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.grid(True)
    st.pyplot(plt)


# Streamlit app
st.title("Stock Price Analysis")

# Dropdown to select ticker
ticker = st.selectbox("Select Ticker", [
    "URE", "DBA", "VNQ", "EWY", "UNG", "USO", "THD", "IBB", "SPY", "EWJ",
    "XLI", "XLP", "TUR", "GDX", "LMZSDS03", "XAUUSD", "ARKK", "VWO",
    "LMAHDS03", "EWO", "JJC", "EWW", "EWQ", "XW1", "EWC", "IWM", "DBC",
    "XLF", "SOXX", "EWA", "XLE", "EZA", "XLK", "CPER", "XPDUSD",
    "USGG10YR", "BHP_AU", "GSG", "BHP_LN", "EWZ", "EWL", "EWT", "EWM",
    "SCO1", "GLD", "EWP", "MXEF", "XRT", "700_HK", "EEM_US", "EWK",
    "DBB", "XPTUSD", "XLV", "FXI", "XLB", "IDXX", "QQQ", "EWN", "EWU",
    "XME", "VGK", "XO1", "SLV", "HYG", "HG1", "EWD", "XLU", "XLY",
    "AAPL_US", "EWS", "CO1", "EWG", "EWI"
])

# Load data
df = load_data(ticker)

# User input for window length and order
window_length = st.selectbox("Select Window Length", [252, 504])
order = st.selectbox("Select Order", [1, 2, 3, 4, 5])

# Initialize session state
if "start_index" not in st.session_state:
    st.session_state.start_index = 0

# Input for tolerance value
tolerance_input = st.text_input("Tolerance (0 to 1)", "0.005")

# Validate tolerance input
try:
    tolerance = float(tolerance_input.replace(",", "."))
    if tolerance < 0 or tolerance > 1:
        st.error("Tolerance must be a number between 0 and 1.")
    else:
        # Plot the window
        plot_window(df, st.session_state.start_index, order, window_length, tolerance)
except ValueError:
    st.error("Invalid input. Please enter a number between 0 and 1.")


# Function to update the plot by days
def update_plot_by_days(change):
    st.session_state.start_index += change
    if st.session_state.start_index < 0:
        st.session_state.start_index = 0
    elif st.session_state.start_index > len(df) - window_length:
        st.session_state.start_index = len(df) - window_length


# Function to update the plot by months (30 days)
def update_plot_by_months(change):
    st.session_state.start_index += change * 30
    if st.session_state.start_index < 0:
        st.session_state.start_index = 0
    elif st.session_state.start_index > len(df) - window_length:
        st.session_state.start_index = len(df) - window_length


# Buttons to navigate by days
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("Previous Day"):
        update_plot_by_days(-1)
with col2:
    if st.button("Next Day"):
        update_plot_by_days(1)

# Buttons to navigate by months
with col3:
    if st.button("Previous Month"):
        update_plot_by_months(-1)
with col4:
    if st.button("Next Month"):
        update_plot_by_months(1)

# Slider to set start index
start_index = st.slider("Start Index", 0, len(df) - window_length, st.session_state.start_index)
st.session_state.start_index = start_index
