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


# Function to read ABOUT.md file
def read_about():
    with open('ABOUT.md', 'r') as file:
        return file.read()


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
def detect_horizontal_lines(points, tolerance=0.005, consecutive_only=False):
    horizontal_lines = []
    if consecutive_only:
        for i in range(len(points) - 2):
            if (abs(points[i][2] - points[i + 1][2]) / points[i][2] < tolerance and
                    abs(points[i][2] - points[i + 2][2]) / points[i][2] < tolerance):
                horizontal_lines.append(points[i][2])
    else:
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                for k in range(j + 1, len(points)):
                    if (abs(points[i][2] - points[j][2]) / points[i][2] < tolerance and
                            abs(points[i][2] - points[k][2]) / points[i][2] < tolerance):
                        horizontal_lines.append(points[i][2])
    return horizontal_lines


# Function to check if a line is crossed after a certain index
def is_line_crossed(data, line, start_index):
    return any(data[start_index:] > line) or any(data[start_index:] < line)

# Function to plot the window
def plot_window(df, start, order, window_length, tolerance, consecutive_only, log_price, ema_toggle, ema_span, invalidate_lines):
    fig, ax = plt.subplots(figsize=(10, 5))
    window = df.iloc[start:start + window_length]

    if log_price:
        data = np.log(window["close"].to_numpy())
    else:
        data = window["close"].to_numpy()

    idx = window.index

    # Detect tops and bottoms
    tops, bottoms = rw_extremes(data, order)

    # Detect horizontal lines for triple tops and bottoms
    triple_tops = detect_horizontal_lines(tops, tolerance, consecutive_only)
    triple_bottoms = detect_horizontal_lines(bottoms, tolerance, consecutive_only)

    # Plot the closing prices
    ax.plot(idx, data, label="Close (Log)" if log_price else "Close")

    # Plot detected tops and bottoms
    for top in tops:
        ax.plot(idx[top[1]], top[2], marker="o", color="green")

    for bottom in bottoms:
        ax.plot(idx[bottom[1]], bottom[2], marker="o", color="red")

    # Plot horizontal lines for triple tops and bottoms
    for top in triple_tops:
        last_top_index = max([t[1] for t in tops if t[2] == top])
        if invalidate_lines and any(data[last_top_index:] > top):
            line_color = "grey"
        else:
            line_color = "green"
        ax.axhline(y=top, color=line_color, linestyle="--", label="Triple Top")

    for bottom in triple_bottoms:
        last_bottom_index = max([b[1] for b in bottoms if b[2] == bottom])
        if invalidate_lines and any(data[last_bottom_index:] < bottom):
            line_color = "grey"
        else:
            line_color = "red"
        ax.axhline(y=bottom, color=line_color, linestyle="--", label="Triple Bottom")

    # Overlay EMA if toggle is enabled
    if ema_toggle:
        ema = window['close'].ewm(span=ema_span).mean()
        if log_price:
            ema = np.log(ema)
        ax.plot(idx, ema, label=f"EMA (span={ema_span})", linestyle='--')

    ax.set_title(f"{window_length}-Day Window of {'Log ' if log_price else ''}Closing Prices")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log Closing Price" if log_price else "Closing Price")
    ax.grid(True)
    st.pyplot(fig, clear_figure=True)


# Streamlit app
st.title("Stock Price Technical Analysis App")

# Display ABOUT.md content
about_content = read_about()
st.markdown(about_content)

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
order = st.selectbox("Select Order", [1, 2, 3, 4, 5, 6, 7])

# Initialize session state
if "start_index" not in st.session_state:
    st.session_state.start_index = 0

# Input for tolerance value
tolerance_input = st.text_input("Tolerance (0 to 1)", "0.005")

# Input for consecutive only parameter
consecutive_only = st.checkbox("Consecutive Only", value=False)

# Input for log price parameter
log_price = st.checkbox("Use Log Price", value=False)

# Toggle for EMA
ema_toggle = st.checkbox("Overlay EMA", value=False)
ema_span = None

if ema_toggle:
    ema_span_input = st.text_input("EMA Span (positive integer)", "20")
    try:
        ema_span = int(ema_span_input)
        if ema_span <= 0:
            st.error("EMA span must be a positive integer.")
            ema_span = None
    except ValueError:
        st.error("Invalid input. Please enter a positive integer.")

# Toggle for invalidating crossed lines
invalidate_lines = st.checkbox("Invalidate Crossed Lines", value=False)

# Validate tolerance input
try:
    tolerance = float(tolerance_input.replace(",", "."))
    if tolerance < 0 or tolerance > 1:
        st.error("Tolerance must be a number between 0 and 1.")
    else:
        # Plot the window
        plot_window(df, st.session_state.start_index, order, window_length, tolerance, consecutive_only, log_price,
                    ema_toggle, ema_span, invalidate_lines)
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
