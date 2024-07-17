import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Function to load data
@st.cache_data
def load_data(ticker):
    file_path = f'data/{ticker.lower()}.csv'
    df = pd.read_csv(file_path)
    df.set_index('Date', inplace=True)
    df.index = pd.to_datetime(df.index)
    return df


# Initialize testing_df with necessary columns
def initialize_testing_df(df):
    testing_df = df[['close']].copy()
    testing_df['tops'] = pd.Series(dtype='object')
    testing_df['bottoms'] = pd.Series(dtype='object')
    testing_df['support_levels'] = pd.Series(dtype='object')
    testing_df['resistance_levels'] = pd.Series(dtype='object')
    testing_df['support_breaks'] = False
    testing_df['resistance_breaks'] = False
    testing_df['true_support_breaks'] = False
    testing_df['false_support_breaks'] = False
    testing_df['true_resistance_breaks'] = False
    testing_df['false_resistance_breaks'] = False
    return testing_df


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
                horizontal_lines.append(points[i + 2][2])
    else:
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                for k in range(j + 1, len(points)):
                    if (abs(points[i][2] - points[j][2]) / points[i][2] < tolerance and
                            abs(points[i][2] - points[k][2]) / points[i][2] < tolerance):
                        horizontal_lines.append(points[k][2])
    return horizontal_lines


def is_true_breakout(prices, level, index, threshold_days=5, threshold_percentage=0.01, is_top=True):
    future_index = index + threshold_days
    if future_index >= len(prices):
        return False

    future_price = prices.iloc[future_index]
    if is_top:
        return future_price > level * (1 + threshold_percentage)
    else:
        return future_price < level * (1 - threshold_percentage)


def walk_forward_test(prices, window_size=252, order=5, threshold_days=5, threshold_percentage=0.01,
                      tolerance=0.005, consecutive_only=False, log_price=False, invalidate_lines=False):

    close_prices = prices['close'].to_numpy()
    if log_price:
        close_prices = np.log(close_prices)
        testing_df = initialize_testing_df(np.log(prices))
    else:
        testing_df = initialize_testing_df(prices)

    # Keep track of invalidated levels
    invalidated_top_levels = set()
    invalidated_bottom_levels = set()

    for i in range(len(prices) - window_size):
        window_data = close_prices[i:i + window_size]
        tops, bottoms = rw_extremes(window_data, order)

        # Extract only the float values for tops and bottoms and convert to Python float
        tops_values = [float(top[2]) for top in tops]
        bottoms_values = [float(bottom[2]) for bottom in bottoms]

        top_levels = detect_horizontal_lines(tops, tolerance, consecutive_only)
        bottom_levels = detect_horizontal_lines(bottoms, tolerance, consecutive_only)

        top_level_values = [float(top) for top in top_levels]
        bottom_level_values = [float(bottom) for bottom in bottom_levels]

        # Update testing_df with tops and bottoms
        testing_df.at[prices.index[i + window_size - 1], 'tops'] = tops_values
        testing_df.at[prices.index[i + window_size - 1], 'bottoms'] = bottoms_values

        # Update support and resistance levels
        testing_df.at[prices.index[i + window_size - 1], 'support_levels'] = bottom_level_values
        testing_df.at[prices.index[i + window_size - 1], 'resistance_levels'] = top_level_values

        # Check for breaks and update columns
        current_price = close_prices[i + window_size - 1]
        for level in top_levels:
            if current_price > level and level not in invalidated_top_levels:
                is_true = is_true_breakout(prices['close'], level, i + window_size - 1, threshold_days,
                                           threshold_percentage, is_top=True)
                testing_df.at[prices.index[i + window_size - 1], 'resistance_breaks'] = True
                if is_true:
                    testing_df.at[prices.index[i + window_size - 1], 'true_resistance_breaks'] = True
                else:
                    testing_df.at[prices.index[i + window_size - 1], 'false_resistance_breaks'] = True
                if invalidate_lines:
                    invalidated_top_levels.add(level)

        for level in bottom_levels:
            if current_price < level and level not in invalidated_bottom_levels:
                is_true = is_true_breakout(prices['close'], level, i + window_size - 1, threshold_days,
                                           threshold_percentage, is_top=False)
                testing_df.at[prices.index[i + window_size - 1], 'support_breaks'] = True
                if is_true:
                    testing_df.at[prices.index[i + window_size - 1], 'true_support_breaks'] = True
                else:
                    testing_df.at[prices.index[i + window_size - 1], 'false_support_breaks'] = True
                if invalidate_lines:
                    invalidated_bottom_levels.add(level)

    return testing_df


# Function to check if a line is crossed after a certain index
def is_line_crossed_after_point(data, line, start_index, check_greater=True):
    if check_greater:
        cross = any(data[start_index + 1:] > line)
    else:
        cross = any(data[start_index + 1:] < line)

    return cross


def calculate_statistics(testing_df):
    total_support_breaks = testing_df['support_breaks'].sum()
    true_support_breaks = testing_df['true_support_breaks'].sum()
    false_support_breaks = testing_df['false_support_breaks'].sum()
    total_resistance_breaks = testing_df['resistance_breaks'].sum()
    true_resistance_breaks = testing_df['true_resistance_breaks'].sum()
    false_resistance_breaks = testing_df['false_resistance_breaks'].sum()

    support_accuracy = true_support_breaks / total_support_breaks if total_support_breaks > 0 else 0
    resistance_accuracy = true_resistance_breaks / total_resistance_breaks if total_resistance_breaks > 0 else 0

    total_tops = testing_df['tops'].dropna().apply(len).sum()
    total_bottoms = testing_df['bottoms'].dropna().apply(len).sum()

    avg_support_levels = testing_df['support_levels'].dropna().apply(lambda x: np.mean(x) if x else np.nan).mean()
    avg_resistance_levels = testing_df['resistance_levels'].dropna().apply(lambda x: np.mean(x) if x else np.nan).mean()

    max_support_levels = testing_df['support_levels'].dropna().apply(lambda x: np.max(x) if x else np.nan).max()
    min_support_levels = testing_df['support_levels'].dropna().apply(lambda x: np.min(x) if x else np.nan).min()
    max_resistance_levels = testing_df['resistance_levels'].dropna().apply(lambda x: np.max(x) if x else np.nan).max()
    min_resistance_levels = testing_df['resistance_levels'].dropna().apply(lambda x: np.min(x) if x else np.nan).min()

    false_support_break_rate = false_support_breaks / total_support_breaks if total_support_breaks > 0 else 0
    false_resistance_break_rate = false_resistance_breaks / total_resistance_breaks if total_resistance_breaks > 0 else 0

    return {
        "total_support_breaks": total_support_breaks,
        "true_support_breaks": true_support_breaks,
        "false_support_breaks": false_support_breaks,
        "support_accuracy": support_accuracy,
        "total_resistance_breaks": total_resistance_breaks,
        "true_resistance_breaks": true_resistance_breaks,
        "false_resistance_breaks": false_resistance_breaks,
        "resistance_accuracy": resistance_accuracy,
        "total_tops": total_tops,
        "total_bottoms": total_bottoms,
        "avg_support_levels": avg_support_levels,
        "avg_resistance_levels": avg_resistance_levels,
        "max_support_levels": max_support_levels,
        "min_support_levels": min_support_levels,
        "max_resistance_levels": max_resistance_levels,
        "min_resistance_levels": min_resistance_levels,
        "false_support_break_rate": false_support_break_rate,
        "false_resistance_break_rate": false_resistance_break_rate
    }


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
        last_top_index = max([t[0] for t in tops if t[2] == top])
        if invalidate_lines and is_line_crossed_after_point(data, top, last_top_index, check_greater=True):
            line_color = "grey"
        else:
            line_color = "green"
        ax.axhline(y=top, color=line_color, linestyle="--", label="Triple Top")

    for bottom in triple_bottoms:
        last_bottom_index = max([b[0] for b in bottoms if b[2] == bottom])
        if invalidate_lines and is_line_crossed_after_point(data, bottom, last_bottom_index, check_greater=False):
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
wf_df = df.copy()

# User input for window length and order
window_length = st.selectbox("Select Window Length", [252, 504])
order = st.selectbox("Select Order", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

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

# Convert pandas.Timestamp to datetime
min_date = df.index.min().to_pydatetime()
max_date = df.index.max().to_pydatetime()
current_end_date = df.index[min(len(df) - 1, st.session_state.start_index + window_length - 1)].to_pydatetime()

# Slider to set start index based on the end date of the window
start_date = st.slider("Start Date", min_date, max_date, current_end_date)
st.session_state.start_index = max(0, df.index.get_loc(pd.to_datetime(start_date)) - window_length + 1)

# Input for breakout threshold
threshold_days = st.number_input("Breakout Threshold (days)", min_value=1, value=5)
threshold_percentage = st.number_input("Breakout Threshold (percentage)", min_value=0.001, max_value=1.0, value=0.01,
                                       step=0.001, format="%.3f")

# Ensure that threshold_days and threshold_percentage changes are captured
if "threshold_days" not in st.session_state:
    st.session_state.threshold_days = threshold_days
if "threshold_percentage" not in st.session_state:
    st.session_state.threshold_percentage = threshold_percentage

threshold_days = st.session_state.threshold_days
threshold_percentage = st.session_state.threshold_percentage

# Button to run analysis
if st.button("Run Analysis"):
    if not invalidate_lines:
        st.error("Please check 'Invalidate Crossed Lines' before running the analysis.")
    else:
        try:
            tolerance = float(tolerance_input.replace(",", "."))
            if tolerance < 0 or tolerance > 1:
                st.error("Tolerance must be a number between 0 and 1.")
            else:
                testing_df = walk_forward_test(wf_df, window_size=window_length, order=order,
                                               threshold_days=threshold_days,
                                               threshold_percentage=threshold_percentage, tolerance=tolerance,
                                               consecutive_only=consecutive_only, log_price=log_price,
                                               invalidate_lines=invalidate_lines)

                # Calculate statistics
                stats = calculate_statistics(testing_df)

                # Store statistics in session state
                st.session_state.stats = stats

                # Option to download testing_df
                csv = testing_df.to_csv(index=True)
                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name=f'{ticker.lower()}_results.csv',
                    mime='text/csv',
                )
        except ValueError:
            st.error("Invalid input for Tolerance. Please enter a number between 0 and 1.")

# Display statistics if they are stored in session state
if "stats" in st.session_state:
    stats = st.session_state.stats
    st.write(f"Support Break Accuracy: {stats['support_accuracy']:.2%}")
    st.write(f"Resistance Break Accuracy: {stats['resistance_accuracy']:.2%}")
    st.write(f"Total Support Breaks: {stats['total_support_breaks']}")
    st.write(f"Total Resistance Breaks: {stats['total_resistance_breaks']}")
    st.write(f"Average Support Level: {stats['avg_support_levels']:.2f}")
    st.write(f"Average Resistance Level: {stats['avg_resistance_levels']:.2f}")
    st.write(f"Maximum Support Level: {stats['max_support_levels']:.2f}")
    st.write(f"Minimum Support Level: {stats['min_support_levels']:.2f}")
    st.write(f"Maximum Resistance Level: {stats['max_resistance_levels']:.2f}")
    st.write(f"Minimum Resistance Level: {stats['min_resistance_levels']:.2f}")
    st.write(f"False Support Break Rate: {stats['false_support_break_rate']:.2%}")
    st.write(f"False Resistance Break Rate: {stats['false_resistance_break_rate']:.2%}")
