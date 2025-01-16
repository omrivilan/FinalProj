import streamlit as st
from datetime import datetime
from GraphDrawer import GraphDrawer
import google.generativeai as genai
import matplotlib.pyplot as plt
from io import BytesIO
import os
import pandas as pd
import base64
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
# Global Configuration
API_KEY = 'AIzaSyCTNncuxKui7XIzrZWt1o_EtLIxiew8qtE'
MAPPING_FILE_PATH = "company_name_to_ticker.xlsx"
LSTM_CSV_PATH = "/workspaces/FinalProj/LSTM/actual_vs_pred_stocks.csv"
XGBOOST_CSV_PATH = "/workspaces/FinalProj/XGBoost/model_XGBoost_metrics_and_predictions.csv"
LIGHTGBM_CSV_PATH = "/workspaces/FinalProj/LightGBM/LightGBM_metrics_and_predictions.csv"
BEST_MODEL_CSV = "/workspaces/FinalProj/Metrics/best_model_per_stock.csv"
SECTORS_DF_PATH = "sectors_df.csv"
# Initialize API and Streamlit

# Noam's additional imports
import yfinance as yf
from GraphDrawer import plot_actual_vs_forecasted


def initialize_genai():
    genai.configure(api_key=API_KEY)
    return genai.GenerativeModel(
        model_name="gemini-1.5-flash-8b",
        system_instruction="you are a chat bot designed to help with stock analysis and recommendations over the Israeli (TA) stock market. You will do function calling and textual answers. When you return a graph, add analysis and recommendations. Please be nice and polite."
    )

# Load Ticker Mapping
def load_ticker_mapping():
    company_name_to_ticker = pd.read_excel(MAPPING_FILE_PATH)
    return {row["CompanyName"].upper(): row["Ticker"].upper() for _, row in company_name_to_ticker.iterrows()}

def initialize_data():
    return pd.read_csv(LSTM_CSV_PATH)

# Filter Data
def filter_stock_data(data, ticker, start_date, end_date):
    return data[(data["Ticker"] == ticker) &
                (pd.to_datetime(data["Date"]) >= start_date) &
                (pd.to_datetime(data["Date"]) <= end_date)]

def resolve_ticker(user_input, ticker_mapping):
    user_input = user_input.strip().upper()
    for company_name, ticker in ticker_mapping.items():
        if user_input in company_name or user_input == ticker:
            return ticker
    raise ValueError(f"Ticker or company name '{user_input}' not found in the mapping.")

# Extract Company Names
def extract_company_name(user_input):
    """
    Extracts the company name for single graph requests.
    Matches expanded keywords dynamically.
    """
    user_input = user_input.lower()
    graph_keywords = ["graph of", "graph", "plot of", "plot", "visualize", "show", "chart of", "chart"]

    for keyword in graph_keywords:
        if keyword in user_input:
            return user_input.split(keyword)[1].strip()

    raise ValueError("No recognizable company name in input.")

def extract_company_names(user_input):
    user_input = user_input.lower()
    delimiters = ["compare","comparison", "difference between", "versus", "vs", ",","&","and","And","AND"]
    for delimiter in delimiters:
        user_input = user_input.replace(delimiter, ",")

    companies = user_input.split(",")
    companies = [company.strip() for company in companies if company.strip()]
    if len(companies) < 2:
        raise ValueError("At least two companies are required for comparison.")
    return companies
def detect_intent(user_input):
    """
    Detects the user's intent based on keywords and sentence patterns.
    Returns a dictionary with the detected intent and related information.
    """
    user_input = user_input.lower()

    # Expanded keywords for graph requests
    sector_keywords = ["actual values", "sector", "companies in sector", "all"]
    if any(keyword in user_input for keyword in sector_keywords):
        sector_name = user_input.split("sector")[1].strip() if "sector" in user_input else None
        if sector_name:
            return {"intent": "sector_values", "sector": sector_name}
        raise ValueError("Please specify the sector name for the query.")
    # Expanded keywords for comparison requests
    compare_keywords = ["compare","comparison", "difference between", "versus", "vs", ",","&","and","And","AND"]
    if any(keyword in user_input for keyword in compare_keywords):
        return {"intent": "compare", "companies": extract_company_names(user_input)}
    graph_keywords = ["graph", "plot", "visualize", "show", "chart", "chart of", "plot of", "graph of"]
    if any(keyword in user_input for keyword in graph_keywords):
        if "compare" in user_input or ("and" in user_input or "," in user_input):
            return {"intent": "compare", "companies": extract_company_names(user_input)}
        else:
            return {"intent": "graph", "company": extract_company_name(user_input)}

    # Default to text-based intent
    return {"intent": "text"}

# Extract Data by Model
def extract_data_by_model(company, stocks_model, start_date, end_date):
    csv_path_map = {
        "LSTM": LSTM_CSV_PATH,
        "GRU": "/workspaces/FinalProj/GRU/actual_vs_pred_stocks.csv",
        "XGBoost": XGBOOST_CSV_PATH,
        "LightGBM": LIGHTGBM_CSV_PATH,
    }

    if stocks_model in csv_path_map:
        data = pd.read_csv(csv_path_map[stocks_model])
        if data.empty:
            raise ValueError(f"No data available in the file: {csv_path_map[stocks_model]}")

        if stocks_model in ["LSTM", "GRU"]:
            return filter_stock_data(data, company, start_date, end_date)
        elif stocks_model in ["XGBoost", "LightGBM"]:
            company_data = data[data["Stock"] == company].iloc[0]
            return process_model_data(company_data, start_date)

    if stocks_model == "ARIMA":
        ticker_to_company_mapping = {v: k for k, v in load_ticker_mapping().items()}
        stripped_ticker = company.replace(".TA", "")
        company_name = ticker_to_company_mapping.get(stripped_ticker)

        if not company_name:
            raise ValueError(f"Company name not found for ticker {company}")

        normalized_company_name = company_name.strip().upper()

        actual_path = f"/workspaces/FinalProj/ARIMA/Actuals/{normalized_company_name}_actuals.pkl"
        predicted_path = f"/workspaces/FinalProj/ARIMA/Predictions/{normalized_company_name}_predictions.pkl"

        if not os.path.exists(actual_path) or not os.path.exists(predicted_path):
            raise FileNotFoundError(f"Actuals or predictions file not found for {normalized_company_name}")

        with open(actual_path, "rb") as f:
            actual_data = pickle.load(f)
        with open(predicted_path, "rb") as f:
            predicted_data = pickle.load(f)

        if isinstance(actual_data, np.ndarray):
            actual_data = pd.Series(actual_data, name="Actual")
        if isinstance(predicted_data, np.ndarray):
            predicted_data = pd.Series(predicted_data, name="Predicted")

        if not isinstance(actual_data.index, pd.DatetimeIndex):
            actual_data.index = pd.date_range(start=start_date, periods=len(actual_data))
        if not isinstance(predicted_data.index, pd.DatetimeIndex):
            predicted_data.index = pd.date_range(start=start_date, periods=len(predicted_data))

        actual_df = pd.DataFrame({"Date": actual_data.index, "Actual": actual_data.values})
        predicted_df = pd.DataFrame({"Date": predicted_data.index, "Predicted": predicted_data.values})

        merged_data = pd.merge(actual_df, predicted_df, on="Date")
        return merged_data[
            (pd.to_datetime(merged_data["Date"]) >= start_date) &
            (pd.to_datetime(merged_data["Date"]) <= end_date)
        ]

    raise ValueError(f"Unsupported model type: {stocks_model}")

def process_model_data(company_data, start_date):
    company_data["y_test"] = eval(company_data["y_test"])
    company_data["y_pred"] = eval(company_data["y_pred"])

    actual_df = pd.DataFrame({
        "Date": pd.date_range(start=start_date, periods=len(company_data["y_test"])),
        "Actual": company_data["y_test"]
    })
    predicted_df = pd.DataFrame({
        "Date": pd.date_range(start=start_date, periods=len(company_data["y_pred"])),
        "Predicted": company_data["y_pred"]
    })
    return pd.merge(actual_df, predicted_df, on="Date")

def extract_forecasted_values(stock, last_predicted_date):
    """
    Extract forecasted values for a given stock and align the forecasted dates with last_predicted_date.

    Args:
        stock (str): Ticker of the stock.
        last_predicted_date (datetime): The last date in the predicted data.

    Returns:
        DataFrame: A DataFrame containing 'Date' and 'Forecasted' columns.
    """
    # Define file paths for each model
    forecast_files = {
        "LSTM": "/workspaces/FinalProj/LSTM/forecast_stocks.csv",
        "GRU": "/workspaces/FinalProj/GRU/forecast_stocks.csv",
        "LightGBM": "/workspaces/FinalProj/LightGBM/LightGBM_forecast_stocks.csv",
        "XGBoost": "/workspaces/FinalProj/XGBoost/XGBoost_forecast_stocks.csv"
    }

    # Load the model-to-stock mapping
    best_model_df = pd.read_csv("/workspaces/FinalProj/Metrics/without_ARIMA_model_to_stock.csv")

    # Get the best model for the stock
    best_model = best_model_df.loc[best_model_df["Company"] == stock, "Model"].values[0]

    if best_model not in forecast_files:
        raise ValueError(f"Unsupported model for forecasting: {best_model}")

    # Load the forecast file
    forecast_file = forecast_files[best_model]
    forecast_data = pd.read_csv(forecast_file)

    if best_model in ["LSTM", "GRU"]:
        # Filter for the specific stock and select the necessary columns
        filtered_forecast = forecast_data[forecast_data["Ticker"] == stock][["Date", "Forecast"]]
        filtered_forecast["Date"] = pd.to_datetime(filtered_forecast["Date"])
    elif best_model in ["LightGBM", "XGBoost"]:
        # Filter for the specific stock
        filtered_forecast = forecast_data[forecast_data["Ticker"] == stock]

        # Extract the "Day" and "Forecast" columns
        filtered_forecast = filtered_forecast[["Day", "Forecast"]]

        # Generate actual dates from the "Day" column
        day_offsets = filtered_forecast["Day"].str.extract(r"Day_(\d+)").astype(int)
        filtered_forecast["Date"] = [
            last_predicted_date + timedelta(days=int(offset)) for offset in day_offsets[0]
        ]

        # Drop the "Day" column
        filtered_forecast = filtered_forecast[["Date", "Forecast"]]

    # Rename "Forecast" column to "Forecasted" for consistency
    filtered_forecast.rename(columns={"Forecast": "Forecasted"}, inplace=True)

    return filtered_forecast

def generate_graph_with_forecast(stock, model):
    """
    Generates a graph showing actual, predicted, and forecasted values for a stock.

    Args:
        stock (str): Ticker of the stock.
        model (str): Model type (LSTM, GRU, LightGBM, XGBoost, ARIMA).

    Returns:
        BytesIO: Buffer containing the graph image.
    """
    # Step 1: Extract actual and predicted data
    actual_predicted = extract_data_by_model(stock, model, datetime(2024, 1, 1), datetime.today())

    # Ensure Date column in actual_predicted is properly converted
    actual_predicted["Date"] = pd.to_datetime(actual_predicted["Date"], errors="coerce")
    if actual_predicted["Date"].isna().any():
        raise ValueError("Invalid dates found in actual or predicted data.")

    # Step 2: Extract forecasted values
    last_predicted_date = actual_predicted["Date"].max()
    forecasted = extract_forecasted_values(stock, last_predicted_date)

    # Step 3: Align forecasted dates
    forecasted["Date"] = pd.date_range(
        start=last_predicted_date + timedelta(days=1),
        periods=len(forecasted),
        freq="D"
    )

    # Step 4: Concatenate the datasets
    combined_data = pd.concat(
        [
            actual_predicted[["Date", "Actual", "Predicted"]],
            forecasted.rename(columns={"Forecast": "Forecasted"})[["Date", "Forecasted"]],
        ],
        ignore_index=True,
    ).sort_values(by="Date")

    # Step 5: Plot the data
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot actual values
    ax.plot(
        combined_data["Date"],
        combined_data["Actual"],
        label="Actual",
        color="blue",
        linewidth=1.5,
    )

    # Plot predicted values
    ax.plot(
        combined_data["Date"],
        combined_data["Predicted"],
        label="Predicted",
        color="orange",
        linestyle="--",
        linewidth=1.5,
    )

    # Plot forecasted values
    if "Forecasted" in combined_data:
        ax.plot(
            combined_data["Date"],
            combined_data["Forecasted"],
            label="Forecasted",
            color="green",
            linestyle=":",
            linewidth=1.5,
        )

    # Step 6: Set plot details
    ax.set_title(f"Actual, Predicted, and Forecasted Values for {stock} ({model})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

    # Step 7: Save the plot to a buffer
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close(fig)

    return buffer





# Generate and Display Graph
def generate_graph(data, title):
    """
    Generate a graph with consistent x-axis formatting (YYYY-MM) and dynamic date range.
    
    Args:
        data: DataFrame containing "Date", "Actual", and "Predicted"
        title: Title of the graph
    
    Returns:
        BytesIO buffer with the graph image
    """
    # Parse dates
    data["Date"] = pd.to_datetime(data["Date"], dayfirst=True)
    
    # Determine dynamic start_date and end_date
    start_date = data["Date"].min()
    end_date = data["Date"].max()
    
    # Filter data based on the dynamic range
    filtered_data = data[(data["Date"] >= start_date) & (data["Date"] <= end_date)]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(filtered_data["Date"], filtered_data["Actual"], label="Actual", linestyle="-", color="blue")
    ax.plot(filtered_data["Date"], filtered_data["Predicted"], label="Predicted", linestyle="--", color="orange")
    
    # Set x-axis limits dynamically
    ax.set_xlim([start_date, end_date])
    
    # Format the x-axis as YYYY-MM
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # Adjust interval as needed
    plt.xticks(rotation=45)  # Rotate for readability
    
    # Add labels and title
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    
    # Save the plot to a buffer
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close(fig)
    return buffer


def display_graph(buffer):
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    image_html = f'<img src="data:image/png;base64,{image_base64}" alt="Stock Plot">'
    return image_html,image_base64

def synchronize_x_axis(dataframes):
    """
    Synchronize the x-axis (Date) across all dataframes by creating a unified date range.
    
    Args:
        dataframes (list): List of pandas DataFrames to synchronize.
    
    Returns:
        list: List of synchronized DataFrames.
    """
    # Determine the common date range
    all_dates = pd.concat([df['Date'] for df in dataframes])
    unified_date_range = pd.date_range(start=all_dates.min(), end=all_dates.max())

    # Align each dataframe to the unified date range
    synchronized_dataframes = []
    for df in dataframes:
        df.set_index('Date', inplace=True)
        df = df.reindex(unified_date_range).reset_index()
        df.rename(columns={'index': 'Date'}, inplace=True)
        df.fillna(method='ffill', inplace=True)  # Forward-fill to handle missing data
        synchronized_dataframes.append(df)

    return synchronized_dataframes

# New Function to Fetch Actual Values by Sector
def get_actual_values_by_sector(sector, sectors_df_path=SECTORS_DF_PATH, start_date=None, end_date=None):
    try:
        # Load sector data
        sectors_df = pd.read_csv(sectors_df_path)

        # Filter companies in the given sector
        sector_companies = sectors_df[sectors_df["Market Sector"].str.lower() == sector.lower()]
        if sector_companies.empty:
            raise ValueError(f"No companies found in the sector '{sector}'.")

        # Extract tickers
        tickers = sector_companies["Symbol"].tolist()

        # Container for data
        all_actual_values = []

        for ticker in tickers:
            # Add ".TA" for Israeli market tickers
            ticker += ".TA"

            # Use LSTM as the default model for demonstration (this can be extended)
            stocks_model = "LSTM"

            # Fetch data
            data = extract_data_by_model(ticker, stocks_model, start_date, end_date)
            if not data.empty:
                all_actual_values.append((ticker, data["Date"], data["Actual"]))

        # Combine all actual values
        if not all_actual_values:
            raise ValueError("No actual values found for the companies in the sector.")

        result_df = pd.DataFrame(columns=["Ticker", "Date", "Actual"])
        for ticker, dates, actuals in all_actual_values:
            df = pd.DataFrame({"Ticker": ticker, "Date": dates, "Actual": actuals})
            result_df = pd.concat([result_df, df], ignore_index=True)

        return result_df

    except Exception as e:
        st.error(f"Error processing sector data: {e}")
        return None

def get_predicted_values_by_sector(sector, sectors_df_path=SECTORS_DF_PATH, start_date=None, end_date=None):
    try:
        # Load sector data
        sectors_df = pd.read_csv(sectors_df_path)

        # Filter companies in the given sector
        sector_companies = sectors_df[sectors_df["Market Sector"].str.lower() == sector.lower()]
        if sector_companies.empty:
            raise ValueError(f"No companies found in the sector '{sector}'.")

        # Extract tickers
        tickers = sector_companies["Symbol"].tolist()

        # Reverse the ticker mapping
        ticker_to_company_name = {v: k for k, v in load_ticker_mapping().items()}

        # Load best model mapping
        best_model_df = pd.read_csv(BEST_MODEL_CSV)

        # Container for data
        all_predicted_values = []

        for ticker in tickers:
            # Add ".TA" for Israeli market tickers
            ticker += ".TA"

            # Get the best model for the ticker
            stocks_model = best_model_df.loc[best_model_df["Company"] == ticker, "Model"].values[0]

            # Fetch data
            data = extract_data_by_model(ticker, stocks_model, start_date, end_date)
            if not data.empty:
                all_predicted_values.append((ticker, data["Date"], data["Predicted"]))

        # Combine all predicted values
        if not all_predicted_values:
            raise ValueError("No predicted values found for the companies in the sector.")

        result_df = pd.DataFrame(columns=["Ticker", "Date", "Predicted"])
        for ticker, dates, predicted in all_predicted_values:
            df = pd.DataFrame({
                "Ticker": ticker,
                "Date": dates,
                "Predicted": predicted,
                "Company": ticker_to_company_name.get(ticker.replace(".TA", ""), ticker)  # Map company names
            })
            result_df = pd.concat([result_df, df], ignore_index=True)

        return result_df

    except Exception as e:
        st.error(f"Error processing sector predictions: {e}")
        return None

def generate_predicted_values_graph(data, sector_name):
    fig, ax = plt.subplots(figsize=(12, 8))

    for company, company_data in data.groupby("Company"):
        ax.plot(
            pd.to_datetime(company_data["Date"]),
            company_data["Predicted"],
            label=company
        )

    ax.set_title(f"Predicted Values for {sector_name} Sector")
    ax.set_xlabel("Date")
    ax.set_ylabel("Predicted Value")
    ax.legend()

    # Format x-axis for every 50 days
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=50))  # Set interval to 50 days
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))  # Format as 'Month Day, Year'
    plt.xticks(rotation=45)  # Rotate labels for better readability

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close(fig)

    return buffer


# Helper Functions
def calculate_monthly_trends(data):
    data['Month'] = pd.to_datetime(data['Date']).dt.to_period('M')
    trends = data.groupby('Month')['Actual'].agg(['first', 'last']).reset_index()
    trends['Trend'] = trends['last'] - trends['first']
    return trends

def identify_model_misses(data, threshold=0.1):
    data['Absolute_Error'] = abs(data['Actual'] - data['Predicted'])
    significant_misses = data[data['Absolute_Error'] > threshold * data['Actual']]
    return significant_misses
def synchronize_and_plot_lstm_adjusted_comparison(dataframes, company_models, title):
    """
    Synchronize the date range across multiple dataframes by modifying only LSTM/GRU models.
    
    Args:
        dataframes: List of tuples (ticker, stocks_model, data)
        company_models: Dictionary of company and best model type
        title: Title for the plot
    """
    # Find the unified date range
    all_dates = pd.concat([df[2]['Date'] for df in dataframes])
    unified_start_date = all_dates.min()
    unified_end_date = all_dates.max()

    # Synchronize data
    synchronized_dataframes = []
    for ticker, stocks_model, data in dataframes:
        # Convert Date to datetime
        data["Date"] = pd.to_datetime(data["Date"], dayfirst=True)
        
        if stocks_model in ["LSTM", "GRU"]:
            # Extend Actual values backward
            extended_data = data.copy()
            first_actual_value = extended_data["Actual"].iloc[0]
            extended_data = pd.concat([
                pd.DataFrame({"Date": pd.date_range(unified_start_date, extended_data["Date"].iloc[0] - pd.Timedelta(days=1)),
                              "Actual": first_actual_value}),
                extended_data
            ], ignore_index=True)
            # Truncate data to unified_end_date
            extended_data = extended_data[extended_data["Date"] <= unified_end_date]
            synchronized_dataframes.append((ticker, stocks_model, extended_data))
        else:
            # For other models, truncate directly
            truncated_data = data[(data["Date"] >= unified_start_date) & (data["Date"] <= unified_end_date)]
            synchronized_dataframes.append((ticker, stocks_model, truncated_data))
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    for ticker, stocks_model, data in synchronized_dataframes:
        company_name = ticker.replace(".TA", "")  # Replace ".TA" to get the company name
        ax.plot(
            data["Date"],
            data["Actual"],
            label=f"{company_name} Actual ({stocks_model})",
            linestyle="-"
        )
        ax.plot(
            data["Date"],
            data["Predicted"],
            label=f"{company_name} Predicted ({stocks_model})",
            linestyle="--"
        )
    
    # Set titles and labels
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)

    # Save and return the plot
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close(fig)
    return buffer


# Chatbot Response Logic
def chatbot_response(user_input, model, ticker_mapping):
    try:
        intent_data = detect_intent(user_input)
        if intent_data["intent"] == "sector_values":
            sector_name = intent_data["sector"]
            start_date, end_date = datetime(2024, 1, 1), datetime.today()
            sector_data_actual = get_actual_values_by_sector(sector_name, start_date=start_date, end_date=end_date)
            sector_data_actual["Date"] = pd.to_datetime(sector_data_actual["Date"], dayfirst=True)

            # Reverse the mapping to get company names from tickers
            ticker_to_company_name = {v: k for k, v in ticker_mapping.items()}

            # Generate Actual Values Graph and Collect Summary
            actual_values_summary = []
            if sector_data_actual is not None and not sector_data_actual.empty:
                fig_actual, ax_actual = plt.subplots(figsize=(12, 8))
                for ticker, data in sector_data_actual.groupby("Ticker"):
                    company_name = ticker_to_company_name.get(ticker.replace(".TA", ""), ticker)
                    ax_actual.plot(pd.to_datetime(data["Date"]), data["Actual"], label=company_name)

                    # Collect actual values summary
                    actual_summary = {
                        "Company Name": company_name,
                        "Min": data["Actual"].min(),
                        "Max": data["Actual"].max(),
                        "Mean": data["Actual"].mean(),
                        "Trend": "upward" if data["Actual"].iloc[-1] > data["Actual"].iloc[0] else "downward"
                    }
                    actual_values_summary.append(actual_summary)

                ax_actual.set_title(f"Actual Values for {sector_name} Sector")
                ax_actual.set_xlabel("Date")
                ax_actual.set_ylabel("Actual Value")
                ax_actual.legend()

                dates_actual = pd.to_datetime(sector_data_actual["Date"].unique())
                tick_positions_actual = dates_actual[::20]
                ax_actual.set_xticks(tick_positions_actual)
                ax_actual.set_xticklabels(tick_positions_actual.strftime('%b %d, %Y'))
                plt.xticks(rotation=45)

                buffer_actual = BytesIO()
                plt.savefig(buffer_actual, format="png")
                buffer_actual.seek(0)
                plt.close(fig_actual)

                image_html_actual, image_base64_actual = display_graph(buffer_actual)

            # Generate Predicted Values Graph and Collect Summary
            sector_data_predicted = get_predicted_values_by_sector(sector_name, start_date=start_date, end_date=end_date)
            sector_data_predicted["Date"] = pd.to_datetime(sector_data_predicted["Date"], dayfirst=True)
            predicted_values_summary = []
            if sector_data_predicted is not None and not sector_data_predicted.empty:
                fig_predicted, ax_predicted = plt.subplots(figsize=(12, 8))
                for ticker, data in sector_data_predicted.groupby("Ticker"):
                    company_name = ticker_to_company_name.get(ticker.replace(".TA", ""), ticker)
                    ax_predicted.plot(pd.to_datetime(data["Date"]), data["Predicted"], label=company_name)

                    # Collect predicted values summary
                    predicted_summary = {
                        "Company Name": company_name,
                        "Min": data["Predicted"].min(),
                        "Max": data["Predicted"].max(),
                        "Mean": data["Predicted"].mean(),
                        "Trend": "upward" if data["Predicted"].iloc[-1] > data["Predicted"].iloc[0] else "downward"
                    }
                    predicted_values_summary.append(predicted_summary)

                ax_predicted.set_title(f"Predicted Values for {sector_name} Sector")
                ax_predicted.set_xlabel("Date")
                ax_predicted.set_ylabel("Predicted Value")
                ax_predicted.legend()

                dates_predicted = pd.to_datetime(sector_data_predicted["Date"].unique())
                tick_positions_predicted = dates_predicted[::20]
                ax_predicted.set_xticks(tick_positions_predicted)
                ax_predicted.set_xticklabels(tick_positions_predicted.strftime('%b %d, %Y'))
                plt.xticks(rotation=45)

                buffer_predicted = BytesIO()
                plt.savefig(buffer_predicted, format="png")
                buffer_predicted.seek(0)
                plt.close(fig_predicted)

                image_html_predicted, image_base64_predicted = display_graph(buffer_predicted)

            # Format sector-wide summary for text generation
            formatted_sector_input = (
                f"Sector-wide analysis for {sector_name} sector:\n\n"
                "Actual Values Summary:\n" +
                "\n".join(
                    f"- {summary['Company Name']}:\n"
                    f"  Min: {summary['Min']:.2f}, Max: {summary['Max']:.2f}, "
                    f"Mean: {summary['Mean']:.2f}, Trend: {summary['Trend']}"
                    for summary in actual_values_summary
                ) +
                "\n\nPredicted Values Summary:\n" +
                "\n".join(
                    f"- {summary['Company Name']}:\n"
                    f"  Min: {summary['Min']:.2f}, Max: {summary['Max']:.2f}, "
                    f"Mean: {summary['Mean']:.2f}, Trend: {summary['Trend']}"
                    for summary in predicted_values_summary
                ) +
                "\n\nPlease provide an analysis comparing the trends, values, and predictions for these companies, "
                "highlight which companies seem to be performing better in the sector, and suggest any insights or recommendations."
            )

            # Generate textual analysis
            generated_sector_text = model.generate_content(formatted_sector_input).text
            

            # Return both graphs and text
            return {
                "actual_html": image_html_actual,
                "actual_image": image_base64_actual,
                "predicted_html": image_html_predicted,
                "predicted_image": image_base64_predicted,
                "text": generated_sector_text
            }



        if intent_data["intent"] == "graph":
            company_name = extract_company_name(user_input).upper()
            ticker = resolve_ticker(company_name, ticker_mapping) + ".TA"
            best_model_df = pd.read_csv(BEST_MODEL_CSV)
            stocks_model = best_model_df.loc[best_model_df["Company"] == ticker, "Model"].values[0]

            start_date, end_date = datetime(2024, 1, 1), datetime.today()
            data = extract_data_by_model(ticker, stocks_model, start_date, end_date)

            # Generate the graph
            buffer = generate_graph_with_forecast(ticker, stocks_model)
            image_html, image_base64 = display_graph(buffer)

            # Reformat the data for better text generation
            summary_data = {
                "Company Name": company_name,
                "Date Range": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                "Model Used": stocks_model,
                "Data Points": len(data),
                "Actual Values Summary": {
                    "Min": data["Actual"].min(),
                    "Max": data["Actual"].max(),
                    "Mean": data["Actual"].mean(),
                    "Trend": "upward" if data["Actual"].iloc[-1] > data["Actual"].iloc[0] else "downward"
                },
                "Predicted Values Summary": {
                    "Min": data["Predicted"].min(),
                    "Max": data["Predicted"].max(),
                    "Mean": data["Predicted"].mean(),
                    "Trend": "upward" if data["Predicted"].iloc[-1] > data["Predicted"].iloc[0] else "downward"
                }
            }

            # Format the summary data into a readable text input for the generative model
            formatted_input = (
                f"Stock analysis for {company_name}:\n"
                f"Date range: {summary_data['Date Range']}\n"
                f"Model used: {summary_data['Model Used']}\n"
                f"Number of data points: {summary_data['Data Points']}\n\n"
                f"Actual Values Summary:\n"
                f"  - Min: {summary_data['Actual Values Summary']['Min']:.2f}\n"
                f"  - Max: {summary_data['Actual Values Summary']['Max']:.2f}\n"
                f"  - Mean: {summary_data['Actual Values Summary']['Mean']:.2f}\n"
                f"  - Trend: {summary_data['Actual Values Summary']['Trend']}\n\n"
                f"Predicted Values Summary:\n"
                f"  - Min: {summary_data['Predicted Values Summary']['Min']:.2f}\n"
                f"  - Max: {summary_data['Predicted Values Summary']['Max']:.2f}\n"
                f"  - Mean: {summary_data['Predicted Values Summary']['Mean']:.2f}\n"
                f"  - Trend: {summary_data['Predicted Values Summary']['Trend']}\n\n"
                "Please provide an analysis discussing the trends, values, and predictions for this company, "
                "with any observations about the stock's performance based on the provided data."
            )

            # Use the generative AI model to create an analysis
            generated_text = model.generate_content(formatted_input).text

            # Return both graph and text
            return {
                "html": image_html,
                "image": image_base64,
                "text": generated_text
            }


        elif intent_data["intent"] == "compare":
            company_names = extract_company_names(user_input)
            tickers = [resolve_ticker(name.upper(), ticker_mapping) + ".TA" for name in company_names]

            # Reverse the mapping to get company names from tickers
            ticker_to_company_name = {v: k for k, v in ticker_mapping.items()}

            # Define date range
            start_date, end_date = datetime(2024, 1, 1), datetime.today()

            # Load the best model mapping
            best_model_df = pd.read_csv(BEST_MODEL_CSV)

            # Container for data
            data_frames = []

            # Collect data and metadata for each ticker
            comparison_summary = []
            for ticker in tickers:
                try:
                    # Get the best model for the company
                    stocks_model = best_model_df.loc[best_model_df["Company"] == ticker, "Model"].values[0]

                    # Extract the data
                    data = extract_data_by_model(ticker, stocks_model, start_date, end_date)
                    if not data.empty:
                        # Add to data frames for plotting
                        data_frames.append((ticker, stocks_model, data))

                        # Prepare summary data
                        company_name = ticker_to_company_name.get(ticker.replace(".TA", ""), ticker)
                        company_summary = {
                            "Company Name": company_name,
                            "Model Used": stocks_model,
                            "Actual Values Summary": {
                                "Min": data["Actual"].min(),
                                "Max": data["Actual"].max(),
                                "Mean": data["Actual"].mean(),
                                "Trend": "upward" if data["Actual"].iloc[-1] > data["Actual"].iloc[0] else "downward"
                            },
                            "Predicted Values Summary": {
                                "Min": data["Predicted"].min(),
                                "Max": data["Predicted"].max(),
                                "Mean": data["Predicted"].mean(),
                                "Trend": "upward" if data["Predicted"].iloc[-1] > data["Predicted"].iloc[0] else "downward"
                            }
                        }
                        comparison_summary.append(company_summary)
                    else:
                        st.warning(f"No data available for ticker {ticker}. Skipping.")
                except Exception as e:
                    st.error(f"Error processing {ticker}: {e}")

            # Check if data exists
            if not data_frames:
                return {"text": "No data available for the selected companies."}

            # Plot comparison graph
            fig, ax = plt.subplots(figsize=(12, 8))
            for ticker, stocks_model, data in data_frames:
                company_name = ticker_to_company_name.get(ticker.replace(".TA", ""), ticker)
                ax.plot(
                    pd.to_datetime(data["Date"]),
                    data["Actual"],
                    label=f"{company_name} Actual ({stocks_model})",
                    linestyle="-"
                )
                ax.plot(
                    pd.to_datetime(data["Date"]),
                    data["Predicted"],
                    label=f"{company_name} Predicted ({stocks_model})",
                    linestyle="--"
                )

            ax.set_title("Stock Price Comparison (Actual vs Predicted)")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()

            # Save the plot to a buffer
            buffer = BytesIO()
            plt.savefig(buffer, format="png")
            buffer.seek(0)
            plt.close(fig)

            # Generate HTML for the graph
            image_html, image_base64 = display_graph(buffer)

            # Format comparison summary for the generative model
            formatted_comparison_input = (
                f"Comparison of multiple stocks:\n\n"
                + "\n\n".join(
                    f"Company: {summary['Company Name']}\n"
                    f"Model Used: {summary['Model Used']}\n"
                    f"Actual Values Summary:\n"
                    f"  - Min: {summary['Actual Values Summary']['Min']:.2f}\n"
                    f"  - Max: {summary['Actual Values Summary']['Max']:.2f}\n"
                    f"  - Mean: {summary['Actual Values Summary']['Mean']:.2f}\n"
                    f"  - Trend: {summary['Actual Values Summary']['Trend']}\n"
                    f"Predicted Values Summary:\n"
                    f"  - Min: {summary['Predicted Values Summary']['Min']:.2f}\n"
                    f"  - Max: {summary['Predicted Values Summary']['Max']:.2f}\n"
                    f"  - Mean: {summary['Predicted Values Summary']['Mean']:.2f}\n"
                    f"  - Trend: {summary['Predicted Values Summary']['Trend']}"
                    for summary in comparison_summary
                )
                + "\n\nPlease provide an analysis comparing the trends, values, and predictions for these companies, "
                "highlight which company seems better based on actual values, and include any interesting observations."
            )

            # Generate textual analysis
            generated_comparison_text = model.generate_content(formatted_comparison_input).text

            # Return both graph and text
            return {
                "html": image_html,
                "image": image_base64,
                "text": generated_comparison_text
            }




        else:
            return {"text": model.generate_content(user_input).text}

    except Exception as e:
        return {"text": f"Error: {str(e)}"}


# Downloads the actual values of Adj Closing Price for each stock, between the specified dates
def download_actual_values(tickers, start_date, end_date, output_file):
    """
    Downloads actual Adj Close prices from Yahoo Finance for the specified tickers and date range,
    and saves them to a CSV file.
    
    :param tickers: List of stock tickers
    :param start_date: Start date as a string in 'YYYY-MM-DD'
    :param end_date: End date as a string in 'YYYY-MM-DD'
    :param output_file: Path to the output CSV file
    """
    all_data = {}
    
    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        if 'Adj Close' in data.columns:
            all_data[ticker] = data['Adj Close']
    
    result_df = pd.DataFrame(all_data)
    result_df.index.name = 'Date'
    result_df.to_csv(output_file)
    print(f"Actual values saved to {output_file}")

# Saves the Forecasted values in a CSV file
def save_forecasted_values(predictions, output_file):
    """
    Saves predicted and forecasted values into a CSV file.
    
    :param predictions: DataFrame containing 'Date', 'Ticker', 'Predicted', 'Forecasted'
    :param output_file: Path to the output CSV file
    """
    predictions.to_csv(output_file, index=False)
    print(f"Predicted and forecasted values saved to {output_file}")



# Main Streamlit App
def main():
    st.set_page_config(page_title="Chatbot", layout="wide")
    st.title("Chatbot Application")

    st.sidebar.title("About")
    st.sidebar.info("This is a simple chatbot app built using Streamlit.")

    model = initialize_genai()
    ticker_mapping = load_ticker_mapping()

    ######################################################################
    #download_actual_values()
    #save_forecasted_values()

    st.sidebar.subheader("Actual vs. Forecasted Graph")
    stock_ticker = st.sidebar.text_input("Enter Stock Ticker", value="")
    actual_file = st.sidebar.text_input("Path to Actual Values CSV", value="path/to/actual_values.csv")
    forecasted_file = st.sidebar.text_input("Path to Forecasted Values CSV", value="path/to/forecasted_values.csv")

    if st.sidebar.button("Show Graph"):
        if stock_ticker and actual_file and forecasted_file:
            try:
                plot_actual_vs_forecasted(actual_file, forecasted_file)
                st.sidebar.success(f"Graph displayed for {stock_ticker}.")
            except Exception as e:
                st.sidebar.error(f"Error generating graph: {e}")
        else:
            st.sidebar.warning("Please provide all required inputs.")
    ######################################################################

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            # Check if the message contains an image
            if "image" in message and "content" in message:
                # Display the image
                image_html = f'<img src="data:image/png;base64,{message["image"]}" alt="Stock Plot">'
                st.markdown(image_html, unsafe_allow_html=True)

                # Display the text content
                st.markdown(message["content"])
            elif "image" in message:
                image_html = f'<img src="data:image/png;base64,{message["image"]}" alt="Stock Plot">'
                st.markdown(image_html, unsafe_allow_html=True)
            elif "actual_image" in message and "predicted_image" in message and "content" in message:
                # Render actual values graph
                actual_image_html = f'<img src="data:image/png;base64,{message["actual_image"]}" alt="Actual Values Plot">'
                st.markdown(actual_image_html, unsafe_allow_html=True)
                # Render predicted values graph
                predicted_image_html = f'<img src="data:image/png;base64,{message["predicted_image"]}" alt="Predicted Values Plot">'
                st.markdown(predicted_image_html, unsafe_allow_html=True)
                st.markdown(message["content"])
            elif "actual_image" in message and "predicted_image" in message:
                # Render actual values graph
                actual_image_html = f'<img src="data:image/png;base64,{message["actual_image"]}" alt="Actual Values Plot">'
                st.markdown(actual_image_html, unsafe_allow_html=True)
                # Render predicted values graph
                predicted_image_html = f'<img src="data:image/png;base64,{message["predicted_image"]}" alt="Predicted Values Plot">'
                st.markdown(predicted_image_html, unsafe_allow_html=True)
            elif "content" in message:
                # Render text content
                st.markdown(message["content"])
            else:
                # Handle unexpected cases gracefully
                st.markdown("An unknown message format was encountered.")


    if prompt := st.chat_input("What is up?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        response = chatbot_response(prompt, model, ticker_mapping)
        with st.chat_message("assistant"):
            # Check if response contains actual and predicted images
            if "image" in response and "text" in response:
                # Display the graph
                image_html = f'<img src="data:image/png;base64,{response["image"]}" alt="Stock Plot">'
                st.markdown(image_html, unsafe_allow_html=True)

                # Display the generated text
                st.markdown(response["text"])

                # Save both to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "image": response["image"],
                    "content": response["text"],
                })
            elif "actual_image" in response and "predicted_image" in response and "text" in response:
                # Display actual values graph
                actual_image_html = f'<img src="data:image/png;base64,{response["actual_image"]}" alt="Actual Values Plot">'
                st.markdown(actual_image_html, unsafe_allow_html=True)

                # Display predicted values graph
                predicted_image_html = f'<img src="data:image/png;base64,{response["predicted_image"]}" alt="Predicted Values Plot">'
                st.markdown(predicted_image_html, unsafe_allow_html=True)

                # Display the generated text
                st.markdown(response["text"])

                # Save both graphs and text to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "actual_image": response["actual_image"],
                    "predicted_image": response["predicted_image"],
                    "content": response["text"]
                })
            elif "actual_image" in response and "predicted_image" in response and "text" not in response:
                # Display actual values graph
                actual_image_html = f'<img src="data:image/png;base64,{response["actual_image"]}" alt="Actual Values Plot">'
                st.markdown(actual_image_html, unsafe_allow_html=True)

                # Display predicted values graph
                predicted_image_html = f'<img src="data:image/png;base64,{response["predicted_image"]}" alt="Predicted Values Plot">'
                st.markdown(predicted_image_html, unsafe_allow_html=True)

                # Save both graphs to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "actual_image": response["actual_image"],
                    "predicted_image": response["predicted_image"],
                })

            # Check if response contains a single image
            elif "image" in response:
                image_html = f'<img src="data:image/png;base64,{response["image"]}" alt="Stock Plot">'
                st.markdown(image_html, unsafe_allow_html=True)
                st.session_state.chat_history.append({"role": "assistant", "image": response["image"]})

            # Default to text response
            elif "text" in response:
                st.markdown(response["text"])
                st.session_state.chat_history.append({"role": "assistant", "content": response["text"]})

            # Handle unexpected cases (e.g., no text or image in response)
            else:
                st.markdown("I'm sorry, I couldn't process your request.")
                st.session_state.chat_history.append({"role": "assistant", "content": "Error: No valid response."})



if __name__ == "__main__":
    main()
