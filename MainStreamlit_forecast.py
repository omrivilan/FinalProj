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
import plotly.graph_objects as go
import warnings
import time
import emoji

warnings.filterwarnings("ignore")

# Global Configuration
API_KEY = 'AIzaSyCTNncuxKui7XIzrZWt1o_EtLIxiew8qtE'
MAPPING_FILE_PATH = "company_name_to_ticker.xlsx"
LSTM_CSV_PATH = "/workspaces/FinalProj/LSTM/actual_vs_pred_stocks.csv"
XGBOOST_CSV_PATH = "/workspaces/FinalProj/XGBoost/model_XGBoost_metrics_and_predictions.csv"
LIGHTGBM_CSV_PATH = "/workspaces/FinalProj/LightGBM/LightGBM_metrics_and_predictions.csv"
BEST_MODEL_CSV = "/workspaces/FinalProj/Metrics/best_model_per_stock.csv"
SECTORS_DF_PATH = "sectors_df.csv"
# Initialize API and Streamlit

if "mentioned_tickers" not in st.session_state:
    st.session_state.mentioned_tickers = set()

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
    delimiters = [
        "compare", "comparison", "difference between", "versus", "vs",
        ",", "&", "and", "And", "AND", "compare me the stocks of", "with"
    ]

    for delimiter in delimiters:
        user_input = user_input.replace(delimiter, ",")

    companies = user_input.split(",")
    companies = [company.strip() for company in companies if company.strip()]
    
    # Load ticker mapping
    ticker_mapping = load_ticker_mapping()

    # Match substrings to valid company names
    valid_companies = []
    for company in companies:
        for valid_name in ticker_mapping.keys():
            if valid_name in company.upper():
                valid_companies.append(valid_name)
                st.session_state.mentioned_tickers.add(valid_name)  # Add to global mentioned tickers
                break  # Avoid duplicates if the same substring matches multiple names


    compare_set = set(valid_companies) | set(st.session_state.mentioned_tickers)
    if len(compare_set) < 2:
        raise ValueError("At least two valid companies are required for comparison.")
    if "them" or "it" or "previously stocks" in companies:
        return st.session_state.mentioned_tickers
    else:
        return valid_companies

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
    compare_keywords = ["compare","comparison", "difference between", "versus",
     "vs", ",","&","and","And","AND", "compare me the stocks of", "with"]
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

def extract_forecasted_values(stock, last_predicted_date, actual_predicted):
    """
    Extract forecasted values for a given stock and align the forecasted dates with last_predicted_date.
    The first forecasted value is adjusted to match the last predicted value from the actual_predicted DataFrame,
    and the difference is propagated to all subsequent forecasted values.

    Args:
        stock (str): Ticker of the stock.
        last_predicted_date (datetime): The last date in the predicted data.
        actual_predicted (DataFrame): DataFrame containing actual and predicted values.

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

    # Adjust the forecasted values using the last predicted value from actual_predicted
    last_predicted_value = actual_predicted.loc[actual_predicted["Date"] == last_predicted_date, "Predicted"].values[0]
    adjustment = last_predicted_value - filtered_forecast["Forecast"].iloc[0]
    filtered_forecast["Forecast"] += adjustment

    # Rename "Forecast" column to "Forecasted" for consistency
    filtered_forecast.rename(columns={"Forecast": "Forecasted"}, inplace=True)

    return filtered_forecast


def generate_graph_with_forecast(stock, company_name, model):
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
    forecasted = extract_forecasted_values(stock, last_predicted_date, actual_predicted)

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
    # Create the figure
    fig = go.Figure()

    # Plot actual values
    fig.add_trace(
        go.Scatter(
            x=combined_data["Date"],
            y=combined_data["Actual"],
            mode='lines',
            name='Actual',
            line=dict(color='blue', width=2)
        )
    )

    # Plot predicted values
    fig.add_trace(
        go.Scatter(
            x=combined_data["Date"],
            y=combined_data["Predicted"],
            mode='lines',
            name='Predicted',
            line=dict(color='orange', dash='dash', width=2)
        )
    )

    # Plot forecasted values
    if "Forecasted" in combined_data:
        fig.add_trace(
            go.Scatter(
                x=combined_data["Date"],
                y=combined_data["Forecasted"],
                mode='lines',
                name='Forecasted',
                line=dict(color='green', dash='dot', width=2)
            )
        )

    # Set plot details
    fig.update_layout(
        title=f"Actual, Predicted, and Forecasted Values for {stock} ({model})",
        xaxis_title="Date",
        yaxis_title="Value",
        xaxis=dict(tickformat='%Y-%m'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=600,
        width=1000
    )

    return fig, forecasted






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
    
    # Create the figure
    fig = go.Figure()

    # Plot actual values
    fig.add_trace(
        go.Scatter(
            x=filtered_data["Date"],
            y=filtered_data["Actual"],
            mode='lines',
            name='Actual',
            line=dict(color='blue', width=2, dash='solid')
        )
    )

    # Plot predicted values
    fig.add_trace(
        go.Scatter(
            x=filtered_data["Date"],
            y=filtered_data["Predicted"],
            mode='lines',
            name='Predicted',
            line=dict(color='orange', width=2, dash='dash')
        )
    )

    # Set x-axis limits dynamically
    fig.update_xaxes(
        range=[start_date, end_date],
        tickformat='%Y-%m',
        tickangle=45
    )

    # Add labels and title
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=600,
        width=1000
    )

    # Show the plot
    fig.show()

    
    st.plotly_chart(fig, use_container_width=True)



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
        # st.error(f"Error processing sector data: {e}")
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
        # st.error(f"Error processing sector predictions: {e}")
        return None

def generate_predicted_values_graph(data, sector_name):
    # Create the figure
    fig = go.Figure()

    # Plot data for each company
    for company, company_data in data.groupby("Company"):
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(company_data["Date"]),
                y=company_data["Predicted"],
                mode='lines',
                name=company
            )
        )

    # Set title and axis labels
    fig.update_layout(
        title=f"Predicted Values for {sector_name} Sector",
        xaxis_title="Date",
        yaxis_title="Predicted Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=600,
        width=1000
    )

    # Format x-axis for every 50 days
    fig.update_xaxes(
        tickformat='%b %d, %Y',
        dtick=50 * 24 * 60 * 60 * 1000,  # 50 days in milliseconds
        tickangle=45
    )

    st.plotly_chart(fig, use_container_width=True)


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
    # Create the figure
    fig = go.Figure()

    # Plot data for each ticker and model
    for ticker, stocks_model, data in synchronized_dataframes:
        company_name = ticker.replace(".TA", "")  # Replace ".TA" to get the company name
        fig.add_trace(
            go.Scatter(
                x=data["Date"],
                y=data["Actual"],
                mode='lines',
                name=f"{company_name} Actual ({stocks_model})",
                line=dict(color="blue",dash='solid')
            )
        )
        fig.add_trace(
            go.Scatter(
                x=data["Date"],
                y=data["Predicted"],
                mode='lines',
                name=f"{company_name} Predicted ({stocks_model})",
                line=dict(color="orange",dash='dash')
            )
        )

    # Set titles and labels
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=600,
        width=1000
    )

    # Format x-axis as YYYY-MM
    fig.update_xaxes(
        tickformat='%Y-%m',
        dtick=30 * 24 * 60 * 60 * 1000,  # 1 month in milliseconds
        tickangle=45
    )

    st.plotly_chart(fig, use_container_width=True)


# Chatbot Response Logic
def chatbot_response(user_input, model, ticker_mapping):
    try:
        # Include chat history in the prompt
        history = "\n".join(
            f"{msg['role']}: {msg['content']}" 
            for msg in st.session_state.chat_history if 'content' in msg
        )
        prompt_with_history = f"{history}\nuser: {user_input}\nassistant:"
        
        intent_data = detect_intent(user_input)
        if intent_data["intent"] == "sector_values":
            sector_name = intent_data["sector"]
            start_date, end_date = datetime(2024, 1, 1), datetime.today()
            
            # Load sector and best model data
            sector_data_actual = get_actual_values_by_sector(sector_name, start_date=start_date, end_date=end_date)
            best_model_df = pd.read_csv("/workspaces/FinalProj/Metrics/without_ARIMA_model_to_stock.csv")
            sector_data_actual["Date"] = pd.to_datetime(sector_data_actual["Date"], dayfirst=True)

            # Reverse the mapping to get company names from tickers
            ticker_to_company_name = {v: k for k, v in ticker_mapping.items()}

            # Generate Actual Values Graph and Collect Summary
            actual_values_summary = []
            fig_actual = None
            if sector_data_actual is not None and not sector_data_actual.empty:
                fig_actual = go.Figure()
                for ticker, data in sector_data_actual.groupby("Ticker"):
                    company_name = ticker_to_company_name.get(ticker.replace(".TA", ""), ticker)
                    fig_actual.add_trace(
                        go.Scatter(
                            x=pd.to_datetime(data["Date"]),
                            y=data["Actual"],
                            mode='lines',
                            name=company_name
                        )
                    )

                    # Collect actual values summary
                    actual_summary = {
                        "Company Name": company_name,
                        "Min": data["Actual"].min(),
                        "Max": data["Actual"].max(),
                        "Mean": data["Actual"].mean(),
                        "Trend": "upward" if data["Actual"].iloc[-1] > data["Actual"].iloc[0] else "downward"
                    }
                    actual_values_summary.append(actual_summary)

                fig_actual.update_layout(
                    title=f"Actual Values for {sector_name} Sector",
                    xaxis_title="Date",
                    yaxis_title="Actual Value",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=600,
                    width=1000
                )
                fig_actual.update_xaxes(tickformat='%b %d, %Y', tickangle=45)

            # Generate Predicted Values Graph and Collect Summary
            sector_data_predicted = get_predicted_values_by_sector(sector_name, start_date=start_date, end_date=end_date)
            sector_data_predicted["Date"] = pd.to_datetime(sector_data_predicted["Date"], dayfirst=True)
            predicted_values_summary = []
            fig_predicted = None
            if sector_data_predicted is not None and not sector_data_predicted.empty:
                fig_predicted = go.Figure()
                for ticker, data in sector_data_predicted.groupby("Ticker"):
                    company_name = ticker_to_company_name.get(ticker.replace(".TA", ""), ticker)
                    fig_predicted.add_trace(
                        go.Scatter(
                            x=pd.to_datetime(data["Date"]),
                            y=data["Predicted"],
                            mode='lines',
                            name=company_name
                        )
                    )

                    # Collect predicted values summary
                    predicted_summary = {
                        "Company Name": company_name,
                        "Min": data["Predicted"].min(),
                        "Max": data["Predicted"].max(),
                        "Mean": data["Predicted"].mean(),
                        "Trend": "upward" if data["Predicted"].iloc[-1] > data["Predicted"].iloc[0] else "downward"
                    }
                    predicted_values_summary.append(predicted_summary)

                fig_predicted.update_layout(
                    title=f"Predicted Values for {sector_name} Sector",
                    xaxis_title="Date",
                    yaxis_title="Predicted Value",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=600,
                    width=1000
                )
                fig_predicted.update_xaxes(tickformat='%b %d, %Y', tickangle=45)

            # Generate Forecasted Values Graph and Collect Summary
            forecasted_values_summary = []
            fig_forecasted=go.Figure()
            for ticker in sector_data_actual["Ticker"].unique():
                try:
                    # Determine the best model for the company
                    stocks_model = best_model_df.loc[best_model_df["Company"] == ticker, "Model"].values[0]

                    # Extract actual-predicted data and forecasted values
                    actual_predicted = extract_data_by_model(ticker, stocks_model, start_date, end_date)
                    if actual_predicted.empty:
                        continue

                    last_predicted_date = actual_predicted["Date"].max()
                    forecasted = extract_forecasted_values(ticker, last_predicted_date, actual_predicted)

                    # Align forecasted dates to start from 2024-09-01
                    forecast_start_date = datetime(2024, 9, 1)
                    forecasted["Date"] = pd.date_range(start=forecast_start_date, periods=len(forecasted), freq="D")

                    # Plot forecasted values
                    company_name = ticker_to_company_name.get(ticker.replace(".TA", ""), ticker)
                    fig_forecasted.add_trace(
                        go.Scatter(
                            x=pd.to_datetime(forecasted["Date"]),
                            y=forecasted["Forecasted"],
                            mode='lines',
                            name=company_name
                        )
                    )
                    # Collect forecasted values summary
                    forecasted_summary = {
                        "Company Name": company_name,
                        "Min": forecasted["Forecasted"].min(),
                        "Max": forecasted["Forecasted"].max(),
                        "Mean": forecasted["Forecasted"].mean(),
                        "Trend": "upward" if forecasted["Forecasted"].iloc[-1] > forecasted["Forecasted"].iloc[0] else "downward"
                    }
                    forecasted_values_summary.append(forecasted_summary)

                    fig_forecasted.update_layout(
                    title=f"Forecasted Values for {sector_name} Sector",
                    xaxis_title="Date",
                    yaxis_title="Forecasted Value",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=600,
                    width=1000
                    )
                    fig_forecasted.update_xaxes(tickformat='%b %d, %Y', tickangle=45)

                except Exception as e:
                    # st.error(f"Could not process forecasted values for {ticker}: {e}")
                    return None

            # ax_forecasted.set_title(f"Forecasted Values for {sector_name} Sector")
            # ax_forecasted.set_xlabel("Date")
            # ax_forecasted.set_ylabel("Forecasted Value")
            # ax_forecasted.legend()

            # # Format x-axis for better readability
            # dates_forecasted = pd.date_range(start=forecast_start_date, periods=len(forecasted), freq="D")
            # tick_positions_forecasted = dates_forecasted[::5]
            # ax_forecasted.set_xticks(tick_positions_forecasted)
            # ax_forecasted.set_xticklabels(tick_positions_forecasted.strftime('%b %d, %Y'))
            # plt.xticks(rotation=45)

            # buffer_forecasted = BytesIO()
            # plt.savefig(buffer_forecasted, format="png")
            # buffer_forecasted.seek(0)
            # plt.close(fig_forecasted)

            # image_html_forecasted, image_base64_forecasted = display_graph(buffer_forecasted)

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
                "\n\nForecasted Values Summary:\n" +
                "\n".join(
                    f"- {summary['Company Name']}:\n"
                    f"  Min: {summary['Min']:.2f}, Max: {summary['Max']:.2f}, "
                    f"Mean: {summary['Mean']:.2f}, Trend: {summary['Trend']}"
                    for summary in forecasted_values_summary
                ) +
                "\n\nPlease provide an analysis comparing the trends, actual values, predictions and forecasted values for these companies, "
                "highlight which companies seem to be performing better in the sector, and suggest any insights about forecast."
            )

            # Generate textual analysis
            generated_sector_text = model.generate_content(formatted_sector_input).text

            return {"text": generated_sector_text, "graphs": [fig_actual, fig_predicted, fig_forecasted] if fig_actual and fig_predicted else []}

        elif intent_data["intent"] == "graph":
            company_name = extract_company_name(user_input).upper()
            ticker = resolve_ticker(company_name, ticker_mapping) + ".TA"
            best_model_df = pd.read_csv(BEST_MODEL_CSV)
            stocks_model = best_model_df.loc[best_model_df["Company"] == ticker, "Model"].values[0]

            start_date, end_date = datetime(2024, 1, 1), datetime.today()
            data = extract_data_by_model(ticker, stocks_model, start_date, end_date)

            fig, forecasted = generate_graph_with_forecast(ticker, company_name, stocks_model)

            # # Generate the graph
            # fig = go.Figure()
            # fig.add_trace(
            #     go.Scatter(
            #         x=data["Date"],
            #         y=data["Actual"],
            #         mode='lines',
            #         name="Actual",
            #         line=dict(dash='solid')
            #     )
            # )
            # fig.add_trace(
            #     go.Scatter(
            #         x=data["Date"],
            #         y=data["Predicted"],
            #         mode='lines',
            #         name="Predicted",
            #         line=dict(dash='dash')
            #     )
            # )

            # fig.update_layout(
            #     title=f"Actual and Predicted Values for {company_name} ({stocks_model})",
            #     xaxis_title="Date",
            #     yaxis_title="Value",
            #     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            #     height=600,
            #     width=1000
            # )
            # fig.update_xaxes(tickformat='%Y-%m', tickangle=45)


            # Return graph with text
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
                },
                "Forecasted Values Summary": {
                    "Min": forecasted["Forecasted"].min(),
                    "Max": forecasted["Forecasted"].max(),
                    "Mean": forecasted["Forecasted"].mean(),
                    "Trend": "upward" if forecasted["Forecasted"].iloc[-1] > forecasted["Forecasted"].iloc[0] else "downward"
                }
            }

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
                 f"Forecasted Values Summary:\n"
                f"  - Min: {summary_data['Forecasted Values Summary']['Min']:.2f}\n"
                f"  - Max: {summary_data['Forecasted Values Summary']['Max']:.2f}\n"
                f"  - Mean: {summary_data['Forecasted Values Summary']['Mean']:.2f}\n"
                f"  - Trend: {summary_data['Forecasted Values Summary']['Trend']}\n\n"
                "Please provide an analysis discussing the trends, actual values, predictions and forecast values for this company, "
                "with any observations about the stock's performance based on the provided data."
            )

            generated_text = model.generate_content(formatted_input).text
            return {"text": generated_text, "graphs": [fig]}

        elif intent_data["intent"] == "compare":
            company_names = extract_company_names(user_input)
            

            tickers = [resolve_ticker(name.upper(), ticker_mapping) + ".TA" for name in company_names]
            print(tickers)

            # Reverse the mapping to get company names from tickers
            ticker_to_company_name = {v: k for k, v in ticker_mapping.items()}

            # Define date range
            start_date, end_date = datetime(2024, 1, 1), datetime.today()

            # Load the best model mapping
            best_model_df = pd.read_csv(BEST_MODEL_CSV)

            # Container for data
            actual_predicted_data_frames = []
            combined_data_frames = []
            comparison_summary = []
            fig_comparison = go.Figure()
            for ticker in tickers:
                try:
                    # Get the best model for the company
                    stocks_model = best_model_df.loc[best_model_df["Company"] == ticker, "Model"].values[0]

                    # Extract actual and predicted data
                    actual_predicted = extract_data_by_model(ticker, stocks_model, start_date, end_date)
                    if not actual_predicted.empty:
                        # Extract forecasted data
                        last_predicted_date = actual_predicted["Date"].max()
                        if isinstance(last_predicted_date, str):
                            last_predicted_date = pd.to_datetime(last_predicted_date)
                        if last_predicted_date not in actual_predicted["Date"].values:
                            st.warning(f"No data available for last predicted date ({last_predicted_date}) for ticker {ticker}. Skipping.")
                            continue

                        # Retrieve the last predicted value
                        last_predicted_value = actual_predicted.loc[actual_predicted["Date"] == last_predicted_date, "Predicted"].values[0]
                        forecasted = extract_forecasted_values(ticker, last_predicted_date, actual_predicted)
                        

                        # Align forecasted dates
                        forecasted["Date"] = pd.date_range(
                            start=last_predicted_date + timedelta(days=1),
                            periods=len(forecasted),
                            freq="D"
                        )
                        # Combine actual, predicted, and forecasted data
                        combined_data = pd.concat(
                            [
                                actual_predicted[["Date", "Actual", "Predicted"]],
                                forecasted.rename(columns={"Forecast": "Forecasted"})[["Date", "Forecasted"]],
                            ],
                            ignore_index=True,
                        ).sort_values(by="Date")

                        # Add data to plotting containers
                        actual_predicted_data_frames.append((ticker, stocks_model, actual_predicted))
                        combined_data_frames.append((ticker, stocks_model, combined_data))

                        # Prepare summary data
                        company_name = ticker_to_company_name.get(ticker.replace(".TA", ""), ticker)
                        company_summary = {
                            "Company Name": company_name,
                            "Model Used": stocks_model,
                            "Actual Values Summary": {
                                "Min": actual_predicted["Actual"].min(),
                                "Max": actual_predicted["Actual"].max(),
                                "Mean": actual_predicted["Actual"].mean(),
                                "Trend": "upward" if actual_predicted["Actual"].iloc[-1] > actual_predicted["Actual"].iloc[0] else "downward"
                            },
                            "Predicted Values Summary": {
                                "Min": actual_predicted["Predicted"].min(),
                                "Max": actual_predicted["Predicted"].max(),
                                "Mean": actual_predicted["Predicted"].mean(),
                                "Trend": "upward" if actual_predicted["Predicted"].iloc[-1] > actual_predicted["Predicted"].iloc[0] else "downward"
                            },
                            "Forecasted Values Summary": {
                                "Min": forecasted["Forecasted"].min(),
                                "Max": forecasted["Forecasted"].max(),
                                "Mean": forecasted["Forecasted"].mean(),
                                "Trend": "upward" if forecasted["Forecasted"].iloc[-1] > forecasted["Forecasted"].iloc[0] else "downward"
                            }
                        }
                        comparison_summary.append(company_summary)
                except Exception as e:
                    # st.error(f"Error processing {ticker}: {e}")
                    return None
                

            for ticker, stocks_model, data in combined_data_frames:
                company_name = ticker_to_company_name.get(ticker.replace(".TA", ""), ticker)
                # Add to comparison graph
                fig_comparison.add_trace(
                    go.Scatter(
                        x=data["Date"],
                        y=data["Actual"],
                        mode='lines',
                        name=f"{company_name} Actual ({stocks_model})",
                        line=dict(dash='solid')
                    )
                )
                fig_comparison.add_trace(
                    go.Scatter(
                        x=data["Date"],
                        y=data["Predicted"],
                        mode='lines',
                        name=f"{company_name} Predicted ({stocks_model})",
                        line=dict(dash='dash')
                    )
                )
                fig_comparison.add_trace(
                    go.Scatter(
                        x=data["Date"],
                        y=data["Forecasted"],
                        mode='lines',
                        name=f"{company_name} Forecasted ({stocks_model})",
                        line=dict(dash='dash')
                    )
                )

            if not combined_data_frames:
                return {"text": "No data available for the selected companies."}

            fig_comparison.update_layout(
                title="Stock Price Comparison",
                xaxis_title="Date",
                yaxis_title="Price",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=600,
                width=1000
            )
            fig_comparison.update_xaxes(tickformat='%Y-%m', tickangle=45)


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
                    f"  - Mean: {summary['Predicted Values Summary']['Mean']:.2f}\n\n"
                    f"Forecasted Values Summary:\n"
                    f"  - Min: {summary['Forecasted Values Summary']['Min']:.2f}\n"
                    f"  - Max: {summary['Forecasted Values Summary']['Max']:.2f}\n"
                    f"  - Mean: {summary['Forecasted Values Summary']['Mean']:.2f}\n"
                    f"  - Trend: {summary['Forecasted Values Summary']['Trend']}\n"
                    for summary in comparison_summary
                )
                + "\n\nPlease provide an analysis comparing the trends, values, and predictions for these companies, "
                "highlight which company seems better based on actual, predicted, and forecasted values, and forecasted values compared to actual values"
                "and include any interesting observations about their performance or expected future behavior."
            )

            generated_comparison_text = model.generate_content(formatted_comparison_input).text
            return {"text": generated_comparison_text, "graphs": [fig_comparison]}

        else:
            return {"text": model.generate_content(user_input + 'add a sad emoji wherever you find fitting. include it without saying you did, just respobd normaly and add it as well.').text}
    except Exception as e:
            resp = emoji.emojize("Oh no! An exception! :face_screaming_in_fear: Please try again!")
            response = {"text" : resp}
            return response


    


def customize_plotly_theme(fig):
    """
    Apply custom theming to Plotly figures.
    """
    fig.update_layout(
        plot_bgcolor="#F9F6E6",
        font=dict(family="Arial, sans-serif", size=15, color="#333333"),
        colorway=["#001219","#005f73","#0a9396","#94d2bd","#ee9b00","#ca6702","#bb3e03","#9b2226"]
    )
    fig.update_layout(
    hoverlabel=dict(
        font_size=16  # Change hover text font size
    )
    )

    # Customize axis tick font size
    fig.update_layout(
        xaxis=dict(
            tickfont=dict(
                size=14  # X-axis tick font size
            )
        ),
        yaxis=dict(
            tickfont=dict(
                size=14  # Y-axis tick font size
            )
        )
    )
    fig.update_xaxes(tickformat='%Y-%m', tickangle=45, showgrid=True, gridcolor="#B3C8CF")
    fig.update_yaxes(showgrid=True, gridcolor="#B3C8CF")
    return fig

from copy import deepcopy

def text_streamer(text, delay=0.03):
    for word in text.split(" "):
        yield word + " "
        time.sleep(delay)

def main():
    st.set_page_config(page_title="Robo Advisor", layout="wide")
    st.title("Robo Advisor")

    st.sidebar.title("About")
    st.sidebar.info("This is a chatbot app that provides stock insights.")

    model = initialize_genai()
    ticker_mapping = load_ticker_mapping()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history without duplication
    for i, message in enumerate(st.session_state.chat_history):
        with st.chat_message(message["role"]):

            if "graphs" in message and message["graphs"]:
                for j, graph in enumerate(message["graphs"]):
                    # Unique key ensures Streamlit handles elements independently
                    st.plotly_chart(customize_plotly_theme(graph), use_container_width=True, key=f"history_graph_{i}_{j}")
            if "content" in message and message["content"]:
                st.markdown(message["content"])
            else:
                # Handle unexpected cases gracefully
                st.markdown("An unknown message format was encountered.")
    # Handle user input
    if prompt := st.chat_input("What would you like to know?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Get response from chatbot
        response = chatbot_response(prompt, model, ticker_mapping)

        with st.chat_message("assistant"):
            if "graphs" in response and response["graphs"]:
                for j, graph in enumerate(response["graphs"]):
                    # Use unique keys for new graphs
                    st.plotly_chart(customize_plotly_theme(graph), use_container_width=True, key=f"response_graph_{len(st.session_state.chat_history)}_{j}")
                    time.sleep(1)
            if "text" in response:
                st.write_stream(text_streamer(response["text"]))

            # Append response to session history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response.get("text", ""),
                "graphs": [deepcopy(customize_plotly_theme(graph)) for graph in response.get("graphs", [])]  # Save a deep copy of graphs
            })

               



if __name__ == "__main__":
    main()
