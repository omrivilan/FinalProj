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

# Global Configuration
API_KEY = 'AIzaSyCTNncuxKui7XIzrZWt1o_EtLIxiew8qtE'
MAPPING_FILE_PATH = "company_name_to_ticker.xlsx"
LSTM_CSV_PATH = "/workspaces/FinalProj/LSTM/actual_vs_pred_stocks.csv"
XGBOOST_CSV_PATH = "/workspaces/FinalProj/XGBoost/model_XGBoost_metrics_and_predictions.csv"
LIGHTGBM_CSV_PATH = "/workspaces/FinalProj/LightGBM/metrics_and_predictions.csv"
BEST_MODEL_CSV = "/workspaces/FinalProj/Metrics/best_model_per_stock.csv"

# Initialize API and Streamlit

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
    delimiters = ["compare", "difference between", "versus", "vs", ",","&","and","And","AND"]
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
    graph_keywords = ["graph", "plot", "visualize", "show", "chart"]
    if any(keyword in user_input for keyword in graph_keywords):
        if "compare" in user_input or ("and" in user_input or "," in user_input):
            return {"intent": "compare", "companies": extract_company_names(user_input)}
        else:
            return {"intent": "graph", "company": extract_company_name(user_input)}

    # Expanded keywords for comparison requests
    compare_keywords = ["compare", "difference between", "versus", "vs", ",","&","and","And","AND"]
    if any(keyword in user_input for keyword in compare_keywords):
        return {"intent": "compare", "companies": extract_company_names(user_input)}

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

# Generate and Display Graph
def generate_graph(data, title):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(data["Date"], data["Actual"], label="Actual", linestyle="-", color="blue")
    ax.plot(data["Date"], data["Predicted"], label="Predicted", linestyle="--", color="orange")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close(fig)
    return buffer

def display_graph(buffer):
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    image_html = f'<img src="data:image/png;base64,{image_base64}" alt="Stock Plot">'
    return image_html,image_base64

# Chatbot Response Logic
def chatbot_response(user_input, model, ticker_mapping):
    try:
        if "graph" in user_input.lower():
            company_name = extract_company_name(user_input).upper()
            ticker = resolve_ticker(company_name, ticker_mapping) + ".TA"
            stocks_model = pd.read_csv(BEST_MODEL_CSV)["Model"].values[0]

            start_date, end_date = datetime(2024, 1, 1), datetime.today()
            data = extract_data_by_model(ticker, stocks_model, start_date, end_date)

            buffer = generate_graph(data, f"Stock Data for {company_name}")
            image_html, image_base64 = display_graph(buffer)
            return {"html": image_html, "image": image_base64}

        elif "compare" in user_input.lower():
            company_names = extract_company_names(user_input)
            tickers = [resolve_ticker(name.upper(), ticker_mapping) + ".TA" for name in company_names]

            # Define date range
            start_date, end_date = datetime(2024, 1, 1), datetime.today()

            # Load the best model mapping
            best_model_df = pd.read_csv(BEST_MODEL_CSV)

            # Container for data
            data_frames = []

            for ticker in tickers:
                try:
                    # Get the best model for the company
                    stocks_model = best_model_df.loc[best_model_df["Company"] == ticker, "Model"].values[0]

                    # Extract the data
                    data = extract_data_by_model(ticker, stocks_model, start_date, end_date)
                    if not data.empty:
                        data_frames.append((ticker, stocks_model, data))
                    else:
                        st.warning(f"No data available for ticker {ticker}. Skipping.")
                except Exception as e:
                    st.error(f"Error processing {ticker}: {e}")

            # Check if data exists
            if not data_frames:
                return "No data available for the selected tickers."

            # Plot comparison graph
            fig, ax = plt.subplots(figsize=(12, 8))
            for ticker, stocks_model, data in data_frames:
                ax.plot(
                    pd.to_datetime(data["Date"]),
                    data["Actual"],
                    label=f"{ticker} Actual ({stocks_model})",
                    linestyle="-"
                )
                ax.plot(
                    pd.to_datetime(data["Date"]),
                    data["Predicted"],
                    label=f"{ticker} Predicted ({stocks_model})",
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
            return {"html": image_html, "image": image_base64}



        else:
            return {"text": model.generate_content(user_input).text}

    except Exception as e:
        return {"text": f"Error: {str(e)}"}

# Main Streamlit App
def main():
    st.set_page_config(page_title="Chatbot", layout="wide")
    st.title("Chatbot Application")

    st.sidebar.title("About")
    st.sidebar.info("This is a simple chatbot app built using Streamlit.")

    model = initialize_genai()
    ticker_mapping = load_ticker_mapping()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if "image" in message:
                image_html = f'<img src="data:image/png;base64,{message["image"]}" alt="Stock Plot">'
                st.markdown(image_html, unsafe_allow_html=True)
            else:
                st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        response = chatbot_response(prompt, model, ticker_mapping)
        with st.chat_message("assistant"):
            if "image" in response:
                image_html = f'<img src="data:image/png;base64,{response["image"]}" alt="Stock Plot">'
                st.markdown(image_html, unsafe_allow_html=True)
                st.session_state.chat_history.append({"role": "assistant", "image": response["image"]})
            else:
                st.markdown(response["text"])
                st.session_state.chat_history.append({"role": "assistant", "content": response["text"]})


if __name__ == "__main__":
    main()
