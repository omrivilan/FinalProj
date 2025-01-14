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
LIGHTGBM_CSV_PATH = "/workspaces/FinalProj/LightGBM/metrics_and_predictions.csv"
BEST_MODEL_CSV = "/workspaces/FinalProj/Metrics/best_model_per_stock.csv"
SECTORS_DF_PATH = "sectors_df.csv"
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
    sector_keywords = ["actual values", "sector", "companies in sector", "all"]
    if any(keyword in user_input for keyword in sector_keywords):
        sector_name = user_input.split("sector")[1].strip() if "sector" in user_input else None
        if sector_name:
            return {"intent": "sector_values", "sector": sector_name}
        raise ValueError("Please specify the sector name for the query.")
    # Expanded keywords for comparison requests
    compare_keywords = ["compare", "difference between", "versus", "vs", ",","&","and","And","AND"]
    if any(keyword in user_input for keyword in compare_keywords):
        return {"intent": "compare", "companies": extract_company_names(user_input)}
    graph_keywords = ["graph", "plot", "visualize", "show", "chart"]
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





# Chatbot Response Logic
def chatbot_response(user_input, model, ticker_mapping):
    try:
        intent_data = detect_intent(user_input)
        if intent_data["intent"] == "sector_values":
            sector_name = intent_data["sector"]
            start_date, end_date = datetime(2024, 1, 1), datetime.today()
            sector_data_actual = get_actual_values_by_sector(sector_name, start_date=start_date, end_date=end_date)

            # Reverse the mapping to get company names from tickers
            ticker_to_company_name = {v: k for k, v in ticker_mapping.items()}

            if sector_data_actual is not None and not sector_data_actual.empty:
                # Generate Actual Values Graph
                fig_actual, ax_actual = plt.subplots(figsize=(12, 8))
                for ticker, data in sector_data_actual.groupby("Ticker"):
                    company_name = ticker_to_company_name.get(ticker.replace(".TA", ""), ticker)
                    ax_actual.plot(pd.to_datetime(data["Date"]), data["Actual"], label=company_name)

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

            # Generate Predicted Values Graph
            sector_data_predicted = get_predicted_values_by_sector(sector_name, start_date=start_date, end_date=end_date)

            if sector_data_predicted is not None and not sector_data_predicted.empty:
                fig_predicted, ax_predicted = plt.subplots(figsize=(12, 8))
                for ticker, data in sector_data_predicted.groupby("Ticker"):
                    company_name = ticker_to_company_name.get(ticker.replace(".TA", ""), ticker)
                    ax_predicted.plot(pd.to_datetime(data["Date"]), data["Predicted"], label=company_name)

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

            # Return both graphs
            return {
                "actual_html": image_html_actual,
                "actual_image": image_base64_actual,
                "predicted_html": image_html_predicted,
                "predicted_image": image_base64_predicted,
            }


        if intent_data["intent"] == "graph":
            company_name = extract_company_name(user_input).upper()
            ticker = resolve_ticker(company_name, ticker_mapping) + ".TA"
            stocks_model = pd.read_csv(BEST_MODEL_CSV)["Model"].values[0]

            start_date, end_date = datetime(2024, 1, 1), datetime.today()
            data = extract_data_by_model(ticker, stocks_model, start_date, end_date)

            # Generate the graph
            buffer = generate_graph(data, f"Stock Data for {company_name}")
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
            elif "actual_image" in response and "predicted_image" in response:
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
