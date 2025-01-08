import streamlit as st
from datetime import datetime
from GraphDrawer import GraphDrawer
import google.generativeai as genai

import os
import pandas as pd
import base64


API_KEY = 'AIzaSyCTNncuxKui7XIzrZWt1o_EtLIxiew8qtE'
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-8b",
    system_instruction="you are a chat bot designed to help with stock"
                       " analysis and recommendation over the israeli (TA) stock market."
                       "you will do function calling and textual answers."
                       "when you return a graph, add analysis and recommendation."
                       "please be nice and polite.")


# Initialize the app
st.set_page_config(page_title="Chatbot", layout="wide")

# Load the mapping file
mapping_file_path = "company_name_to_ticker.xlsx"
company_name_to_ticker = pd.read_excel(mapping_file_path)

# Convert mapping to a dictionary for fast lookup
ticker_mapping = {row["CompanyName"].upper(): row["Ticker"].upper() for _, row in company_name_to_ticker.iterrows()}

def resolve_ticker(user_input):
    user_input = user_input.strip().upper()
    # Check for direct match in mapping
    for company_name, ticker in ticker_mapping.items():
        if user_input in company_name or user_input == ticker:
            return ticker
    raise ValueError(f"Ticker or company name '{user_input}' not found in the mapping.")

def extract_company_name(user_input):
    user_input = user_input.lower()
    if "graph of" in user_input:
        return user_input.split("graph of")[1].strip()
    elif "graph" in user_input:
        return user_input.split("graph")[1].strip()
    raise ValueError("No recognizable company name in input.")
def extract_company_names(user_input):
    user_input = user_input.lower()
    if "compare" in user_input:
        # Remove unnecessary text between "compare" and "of"
        if "of" in user_input:
            compare_index = user_input.index("compare") + len("compare")
            of_index = user_input.index("of") + len("of")
            user_input = user_input[:compare_index] + user_input[of_index:]

        # Extract company names separated by "and" or "&"
        if "and" in user_input:
            companies = user_input.split("compare")[1].split("and")
        elif "&" in user_input:
            companies = user_input.split("compare")[1].split("&")
        else:
            raise ValueError("No recognizable company names to compare.")
        return [company.strip() for company in companies]
    raise ValueError("No recognizable comparison request in input.")


# Chatbot logic
def chatbot_response(user_input):
    try:
        if "compare" in user_input.lower():
            company_names = extract_company_names(user_input)
            tickers = [resolve_ticker(name.upper()) + ".TA" for name in company_names]
            st.write(f"Resolved Tickers: {', '.join(tickers)}")
            start_date = datetime(2024, 1, 1)
            end_date = datetime.today()

            # Fetch and plot data for both tickers
            data_frames = []
            for ticker in tickers:
                graph_drawer = GraphDrawer(ticker, start_date, end_date)
                graph_drawer.fetch_stock_data()
                data_frames.append(graph_drawer.data)

            # Plot data for comparison
            if len(data_frames) == 2:
                st.write("### Comparison Graph")
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 6))
                for i, df in enumerate(data_frames):
                    plt.plot(df.index, df["Close"], label=tickers[i])
                plt.title("Stock Price Comparison")
                plt.xlabel("Date")
                plt.ylabel("Close Price")
                plt.legend()
                st.pyplot(plt)

            return f"Comparison of stock prices for {', '.join(company_names)} displayed successfully."
        elif "graph" in user_input.lower():
            company_name = extract_company_name(user_input).upper()
            ticker = resolve_ticker(company_name) + ".TA"
            st.write(f"Resolved Ticker: {ticker}")
            start_date = datetime(2024, 1, 1)
            end_date = datetime.today()
            graph_drawer = GraphDrawer(ticker, start_date, end_date)
            graph_drawer.fetch_stock_data()
            graph_drawer.show_data_table()
            plot_buffer = graph_drawer.plot_stock_data()

            if plot_buffer:
                # Convert buffer to base64 string for embedding in session state
                image_base64 = base64.b64encode(plot_buffer.getvalue()).decode("utf-8")
                image_html = f'<img src="data:image/png;base64,{image_base64}" alt="Stock Plot">'

                # Append plot to chat history
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": f"Plot for {company_name}:", "image": image_html}
                )
                # Display the plot in the chat
                with st.chat_message("assistant"):
                    st.markdown(image_html, unsafe_allow_html=True)

            return f"Stock data and graph for {company_name} displayed successfully."
        else:
            response = model.generate_content(user_input).text
            return response
    except ValueError as e:
        st.error(str(e))
        return f"Error: {str(e)}"






# Main app
def main():
    st.title("Chatbot Application")
    st.sidebar.title("About")
    st.sidebar.info("This is a simple chatbot app built using Streamlit.")

    st.write("### Welcome to the Chatbot!")

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if "image" in message:
                # Render image content
                st.markdown(message["image"], unsafe_allow_html=True)
            else:
                # Render text content
                st.markdown(message["content"])


    # User input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        response = chatbot_response(prompt)
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response, })




if __name__ == "__main__":
    main()
