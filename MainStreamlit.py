import streamlit as st
from datetime import datetime
import os
import pandas as pd
from GraphDrawer import GraphDrawer

import google.generativeai as genai
API_KEY = 'AIzaSyCTNncuxKui7XIzrZWt1o_EtLIxiew8qtE'
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-8b",
    system_instruction="you are a chat bot designed to help with stock"
                       " analysis and recommendation over the israeli (TA) stock market."
                       "you will do function calling and textual answers."
                       "please be nice and polite.")


# Initialize the app
st.set_page_config(page_title="Chatbot", layout="wide")


# Load the mapping file
mapping_file_path = "/workspaces/FinalProj/company_name_to_ticker.xlsx"
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


# Chatbot logic
def chatbot_response(user_input):
    try:
        if "graph" in user_input.lower():
            ticker = resolve_ticker(user_input)
            st.write(f"Resolved Ticker: {ticker}")
            # Dynamic date range (adjust as needed)
            start_date = datetime(2021, 1, 1)
            end_date = datetime(2021, 12, 31)
            graph_drawer = GraphDrawer(ticker, start_date, end_date)
            graph_drawer.fetch_stock_data()
            graph_drawer.show_data_table()
            graph_drawer.plot_stock_data()
        else:
            return model.generate_content(user_input)
    except ValueError as e:
        st.error(str(e))



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
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        response = chatbot_response(prompt).text
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})




if __name__ == "__main__":
    main()
