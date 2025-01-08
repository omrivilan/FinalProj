import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import streamlit as st
import io


class GraphDrawer:
    def __init__(self, ticker, start_date, end_date):
        """
        Initialize the GraphDrawer class with stock ticker and date range.

        :param ticker: Stock ticker symbol (string)
        :param start_date: Start date for fetching stock data (datetime.date)
        :param end_date: End date for fetching stock data (datetime.date)
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None

    def fetch_stock_data(self):
        """
        Fetch stock data using yfinance for the given ticker and date range.
        """
        try:
            stock = yf.Ticker(self.ticker)
            self.data = stock.history(start=self.start_date, end=self.end_date)
        except Exception as e:
            st.error(f"Error fetching data for {self.ticker}: {e}")

    def plot_stock_data(self):
        """
        Plot the stock data if available, otherwise display an error.
        """
        if self.data is None or self.data.empty:
            st.error(f"No data available for ticker: {self.ticker}")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(self.data.index, self.data['Close'], label='Close Price')
        plt.title(f"Stock Price of {self.ticker}")
        plt.xlabel("Date")
        plt.ylabel("Price (in USD)")
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

        # Save plot to an in-memory buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)  # Reset buffer to the beginning
        plt.close()  # Close the plot to free up memory
    
        return buffer


    def show_data_table(self):
        """
        Display the fetched stock data as a table in Streamlit.
        """
        if self.data is not None and not self.data.empty:
            st.write(f"### Stock Price Data for {self.ticker}")
            st.dataframe(self.data.tail())
        else:
            st.error(f"No data available for ticker: {self.ticker}")
