import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import streamlit as st
import io
import pandas as pd


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

    def plot_30_day_forecast(self, predictions_file):
        """
        Plot the 30-day predictions for the chosen stock based on the CSV file.

        :param predictions_file: Path to the CSV file containing predictions.
        """
        try:
            # Load the CSV file
            forecasts = pd.read_csv(predictions_file)

            # Filter rows for the chosen stock
            stock_predictions = forecasts[forecasts['Ticker'] == self.ticker]

            if stock_predictions.empty:
                st.error(f"No predictions found for ticker: {self.ticker}")
                return

            # Convert Date column to datetime and sort
            stock_predictions['Date'] = pd.to_datetime(stock_predictions['Date'], dayfirst=True)
            stock_predictions = stock_predictions.sort_values('Date')

            # Plot predictions
            plt.figure(figsize=(10, 5))
            plt.plot(stock_predictions['Date'], stock_predictions['Forecast'], marker='o', label="Predicted Price")
            plt.title(f"30-Day Forecasts for {self.ticker}")
            plt.xlabel("Date")
            plt.ylabel("Forecasted Price (in NIS)")
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.legend()
            st.pyplot(plt)

        except Exception as e:
            st.error(f"Error processing predictions: {e}")

    def plot_forecast_vs_actual(self, predictions_file):
        """
        Plot the last 30-day predictions for a stock against its actual closing prices.

        :param predictions_file: Path to the CSV file containing predictions.
        """
        try:
            # Fetch actual stock data
            self.fetch_stock_data()

            if self.data is None or self.data.empty:
                st.error(f"No actual data available for ticker: {self.ticker}")
                return

            # Restrict to the last 30 days of actual data
            last_30_days = self.data.iloc[-30:]

            # Load predictions
            forecasts = pd.read_csv(predictions_file)

            # Filter rows for the chosen stock
            stock_predictions = forecasts[forecasts['Ticker'] == self.ticker]

            if stock_predictions.empty:
                st.error(f"No predictions found for ticker: {self.ticker}")
                return

            # Convert Date column to datetime and sort
            stock_predictions['Date'] = pd.to_datetime(stock_predictions['Date'], dayfirst=True)
            stock_predictions = stock_predictions.sort_values('Date')

            # Ensure predictions cover the last 30 days
            start_date = last_30_days.index.min().to_pydatetime().date()
            end_date = last_30_days.index.max().to_pydatetime().date()
            stock_predictions = stock_predictions[(stock_predictions['Date'].dt.date >= start_date) & (stock_predictions['Date'].dt.date <= end_date)]

            # Plot actual vs predicted
            plt.figure(figsize=(12, 6))
            plt.plot(last_30_days.index, last_30_days["Close"], label="Actual Price", color="blue")
            plt.plot(stock_predictions['Date'], stock_predictions['Forecast'], label="Forecasted Price", linestyle="--", color="red")

            plt.title(f"Last 30 Days: Actual vs Forecasted Prices for {self.ticker}")
            plt.xlabel("Date")
            plt.ylabel("Price (in NIS)")
            plt.grid(True)
            plt.legend()
            st.pyplot(plt)

        except Exception as e:
            st.error(f"Error processing predictions vs actual data: {e}")
