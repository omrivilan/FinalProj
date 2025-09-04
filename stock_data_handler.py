import pandas as pd
import numpy as np
import os
from datetime import timedelta
import ast

class StockDataHandler:
    """Handles loading, merging, and accessing stock data and mappings for analysis.

    This class provides methods to extract actual, predicted, and forecasted stock values
    by company, industry, or sector. It also manages ticker-symbol resolution and
    Hebrew-English name mappings to support bilingual applications.

    Attributes:
        ticker_mapping (dict): Mapping from company names to ticker symbols.
        model_paths (dict): Dictionary of model names to their corresponding CSV file paths.
        sector_path (str): Path to CSV mapping sectors to company tickers.
        best_model_path (str): Path to CSV specifying the best model per company.
        hebrew_name_mapping (dict): Maps English company names to their Hebrew equivalents.
        hebrew_industry_mapping (dict): Maps English industry names to Hebrew equivalents.
    """
    def __init__(self, ticker_mapping, model_paths: dict, sector_path: str, best_model_path: str):
        """Initializes the StockDataHandler with model paths and name mappings.
    
        This class is responsible for loading and managing stock data, including
        actuals, predictions, forecasts, and sector associations. It also handles
        mapping between English and Hebrew names for companies and industries.
    
        Args:
            ticker_mapping (dict): Mapping from company names to ticker symbols.
            model_paths (dict): Dictionary mapping model names to their CSV file paths.
            sector_path (str): Path to the CSV file containing sector-to-ticker mappings.
            best_model_path (str): Path to the CSV file containing the best model per company.
    
        Attributes:
            ticker_mapping (dict): Maps company names to tickers.
            model_paths (dict): Maps model names to file paths for their predictions.
            sector_path (str): CSV path for sector data.
            best_model_path (str): CSV path for best-performing model per company.
            hebrew_name_mapping (dict): English → Hebrew company name mapping.
            hebrew_industry_mapping (dict): English → Hebrew industry name mapping.
        """
        self.ticker_mapping = ticker_mapping
        self.model_paths = model_paths
        self.sector_path = sector_path
        self.best_model_path = best_model_path

        # Load Hebrew name mapping
        mapping_df = pd.read_excel("company_name_to_ticker.xlsx")
        self.hebrew_name_mapping = dict(zip(mapping_df["CompanyName"].str.upper(), mapping_df["HebrewCompanyName"]))

        sector_df = pd.read_csv("sectors_df.csv")
        self.hebrew_industry_mapping = dict(zip(sector_df["Industry"], sector_df["HebrewIndustryName"]))

    def extract_by_model(self, ticker: str, model: str, start_date, end_date):
        """Extracts actual and predicted stock data for a given ticker and model within a date range.
    
        This method routes the extraction to the appropriate processing logic based on the model type:
        - Neural networks (LSTM, GRU) are filtered using a date range.
        - Boosting models (XGBoost, LightGBM) are parsed from flattened row data.
    
        Args:
            ticker (str): Ticker symbol of the company.
            model (str): The name of the forecasting model ("LSTM", "GRU", "XGBoost", "LightGBM").
            start_date (datetime): Start of the date range.
            end_date (datetime): End of the date range.
    
        Returns:
            pd.DataFrame: A DataFrame containing actual and predicted values indexed by date.
    
        Raises:
            ValueError: If the model is unsupported or unrecognized.
        """
        if model not in self.model_paths:
            raise ValueError(f"Unsupported model: {model}")

        df = pd.read_csv(self.model_paths[model])

        if model in ["LSTM", "GRU"]:
            return self._filter_stock_data_nn(df, ticker, start_date, end_date)
        elif model in ["XGBoost", "LightGBM"]:
            company_data = df[df["Stock"] == ticker].iloc[0]
            return self._process_model_data_boosting(company_data, start_date)

        raise ValueError(f"Unhandled model: {model}")

    def _filter_stock_data_nn(self, df, ticker, start_date, end_date):
        """Filters LSTM/GRU stock data for a given ticker and date range.
    
        This method ensures the ticker is in the correct format and filters the
        DataFrame to include only the relevant rows for the specified time window.
    
        Args:
            df (pd.DataFrame): DataFrame containing LSTM/GRU output with "Date" and "Ticker" columns.
            ticker (str): The stock ticker to filter (with or without ".TA" suffix).
            start_date (datetime): Start date of the desired range.
            end_date (datetime): End date of the desired range.
    
        Returns:
            pd.DataFrame: Filtered DataFrame with rows matching the ticker and date range.
        """

        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
        ticker = ticker if ticker.endswith(".TA") else f"{ticker}.TA"
        return df[(df["Ticker"] == ticker) & (df["Date"] >= start_date) & (df["Date"] <= end_date)]

    def _process_model_data_boosting(self, company_data, start_date):
        """Processes XGBoost/LightGBM model output stored as serialized lists in a single row.
    
        This method deserializes the `y_test` and `y_pred` lists from a row of the DataFrame,
        aligns them to a fixed date range, pads them if needed, and returns a merged DataFrame
        of actual and predicted values.
    
        Args:
            company_data (pd.Series): A single row containing 'y_test' and 'y_pred' as stringified lists.
            start_date (datetime): (Unused) Placeholder to align with API of other extract methods.
    
        Returns:
            pd.DataFrame: A DataFrame with columns ['Date', 'Actual', 'Predicted'] spanning a fixed date range.
        """
        company_data["y_test"] = ast.literal_eval(company_data["y_test"])
        company_data["y_pred"] = ast.literal_eval(company_data["y_pred"])

        fixed_start = pd.to_datetime("2024-07-04")
        fixed_end = pd.to_datetime("2024-12-30")
        total_days = (fixed_end - fixed_start).days + 1

        y_test = company_data["y_test"][:total_days]
        y_pred = company_data["y_pred"][:total_days]

        if len(y_test) < total_days:
            y_test += [y_test[-1]] * (total_days - len(y_test))
        if len(y_pred) < total_days:
            y_pred += [y_pred[-1]] * (total_days - len(y_pred))

        date_range = pd.date_range(start=fixed_start, end=fixed_end)

        actual_df = pd.DataFrame({"Date": date_range, "Actual": y_test})
        predicted_df = pd.DataFrame({"Date": date_range, "Predicted": y_pred})

        return pd.merge(actual_df, predicted_df, on="Date")

    def extract_by_industry(self, industry, start_date, end_date):
        """Extracts actual, predicted, and forecasted data for all companies within a given industry.
        Used in 'Sector Comparison' intent.
        This method identifies all tickers associated with the specified industry, determines each
        company's best model, and collects data over the specified date range. It merges actual/predicted
        values with forecasted values for each company.
    
        Args:
            industry (str): The name of the industry to query.
            start_date (datetime): Start of the analysis window.
            end_date (datetime): End of the analysis window.
    
        Returns:
            pd.DataFrame: A merged DataFrame containing actual, predicted, and forecasted values
                          for all companies in the industry. Columns include "Date", "Actual",
                          "Predicted", "Forecasted", "Ticker", and "Industry".
        """
        df = pd.read_csv(self.sector_path)
        best_models = pd.read_csv(self.best_model_path)
        reverse_map = {v: k for k, v in self.ticker_mapping.items()}

        df.rename(columns={"Symbol": "Ticker"}, inplace=True)
        industry_companies = df[df["Industry"].str.lower() == industry.lower()]
        tickers = industry_companies["Ticker"].tolist()

        result = []

        for t in tickers:
            full_ticker = f"{t}.TA"
            try:
                model = best_models.loc[best_models["Company"] == full_ticker, "Model"].values[0]
                ap_df = self.extract_by_model(full_ticker, model, start_date, end_date)
                if ap_df.empty:
                    continue

                last_date = ap_df["Date"].max()
                forecast_df = self.extract_forecasted_values(full_ticker, last_date, ap_df)

                merged = pd.merge(ap_df, forecast_df, on="Date", how="outer")
                merged["Ticker"] = full_ticker
                merged["Industry"] = industry
                result.append(merged)
            except Exception:
                continue

        return pd.concat(result, ignore_index=True) if result else pd.DataFrame()

    def get_industry_actuals(self, industry, start_date, end_date):
        """Retrieves actual stock values for all companies in a specified industry over a given date range.
        Used in 'industry_values' intent.
        For each company in the industry, this method determines its best model, extracts the actual
        values using that model, and returns a combined DataFrame with ticker, company, and Hebrew name.
    
        Args:
            industry (str): The name of the industry to query.
            start_date (datetime): Start date of the desired time range.
            end_date (datetime): End date of the desired time range.
    
        Returns:
            pd.DataFrame: A DataFrame with columns ["Ticker", "Date", "Actual", "Company", "HebrewCompanyName"]
                          containing actual values for each company in the industry.
        """
        df = pd.read_csv(self.sector_path)
        df.rename(columns={"Symbol": "Ticker"}, inplace=True)
        industry_companies = df[df["Industry"].str.lower() == industry.lower()]
        tickers = industry_companies["Ticker"].tolist()
        best_models = pd.read_csv(self.best_model_path)
        reverse_map = {v: k for k, v in self.ticker_mapping.items()}
        result = []

        for t in tickers:
            full_ticker = f"{t}.TA"
            try:
                model = best_models.loc[best_models["Company"] == full_ticker, "Model"].values[0]
                data = self.extract_by_model(full_ticker, model, start_date, end_date)
                if not data.empty:
                    company_name = reverse_map.get(t, t)
                    hebrew_name = self.hebrew_name_mapping.get(t, company_name)
                    result.append(pd.DataFrame({
                        "Ticker": full_ticker,
                        "Date": data["Date"],
                        "Actual": data["Actual"],
                        "Company": company_name,
                        "HebrewCompanyName": hebrew_name
                    }))
            except Exception:
                continue

        return pd.concat(result, ignore_index=True) if result else pd.DataFrame()

    def get_industry_predictions(self, industry, start_date, end_date):
        """Retrieves predicted stock values for all companies in a specified industry over a given date range.
        Used in 'industry_values' intent.
        This method identifies all tickers within the given industry, determines the best model per company,
        extracts predicted values, and formats the results with company metadata for visualization or analysis.
    
        Args:
            industry (str): The industry to retrieve predictions for.
            start_date (datetime): Start date of the prediction window.
            end_date (datetime): End date of the prediction window.
    
        Returns:
            pd.DataFrame: A DataFrame with columns ["Ticker", "Date", "Predicted", "Company", "HebrewCompanyName"]
                          containing predicted stock values for each company in the industry.
        """

        df = pd.read_csv(self.sector_path)
        best_models = pd.read_csv(self.best_model_path)

        df.rename(columns={"Symbol": "Ticker"}, inplace=True)
        reverse_map = {v: k for k, v in self.ticker_mapping.items()}

        industry_companies = df[df["Industry"].str.lower() == industry.lower()]
        tickers = industry_companies["Ticker"].tolist()

        result = []

        for t in tickers:
            full_ticker = f"{t}.TA"
            try:
                model = best_models.loc[best_models["Company"] == full_ticker, "Model"].values[0]
                data = self.extract_by_model(full_ticker, model, start_date, end_date)
                if not data.empty:
                    company_name = reverse_map.get(t, t)
                    hebrew_name = self.hebrew_name_mapping.get(t, company_name)
                    result.append(pd.DataFrame({
                        "Ticker": full_ticker,
                        "Date": data["Date"],
                        "Predicted": data["Predicted"],
                        "Company": company_name,
                        "HebrewCompanyName": hebrew_name
                    }))
            except Exception:
                continue

        return pd.concat(result, ignore_index=True) if result else pd.DataFrame()

    def extract_forecasted_values(self, stock, last_predicted_date, actual_predicted: pd.DataFrame):
        """Extracts and aligns forecasted values for a stock using its best-performing model.
    
        This method reads model-specific forecast files, extracts the relevant forecasted values
        for the given stock, and aligns them chronologically starting from the day after the last
        predicted date. It also applies an adjustment to smoothly transition from predicted to
        forecasted values.
    
        Args:
            stock (str): The full ticker symbol of the stock (e.g., "TEVA.TA").
            last_predicted_date (datetime): The last date of the available predicted values.
            actual_predicted (pd.DataFrame): DataFrame containing at least "Date" and "Predicted" columns.
    
        Returns:
            pd.DataFrame: A DataFrame with columns ["Date", "Forecasted"], representing the adjusted
                          future forecasted values for the stock.
    
        Raises:
            ValueError: If the forecast file or data is missing for the specified model or stock.
        """

        forecast_paths = {
            "LSTM": os.path.join("LSTM", "forecast_lstm_without_reports.csv"),
            "GRU": os.path.join("GRU", "forecast_gru_without_reports.csv"),
            "LightGBM": os.path.join("LightGBM", "model_LightGBM_metrics_and_predictions_total.csv"),
            "XGBoost": os.path.join("XGBoost", "model_XGBoost_metrics_and_predictions_without_report_parameters.csv")
        }
        best_model_df = pd.read_csv(self.best_model_path, encoding='latin-1')
        model = best_model_df.loc[best_model_df["Company"] == stock, "Model"].values[0]
        path = forecast_paths.get(model)

        if not path:
            raise ValueError(f"No forecast file for model: {model}")

        df = pd.read_csv(path, encoding='latin-1')
        if model in ["LSTM", "GRU"]:
            df = df[df["Ticker"] == stock][["Date", "Forecast"]]
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
        else:
            df = df[df["Stock"] == stock]
            if df.empty:
                raise ValueError(f"No forecast data found for stock: {stock}")
            forecast_cols = [col for col in df.columns if col.startswith("Day_")]
            df_melted = df.melt(value_vars=forecast_cols, value_name="Forecast", var_name="Day")
            df_melted["Day"] = df_melted["Day"].str.extract(r"Day_(\d+)").astype(int)
            df_melted["Date"] = [last_predicted_date + timedelta(days=int(x)) for x in df_melted["Day"]]
            df = df_melted[["Date", "Forecast"]].sort_values("Date").reset_index(drop=True)

        last_pred = actual_predicted.loc[actual_predicted["Date"] == last_predicted_date, "Predicted"].values[0]
        adjustment = last_pred - df["Forecast"].iloc[0]
        df["Forecasted"] = df["Forecast"] + adjustment
        df.drop(columns="Forecast", inplace=True)
        return df

    @staticmethod
    def synchronize_x_axis(dataframes: list) -> list:
        if not dataframes:
            return []
        all_dates = pd.concat([df['Date'] for df in dataframes])
        unified_range = pd.date_range(start=all_dates.min(), end=all_dates.max())
        synchronized = []
        for df in dataframes:
            df = df.set_index('Date').reindex(unified_range).reset_index()
            df.rename(columns={'index': 'Date'}, inplace=True)
            df.fillna(method='ffill', inplace=True)
            synchronized.append(df)
        return synchronized
