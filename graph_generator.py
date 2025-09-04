import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta
import plotly.express as px

class GraphGenerator:
    """Generates stock-related graphs with bilingual support for Israeli market companies.

    This class provides methods to create Plotly visualizations for individual companies,
    multiple companies and industries, including actual, predicted, and forecasted stock values.
    It supports both English and Hebrew labels, using a preloaded mapping from company names
    to their Hebrew equivalents.

    Attributes:
        company_name_to_hebrew (dict): Mapping from English company names to Hebrew names,
            used for generating localized graph titles and labels.
    """

    def __init__(self):
        """Initializes the GraphGenerator by loading company name mappings.
    
        This constructor reads the Excel file `company_name_to_ticker.xlsx` and builds a mapping
        from English company names (uppercase) to their Hebrew equivalents, used later for
        bilingual graph labeling.
    
        Attributes:
            company_name_to_hebrew (dict): Mapping from English uppercase company names to Hebrew names.
        """
        df = pd.read_excel("company_name_to_ticker.xlsx")
        self.company_name_to_hebrew = {
            row["CompanyName"].strip().upper(): row["HebrewCompanyName"].strip()
            for _, row in df.iterrows()
            if pd.notna(row["CompanyName"]) and pd.notna(row["HebrewCompanyName"])
        }

    def _translate_company_name(self, company_name: str) -> str:
        """Translates an English company name to its Hebrew equivalent if available.
    
        Args:
            company_name (str): The English name of the company.
    
        Returns:
            str: The Hebrew company name if found; otherwise, returns the original name.
        """
        key = company_name.strip().upper()
        return self.company_name_to_hebrew.get(key, company_name)

    def generate_actual_predicted_forecast_graph(self, df: pd.DataFrame, stock: str, model: str, language: str = "en"):
        """Generates a line graph of actual, predicted, and forecasted stock values for a single company.
    
        This method uses Plotly to create a time-series chart displaying three trend lines:
        actual values (solid), predicted values (dashed), and forecasted values (dotted). 
        Graph labels are automatically rendered in Hebrew or English.
    
        Args:
            df (pd.DataFrame): DataFrame containing columns "Date" and one or more of "Actual", "Predicted", "Forecasted".
            stock (str): The name of the stock or company to display in the title.
            model (str): The name of the model used for prediction/forecasting.
            language (str, optional): Language for graph labels ("en" or "he"). Defaults to "en".
    
        Returns:
            go.Figure: A Plotly figure visualizing the stock's actual, predicted, and forecasted values.
        """
        labels = {
            "en": {"title": f"Actual, Predicted, and Forecasted Values for {stock} ({model})", "x": "Date", "y": "Value"},
            "he": {"title": f"תחזית מניה עבור {self._translate_company_name(stock)} ({model})", "x": "תאריך", "y": "ערך"}
        }

        fig = go.Figure()

        if "Actual" in df.columns:
            fig.add_trace(go.Scatter(x=df["Date"], y=df["Actual"], mode='lines', name='Actual' if language == "en" else "ערך אמיתי", line=dict(color='blue', width=2)))

        if "Predicted" in df.columns:
            fig.add_trace(go.Scatter(x=df["Date"], y=df["Predicted"], mode='lines', name='Predicted' if language == "en" else "חזוי", line=dict(color='orange', width=2, dash='dash')))

        if "Forecasted" in df.columns:
            fig.add_trace(go.Scatter(x=df["Date"], y=df["Forecasted"], mode='lines', name='Forecasted' if language == "en" else "תחזית", line=dict(color='green', width=2, dash='dot')))

        fig.update_layout(
            title=labels[language]["title"],
            xaxis_title=labels[language]["x"],
            yaxis_title=labels[language]["y"],
            height=600,
            width=1000,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        fig.update_xaxes(tickformat='%Y-%m', tickangle=45, showgrid=True)
        fig.update_yaxes(showgrid=True)
        return self.customize(fig)

    def generate_industry_graph(self, df: pd.DataFrame, value_column: str, industry_name: str, language: str = "en"):
        """Generates a line graph showing stock trends for all companies within a given industry.
    
        This method plots time-series lines for each company in the industry using the specified value column 
        (e.g., "Actual", "Predicted", or "Forecasted"). Axis titles and graph labels are rendered in the selected language.
    
        Args:
            df (pd.DataFrame): DataFrame containing "Date", a value column, and either "Company" or "Ticker".
            value_column (str): The name of the column to plot ("Actual", "Predicted", or "Forecasted").
            industry_name (str): The name of the industry being visualized.
            language (str, optional): Language for axis and title labels ("en" or "he"). Defaults to "en".
    
        Returns:
            go.Figure: A Plotly figure showing the time-series values for each company in the industry.
        """
        title_map = {
            "en": f"{value_column} Values for {industry_name} Industry",
            "he": f"{value_column} עבור תחום {industry_name}"
        }
        x_label = "Date" if language == "en" else "תאריך"
        y_label = value_column if language == "en" else ("ערך אמיתי" if value_column == "Actual" else "חזוי" if value_column == "Predicted" else "תחזית")

        fig = go.Figure()
        for company, group in df.groupby("Company" if "Company" in df else "Ticker"):
            fig.add_trace(go.Scatter(x=pd.to_datetime(group["Date"], dayfirst=True), y=group[value_column], mode='lines', name=company))

        fig.update_layout(
            title=title_map[language],
            xaxis_title=x_label,
            yaxis_title=y_label,
            height=600,
            width=1000,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_xaxes(tickformat='%b %d, %Y', dtick=50 * 24 * 60 * 60 * 1000, tickangle=45)
        fig.update_yaxes(showgrid=True)
        return self.customize(fig)

    def generate_comparison_graph(self, dataframes: list, ticker_to_company_name: dict, language: str = "en"):
        """Generates a multi-line graph comparing actual, predicted, and forecasted values across companies.
    
        This method creates a Plotly figure with separate lines for each company's actual, predicted,
        and forecasted stock values. Each line is labeled with the company name and model used.
        The graph is rendered in English or Hebrew based on the selected language.
    
        Args:
            dataframes (list): A list of tuples (ticker, model, DataFrame), where each DataFrame includes
                "Date", "Actual", "Predicted", and "Forecasted" columns.
            ticker_to_company_name (dict): Mapping from ticker symbols to human-readable company names.
            language (str, optional): Language for labels and titles ("en" or "he"). Defaults to "en".
    
        Returns:
            go.Figure: A Plotly figure comparing stock values across multiple companies.
        """

        fig = go.Figure()
        for ticker, model, df in dataframes:
            name = ticker_to_company_name.get(ticker.replace(".TA", ""), ticker)
            fig.add_trace(go.Scatter(x=df["Date"], y=df["Actual"], mode='lines', name=f"{name} Actual ({model})" if language == "en" else f"{name} ערך אמיתי ({model})", line=dict(dash='solid')))
            fig.add_trace(go.Scatter(x=df["Date"], y=df["Predicted"], mode='lines', name=f"{name} Predicted ({model})" if language == "en" else f"{name} חזוי ({model})", line=dict(dash='dash')))
            fig.add_trace(go.Scatter(x=df["Date"], y=df["Forecasted"], mode='lines', name=f"{name} Forecasted ({model})" if language == "en" else f"{name} תחזית ({model})", line=dict(dash='dot')))

        fig.update_layout(
            title="Stock Price Comparison" if language == "en" else "השוואת מחירי מניות",
            xaxis_title="Date" if language == "en" else "תאריך",
            yaxis_title="Price" if language == "en" else "מחיר",
            height=600,
            width=1000,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_xaxes(tickformat='%Y-%m', tickangle=45)
        fig.update_yaxes(showgrid=True)
        return self.customize(fig)

    def generate_sector_comparison_graph(self, df: pd.DataFrame, label: str = "Actual", language: str = "en"):
        """Generates a line chart comparing average values across multiple sectors over time.
    
        This method uses Plotly Express to plot sector-level averages (e.g., actual, predicted, forecasted)
        with separate lines for each industry. Labels and titles are rendered in either English or Hebrew.
    
        Args:
            df (pd.DataFrame): DataFrame containing "Date", the specified value column (e.g., "Actual"), and "Industry".
            label (str, optional): The name of the value column to compare. Defaults to "Actual".
            language (str, optional): Language for graph title and labels ("en" or "he"). Defaults to "en".
    
        Returns:
            go.Figure: A Plotly figure comparing average sector values over time.
        """

        title = f"Sector Average Comparison – {label}" if language == "en" else f"השוואת מגזרים לפי ממוצע {label}"
        y_label = f"Average {label} Value" if language == "en" else f"ממוצע {label}"
        fig = px.line(
            df,
            x="Date",
            y=label,
            color="Industry",
            title=title,
            labels={label: y_label}
        )
        fig.update_layout(legend_title="Industry" if language == "en" else "תחום", plot_bgcolor="#fdf6e3")
        return fig

    def customize(self, fig):
        """Applies consistent styling and formatting to a Plotly figure.
    
        This method updates layout settings such as background color, font styles,
        color palette, and tick/hover label formatting to ensure visual consistency
        across all generated charts.
    
        Args:
            fig (go.Figure): A Plotly figure to customize.
    
        Returns:
            go.Figure: The same figure with updated layout and style settings.
        """
        fig.update_layout(
            plot_bgcolor="#F9F6E6",
            font=dict(family="Arial, sans-serif", size=15, color="#333333"),
            colorway=["#001219", "#005f73", "#0a9396", "#94d2bd", "#ee9b00", "#ca6702", "#bb3e03", "#9b2226"],
            hoverlabel=dict(font_size=16),
            xaxis=dict(tickfont=dict(size=14)),
            yaxis=dict(tickfont=dict(size=14)),
        )
        return fig

