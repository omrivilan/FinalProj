import streamlit as st
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import warnings
import time
import uuid
from chatbot_engine import ChatbotEngine

from langchain_core.messages import AIMessage, HumanMessage
from intent_detector import IntentDetector
from stock_data_handler import StockDataHandler
from graph_generator import GraphGenerator
from database_mongo import get_chat_by_id
from plotly.io import from_json
warnings.filterwarnings("ignore")

# Global Configuration
API_KEY = 'AIzaSyC3XqPeca_kNxjsSb64aHvJbJvyakyGKQI'
LSTM_CSV_PATH = os.path.join("LSTM", "actual_vs_pred_lstm_without_reports.csv")
XGBOOST_CSV_PATH = os.path.join("XGBoost", "model_XGBoost_metrics_and_predictions_without_report_parameters.csv")
LIGHTGBM_CSV_PATH = os.path.join("LightGBM", "model_LightGBM_metrics_and_predictions_total.csv")
BEST_MODEL_CSV = os.path.join("Metrics", "without_ARIMA_model_to_stock.csv")
SECTORS_DF_PATH = "sectors_df.csv"
MAPPING_FILE_PATH = "company_name_to_ticker.xlsx"

if "mentioned_tickers" not in st.session_state:
    st.session_state.mentioned_tickers = set()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

class AppManager:
    """
    Manages the full Streamlit application lifecycle for the Robo Advisor chatbot.

    This class controls the layout, user interaction flow, and integration between the UI
    and the backend logic (including the chatbot engine, intent detection, and data visualization).
    It handles the display of welcome content, disclaimer enforcement, chat session management,
    user input capture, and response rendering — including both summaries and deeper analysis.

    Key responsibilities include:
    - Displaying app instructions, default prompts, and usage guidelines
    - Tracking chat sessions and enabling multi-turn memory
    - Calling the backend engine to process prompts and generate insights
    - Visualizing actual, predicted, and forecasted stock trends
    - Managing bilingual support (Hebrew/English) for chatbot interaction
    - Supporting optional deep-dive analysis on demand

    This class serves as the main controller that binds together all components of the chatbot
    application into a coherent, responsive user experience.
    """

    def __init__(self):
        """Initializes the AppManager with all necessary components.

        Sets up API keys, loads models results, ticker and industry mappings, and initializes
        the intent detector, stock data handler, graph generator, and chatbot engine.

        Attributes:
            api_key (str): API key for Gemini, pulled from environment or default.
            ticker_mapping (dict): Mapping of company names to their stock tickers.
            industry_mapping (dict): Mapping of industry names to standardized formats.
            intent_detector (IntentDetector): Detects user intent based on input queries.
            model_paths (dict): Dictionary mapping model names to their CSV file paths.
            data_handler (StockDataHandler): Handles extraction and transformation of stock data.
            graph_generator (GraphGenerator): Responsible for generating visual stock plots.
            engine (ChatbotEngine): The main engine for processing user prompts and generating replies.
        """

        self.api_key = os.getenv("GEMINI_API_KEY", API_KEY)
        self.ticker_mapping = self.load_ticker_mapping()
        self.industry_mapping = self.load_industry_mapping()
        self.intent_detector = IntentDetector(self.ticker_mapping, self.industry_mapping)
        self.model_paths = {
            "LSTM": LSTM_CSV_PATH,
            "GRU": os.path.join("GRU", "actual_vs_pred_gru_without_reports.csv"),
            "XGBoost": XGBOOST_CSV_PATH,
            "LightGBM": LIGHTGBM_CSV_PATH,
        }

        self.data_handler = StockDataHandler(
            self.ticker_mapping,
            model_paths=self.model_paths,
            sector_path=SECTORS_DF_PATH,
            best_model_path=BEST_MODEL_CSV
        )

        self.graph_generator = GraphGenerator()
        self.engine = ChatbotEngine(self.data_handler, self.graph_generator, self.intent_detector)

    def load_ticker_mapping(self) -> dict:
        """Loads the mapping of company names to stock tickers.

        Reads from the Excel file defined by `MAPPING_FILE_PATH` and constructs
        a dictionary where keys are uppercase company names and values are uppercase tickers.

        Returns:
            dict: A dictionary mapping company names to their corresponding tickers.
        """
        df = pd.read_excel(MAPPING_FILE_PATH)
        return {row["CompanyName"].upper(): row["Ticker"].upper() for _, row in df.iterrows()}

    def load_industry_mapping(self) -> dict:
        """Loads and standardizes the mapping of noisy industry names to canonical forms.

        Reads industry names from the CSV at `SECTORS_DF_PATH`, removes rows with missing values,
        and constructs a dictionary mapping cleaned, lowercase versions of industry names
        to their original canonical forms.

        Returns:
            dict: A dictionary mapping cleaned industry names to their canonical representations.
        """
        df = pd.read_csv(SECTORS_DF_PATH)
        df = df.dropna(subset=["Industry"])
        canonical_industries = df["Industry"].unique()
        industry_mapping = {}
        for industry in canonical_industries:
            cleaned = (
                str(industry).lower()
                .replace("-", " ")
                .replace(":", " ")
                .replace("_", " ")
                .replace(",", " ")
            )
            cleaned = " ".join(cleaned.split())
            industry_mapping[cleaned] = industry.strip()
        return industry_mapping

    def run(self):
        """
        Launches and manages the full user interface of the Robo Advisor chatbot application.

        This method is responsible for initializing the Streamlit layout, rendering welcome messages,
        managing chat history, handling user input, switching between historical chats, and processing
        responses from the chatbot engine. It also includes support for deeper analysis using Gemini
        models and allows toggling between Hebrew and English modes.

        Workflow:
            - If the user has not accepted the disclaimer, display onboarding content.
            - When the disclaimer is accepted, initialize the chat application.
            - Display previous chat sessions and allow switching between them.
            - Render chat history with text and graphs.
            - Allow new user inputs via preset buttons or manual entry.
            - Use intent detection and language detection to generate structured responses.
            - Optionally offer deeper analysis using Gemini.
            - Log all interactions to the Streamlit session state.

        Key Functional Components:
            - **Welcome Screen and Disclaimer**:
                - Displayed only if `st.session_state.accepted_disclaimer` is `False`.
                - Includes markdown explanations of the app's purpose, capabilities, and academic context.
                - Contains a button that sets the disclaimer as accepted and reloads the app.

            - **Chat History Handling**:
                - Loads available chat sessions from `st.session_state.available_chats`.
                - Allows selection of past sessions through a radio button menu.
                - Rehydrates the selected session, reconstructing user and assistant messages,
                  including graphs and deeper analysis components.

            - **Chat Display Logic**:
                - Renders each user message (`HumanMessage`) and assistant message (`dict` or `AIMessage`).
                - Supports both simple text responses and responses that include one or more Plotly figures.
                - Hebrew and English alignment is adjusted automatically based on language detection.

            - **Default Prompt Buttons**:
                - Provides quick-start prompts like:
                    - "show me the graph of leumi"
                    - "compare leumi and poalim"
                    - "show me all banks"
                - If clicked, sets the `selected_prompt` to trigger chatbot input.

            - **User Input Handling**:
                - Accepts either manual input via `st.chat_input()` or selection from default prompts.
                - Appends user input to the chat history.
                - Triggers chatbot response generation through `self.engine.handle_input(prompt)`.

            - **Structured Response Handling**:
                - Visualizes graphs in the assistant’s response with stable Streamlit keys.
                - Displays both the assistant's main answer and any follow-up deep analysis.
                - Saves the assistant's response back into `st.session_state.chat_history`.

            - **Deeper Analysis Scheduling**:
                - If a relevant intent is detected (e.g., "graph", "compare", "industry_values"), a flag is set.
                - Displays a prompt asking if the user wants deeper analysis.
                - If accepted, invokes `self.engine._generate_deeper_analysis()` and appends the result.

        Returns:
            None
        """
        def get_clean_chat_history():
            #load the chat history
            lines = []
            for m in st.session_state.chat_history:
                if isinstance(m, HumanMessage):
                    lines.append(f"User: {m.content}")
                elif isinstance(m, AIMessage):
                    lines.append(f"Assistant: {m.content}")
                elif isinstance(m, dict):
                    if m.get("role") == "assistant":
                        if "deep_analysis" in m:
                            lines.append(f"Assistant: {m['deep_analysis']}")
                        elif "text" in m:
                            lines.append(f"Assistant: {m['text']}")
                    elif m.get("role") == "user" and "text" in m:
                        lines.append(f"User: {m['text']}")
            return "\n".join(lines)

        if "accepted_disclaimer" not in st.session_state:
            st.session_state.accepted_disclaimer = False

        if not st.session_state.accepted_disclaimer:
            # ── Welcome Screen and Disclaimer ─────────────────────────────────────
            # Displays the initial welcome message and project description.
            # If the user hasn't yet accepted the disclaimer, this section blocks access
            # to the rest of the app and prompts the user to confirm understanding.

            user = st.session_state.get("username", None)
            if user:
                st.markdown(f"### Hello, **{user}** 👋")
            st.title("🤖 Welcome to the Robo Advisor – Israeli Stock Market")

            st.markdown("## 📊 Stock Analysis and Forecasting Application")
            st.markdown("""
        This is a **Streamlit-based tool** that helps you analyze and forecast stock performance, especially for companies in the Israeli (TA) stock market.

        It combines **advanced machine learning models** like **LSTM**, **GRU**, **XGBoost** and **LightGBM** to deliver clear insights into stock behavior.

        Whether you're a student, investor, or just curious — this app makes stock analysis simple, visual, and accessible.
        """)

            st.markdown("## 🧠 What Can This App Do?")
            st.markdown("""
        - 📈 **Visualize actual vs predicted stock performance**
        - 🔮 **Forecast future trends using advanced models**
        - 🏭 **Explore industry-wide behavior across sectors**
        - ⚖️ **Compare multiple stocks side-by-side**
        - 🤖 **Interact with an AI chatbot** that answers your questions and generates graphs

        The chatbot uses **natural language processing** to understand your queries and provide visual + textual feedback.
        """)

            st.markdown("## ⚙️ How Does It Work?")
            st.markdown("""
        The app uses machine learning models to detect patterns in historical stock prices and predict future performance:

        - **LSTM / GRU** → sequence-aware models for time-series forecasting  
        - **XGBoost / LightGBM** → powerful tree-based models for structured tabular data  

        It visualizes **actual**, **predicted**, and **forecasted** data for each company or sector in interactive graphs.
        """)

            st.markdown("## 💬 How Do I Use It?")
            st.markdown("""
        You can ask the chatbot anything like:

        - 🟢 `show me the graph of leumi`
        - 🟢 `compare leumi and poalim`
        - 🟢 `show me all banks`

        The app will detect your intent and generate relevant insights and visuals.
        """)

            st.markdown("## 🎓 Academic Context")
            st.markdown("""
        This application was developed as part of a **Data Science Capstone Project** at **Ben-Gurion University (BGU)**, by students in the Department of Software and Information Systems Engineering.
        """)

            st.markdown("## ⚠️ Disclaimer")
            st.warning("""
        The chatbot's responses are **not binding financial recommendations** and do **not replace professional advice**.  
        All outputs are based on historical data and model estimations and are provided for educational purposes only.
        """)

            st.markdown("")

            if st.button("✅ I understand and wish to continue"):
                st.session_state.accepted_disclaimer = True
                st.rerun()


            return

        # st.set_page_config(page_title="Robo Advisor", layout="wide")

        # ── INSERT A GREETING ─────────────────────────────────────────────────────────
        user = st.session_state.get("username", None)
        if user:
            st.markdown(f"### Hello, **{user}** 👋")
        
        
        # Main interface components shown after the user accepts the disclaimer.
        # Initializes sidebar info, loads chat sessions, and prepares prompt buttons
        # for interacting with the AI stock analysis chatbot.
        # ── Main App Interface (Post-Disclaimer) ─────────────────────────────

        st.title("🤖 Robo Advisor – Israeli Stock Market")
        st.sidebar.title("About")
        st.sidebar.info("This chatbot provides stock analysis using historical and forecasted data.")
        # Displays a list of existing chat sessions in the sidebar.
        # Enables users to switch between conversations by showing their latest message.
        # Automatically loads historical chat content and reconstructed graphs.
        # ── Chat Session Management Sidebar ─────────────────────────────────
        st.sidebar.title("Your Chats")

        # 1️⃣ Sort sessions newest→oldest so “New Chat” appears first
        chat_metas = sorted(
            st.session_state.get("available_chats", []),
            key=lambda c: c["last_updated"],
            reverse=True
        )

        # 2️⃣ Build a list of labels (first user prompt or “New Chat”)
        chat_labels = []
        for meta in chat_metas:
            doc = get_chat_by_id(meta["_id"])
            first = next(
                (m.get("content", "") for m in doc.get("chat_history", []) if m.get("role") == "user"),
                ""
            )
            chat_labels.append(first.strip() or "New Chat")

        # 3️⃣ Render as a radio (segmented) control, defaulting to index 0
        if chat_labels:
            idx = st.sidebar.radio(
                "Switch chats",
                options=list(range(len(chat_labels))),
                format_func=lambda i: chat_labels[i],
                index=0,
                key="selected_chat_idx"
            )

            selected_meta = chat_metas[idx]
            if str(selected_meta["_id"]) != st.session_state.get("current_chat_id"):
                # persist the chat we’re leaving
                self.engine.save_chat()

                # load the newly selected session
                chat_doc = get_chat_by_id(selected_meta["_id"])
                st.session_state.chat_history = []
                for m in chat_doc.get("chat_history", []):
                    # unify old vs new summary key
                    main_text = m.get("content") or m.get("text", "") or ""
                    # rebuild any graphs
                    figs = []
                    for fig_json in m.get("graphs", []):
                        try:
                            figs.append(from_json(fig_json))
                        except:
                            pass

                    # deep analysis might live in its own field
                    deep = m.get("deep_analysis")

                    # skip truly empty placeholder messages
                    if not (main_text.strip() or figs or deep):
                        continue

                    if m.get("role") == "user":
                        st.session_state.chat_history.append(
                            HumanMessage(content=main_text)
                        )
                    else:
                        entry = {
                            "role":      "assistant",
                            "content":   main_text,
                            "graphs":    figs
                        }
                        if deep:
                            entry["deep_analysis"] = deep

                        st.session_state.chat_history.append(entry)


                # mark this as the active chat
                st.session_state.current_chat_id = str(selected_meta["_id"])

        # Offers default queries as clickable buttons for convenience.
        # These prompts demonstrate the system’s capabilities and help new users get started.
        # ── Quick Prompt Buttons (Predefined Queries) ───────────────────────
        # Default prompt buttons
        default_prompts = [
            "show me the graph of leumi",
            "compare leumi and poalim",
            "show me all banks"
        ]
        cols = st.columns(len(default_prompts))
        selected_prompt = None
        for i, p in enumerate(default_prompts):
            if cols[i].button(p, use_container_width=True):
                selected_prompt = p

        # Renders all previous user and assistant messages from the selected chat session.
        # Handles different message types: user text, assistant text, associated graphs,
        # and deeper analysis if it was previously generated.
        # ── Display chat history ─────────────────────────────────────────────────
        for msg_idx, message in enumerate(st.session_state.chat_history):
            if isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.write(message.content)

            # our reconstructed assistant dicts
            elif isinstance(message, dict) and message.get("role") == "assistant":
                with st.chat_message("assistant"):
                    # 1️⃣ Render each Plotly figure with a stable key
                    for fig_idx, fig in enumerate(message.get("graphs", [])):
                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            key=f"chat{msg_idx}_fig{fig_idx}"
                        )

                    # 2️⃣ Render the main summary text
                    content = message.get("content")
                    if content:
                        lang  = st.session_state.get("language", "en")
                        align = "right" if lang == "he" else "left"
                        dir_  = "rtl"   if lang == "he" else "ltr"
                        st.markdown(
                            f'<div dir="{dir_}" style="text-align: {align}; font-size: 18px;">'
                            f'{content}</div>',
                            unsafe_allow_html=True
                        )

                    # 3️⃣ Render deeper analysis if present
                    deep = message.get("deep_analysis")
                    if deep:
                        st.markdown("### 🔍 Deeper Analysis")
                        lang  = st.session_state.get("language", "en")
                        align = "right" if lang == "he" else "left"
                        dir_  = "rtl"   if lang == "he" else "ltr"
                        st.markdown(
                            f'<div dir="{dir_}" style="text-align: {align}; font-size: 18px;">'
                            f'{deep}</div>',
                            unsafe_allow_html=True
                        )

            # legacy AIMessage objects (fallback)
            elif isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.write(message.content)

        # Captures either a typed prompt or selected quick-prompt.
        # Detects input language, adds the user message to the session,
        # and calls the chatbot engine to handle the query (intent detection, graph generation, etc.).
        # ── User Input and Response Generation ──────────────────────────────
        # User input
        manual_input = st.chat_input("What would you like to know?")
        prompt = selected_prompt or manual_input

        if prompt:
            # Language detection & append
            st.session_state["language"] = "he" if any('\u0590' <= c <= '\u05EA' for c in prompt) else "en"
            user_msg = HumanMessage(content=prompt)
            st.session_state.chat_history.append(user_msg)
            with st.chat_message("user"): st.write(prompt)

            # Generate and display summary + graphs
            structured_response = self.engine.handle_input(prompt)
            st.session_state["last_structured_response"] = structured_response
            with st.chat_message("assistant"):
                for i, graph in enumerate(structured_response.get("graphs", [])):
                    st.plotly_chart(graph, use_container_width=True, key=f"main_{i}_{uuid.uuid4()}")
                if structured_response.get("text"):
                    lang = st.session_state.get("language", "en")
                    align = "right" if lang == "he" else "left"
                    dir_attr = "rtl" if lang == "he" else "ltr"
                    st.markdown(
                        f'<div dir="{dir_attr}" style="text-align: {align}; font-size: 18px;">{structured_response["text"]}</div>',
                        unsafe_allow_html=True
                    )

            # Save summary to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": structured_response.get("text", ""),
                "graphs": structured_response.get("graphs", [])
            })

            # If the chatbot successfully processes an intent that supports deeper insights,
            # a flag is set in session state to optionally trigger an additional round of analysis.
            # ── Schedule Deeper Analysis Prompt ─────────────────────────────────
            if structured_response.get("intent") in ["graph", "compare", "industry_values", "sector_comparison", "addition"]:
                st.session_state["deep_analysis_pending"] = {
                    "summary": structured_response["text"],
                    "context": f"{structured_response.get('intent').capitalize()} Analysis"
                }

        # If deeper analysis is pending, prompts the user to select wether he would like deeper explenations.
        # If confirmed, generates a more detailed textual interpretation using Gemini.
        # This result is appended to the chat history and shown to the user.
        # ── Deeper Analysis Execution ───────────────────────────────────────
        if "deep_analysis_pending" in st.session_state:
            st.markdown("### 🔍 Would you like a deeper analysis?")
            c1, c2 = st.columns([1,1])
            if c1.button("🎨 Yes, show me!", use_container_width=True):
                with st.spinner("🔍 Generating deeper analysis..."):
                    pending = st.session_state.pop("deep_analysis_pending")
                    deep_result = self.engine._generate_deeper_analysis(
                        pending["summary"], context_info=pending["context"]
                    )
                    with st.chat_message("assistant"):
                        st.markdown("### 🔍 Deeper Analysis")
                        lang = st.session_state.get("language", "en")
                        align = "right" if lang == "he" else "left"
                        dir_attr = "rtl" if lang == "he" else "ltr"
                        st.markdown(
                            f'<div dir="{dir_attr}" style="text-align: {align}; font-size: 18px;">{deep_result}</div>',
                            unsafe_allow_html=True
                        )
                    st.session_state.chat_history.append({"role": "assistant", "deep_analysis": deep_result})
            if c2.button("❌ No thanks", use_container_width=True):
                st.session_state.pop("deep_analysis_pending", None)
                st.rerun()

if __name__ == "__main__":
    app = AppManager()
    app.run()
