
import os
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain.tools.render import format_tool_to_openai_function
from functools import reduce
from Levenshtein import distance as levenshtein_distance
from database_mongo import create_chat, update_chat
from db import users_chat_col
import plotly.graph_objects as go

API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyC3XqPeca_kNxjsSb64aHvJbJvyakyGKQI")

# Global Configuration
API_KEY = 'AIzaSyC3XqPeca_kNxjsSb64aHvJbJvyakyGKQI'
BASE_DIR = os.getcwd()


MAPPING_FILE_PATH = os.path.join(BASE_DIR, "company_name_to_ticker.xlsx")
LSTM_CSV_PATH = os.path.join(BASE_DIR, "LSTM", "actual_vs_pred_lstm_without_reports.csv")
XGBOOST_CSV_PATH = os.path.join(BASE_DIR, "XGBoost", "model_XGBoost_metrics_and_predictions_without_report_parameters.csv")
LIGHTGBM_CSV_PATH = os.path.join(BASE_DIR, "LightGBM", "model_LightGBM_metrics_and_predictions_total.csv")
BEST_MODEL_CSV = os.path.join(BASE_DIR, "Metrics", "without_ARIMA_model_to_stock.csv")
SECTORS_DF_PATH = os.path.join(BASE_DIR, "sectors_df.csv")
comp_text = pd.read_excel(MAPPING_FILE_PATH).to_markdown(index=False)
sectors_text = pd.read_csv(SECTORS_DF_PATH)[["Industry"]].to_markdown(index=False)
def detect_language(text):
    return "he" if any('֐' <= c <= 'ת' for c in text) else "en"
def get_llm_instance(tools=None):
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        stream=True,
        temperature=0.3,
        google_api_key=API_KEY,
        tools=tools,
        enable_automatic_function_calling=True,
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
        }
    )
def get_common_dates(frames):
    date_sets = [set(df["Date"]) for df in frames if not df.empty]
    return sorted(set.intersection(*date_sets)) if date_sets else []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
class ChatbotEngine:
    def __init__(self, data_handler, graph_generator, intent_detector):
        self.tools = [
            {
                "name": "graph",
                "description": "Show actual, predicted, and forecasted values for a single company as a graph.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input_tup": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "A tuple (ticker, company name)."
                        }
                    },
                    "required": ["input_tup"]
                }
            },
            {
                "name": "compare",
                "description": "Compare multiple companies’ stock performances.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input_tup_lst": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "description": "A list of tuples (ticker, company name)."
                        }
                    },
                    "required": ["input_tup_lst"]
                }
            },
            {
                "name": "industry_values",
                "description": "Return sector-wide stock metrics (actual, predicted, forecasted) for a given sector.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sector_name": {
                            "type": "string",
                            "description": "Exact name of the sector to analyze."
                        }
                    },
                    "required": ["sector_name"]
                }
            },
            {
                "name": "sector_comparison",
                "description": "Compare multiple sectors based on average stock trends.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sector_names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of sector names."
                        }
                    },
                    "required": ["sector_names"]
                }
            }
        ]
        # Register tools

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-lite",  # or "gemini-2.0-flash-lite"
            stream=True,
            temperature=0.3,
            google_api_key=API_KEY,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
            },
            tools=self.tools,           # ✅ here’s the correct fix
            tool_choice="auto"          # ✅ let Gemini pick when to use tools
        )
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        with open("company_name_to_ticker.xlsx", "rb") as f:
            df = pd.read_excel(f)

        valid_names = sorted(set(df["CompanyName"].str.upper().tolist() + df["Ticker"].str.upper().tolist()))
        self.valid_entities_text = "\n".join(f"- {name}" for name in valid_names)



        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             f"You MUST always respond using the function-calling tools if the user's query matches one of the defined tool descriptions.\n\n" 
             f"Never output raw Python code or markdown unless explicitly asked.\n\n" 
             f"Always answer in the language that the user wrote to you-english or hebrew.\n\n" 
             f"Do not answer questions by generating code unless the user says 'give me code' or 'show me how to implement'.\n\n"
            f"You are a chatbot assistant designed to analyze stocks on the Israeli (TA) stock market.\n\n"
            f"🎯 Your primary task is to determine which of the following tools best fits the user's request:\n"
            f"- `'graph'`: Use for a **single company** asking to see actual, predicted, or forecasted values.\n"
            f"- `'compare'`: Use when **two or more companies** are mentioned and the user wants to compare them.\n"
            f"- `'industry_values'`: Use for questions about a **single sector's** overall performance.\n"
            f"- `'sector_comparison'`: Use when comparing **multiple sectors** (e.g., 'tech vs finance').\n\n"
            
            f"🧠 You must reason which tool fits best. If no tool fits well, reply with helpful text.\n"
            f"⚙️ Use function calling whenever a tool clearly matches the prompt.\n"
            f"❌ Do NOT ask for clarification—choose the closest match and proceed.\n\n"


            f"📌 Examples:\n"
            f"- 'Show me teva' → Use `'graph'`\n"
            f"- 'Compare teva and leumi' → Use `'compare'`\n"
            f"- 'How is the banking sector doing?' → Use `'industry_values'`\n"
            f"- 'Compare tech and pharma sectors' → Use `'sector_comparison'`\n\n"

            f"📎 Known companies:\n{self.valid_entities_text}\n\n"
            f"📎 Company Mapping Table:\n{comp_text}\n\n"
            f"📎 Sector Names:\n{sectors_text}\n\n"

            f"⛔ Never hallucinate new tickers or sector names.\n"
            f"If a company/sector isn't recognized, suggest the closest match and ask for confirmation.\n"
            f"Wait for the user to reply 'yes' before proceeding with correction.\n"
            f"If you are given a prompt in the Hebrew language, you must answer in Hebrew in that response only, unless explicitly asked in another language"


            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{user_query}")
        ])


        self.chain = self.prompt | self.llm | StrOutputParser()

        self.data_handler = data_handler
        self.graph_generator = graph_generator
        self.intent_detector = intent_detector

    def save_chat(self):
        """
        Serialize st.session_state.chat_history (including graphs) and upsert.
        """
        username = st.session_state.get("username")
        if not username:
            return

        serialized = []
        for m in st.session_state.chat_history:
            # Assistant messages stored as dict
            if isinstance(m, dict):
                entry = {
                    "role": m.get("role", "assistant"),
                    "content": m.get("content", "")
                }

                # ─── NEW: serialize any Plotly graphs safely ───
                graphs = m.get("graphs", [])
                json_graphs = []
                for fig in graphs:
                    if isinstance(fig, go.Figure):
                        json_graphs.append(fig.to_json())
                if json_graphs:
                    entry["graphs"] = json_graphs

                # carry over any deeper analysis, etc.
                if "deep_analysis" in m:
                    entry["deep_analysis"] = m["deep_analysis"]

            # HumanMessage → simple user entry
            elif isinstance(m, HumanMessage):
                entry = {"role": "user", "content": m.content}

            # AIMessage → simple assistant entry
            elif isinstance(m, AIMessage):
                entry = {"role": "assistant", "content": m.content}

            else:
                continue

            serialized.append(entry)

        # Upsert into MongoDB as before…
        chat_id = st.session_state.get("current_chat_id")
        if chat_id:
            modified = update_chat(chat_id, serialized)
            if modified == 0:
                new_id = create_chat(username, serialized)
                st.session_state.current_chat_id = str(new_id)
        else:
            new_id = create_chat(username, serialized)
            st.session_state.current_chat_id = str(new_id)


    def handle_input(self, user_input: str):
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        def clean_chat_history():
            def convert(m):
                if isinstance(m, (HumanMessage, AIMessage)):
                    return m
                if isinstance(m, dict):
                    if m.get("role") == "user" and "text" in m:
                        return HumanMessage(content=m["text"])
                    if m.get("role") == "assistant" and "text" in m:
                        return AIMessage(content=m["text"])
                    if m.get("role") == "assistant" and "content" in m:
                        return AIMessage(content=m["content"])
                return None

            return [msg for msg in (convert(m) for m in st.session_state.chat_history) if msg]

        try:
            # ── Correction‐confirmation flow ─────────────────────────────────────────
            if user_input.strip().lower() == "yes" and "suggested_correction" in st.session_state:
                corrected_term = st.session_state.pop("suggested_correction")
                last_prompt    = st.session_state.get("original_prompt", "")

                ticker_map = self.intent_detector.ticker_mapping
                if corrected_term in ticker_map.values():
                    names = [k for k, v in ticker_map.items() if v == corrected_term]
                    if names:
                        corrected_term = names[0]

                if corrected_term and last_prompt:
                    new_prompt = last_prompt.rsplit(" ", 1)[0] + " " + corrected_term
                    st.session_state.original_prompt = new_prompt
                    st.session_state.chat_history.append(HumanMessage(content=new_prompt))
                    user_input = new_prompt

            # ── Language detection ────────────────────────────────────────────────────
            language = "he" if any('\u0590' <= c <= '\u05EA' for c in user_input) else "en"
            st.session_state["language"] = language

            # ── Intent detection ──────────────────────────────────────────────────────
            prev_intent = st.session_state.get("last_intent", None)
            detected    = self.intent_detector.detect(user_input, last_intent=prev_intent)
            intent      = detected.get("intent")
            comps       = detected.get("companies", None)
            industries  = detected.get("industries", None)
            st.session_state["last_intent"] = intent
            print(intent)

            # ── 0️⃣ Handle “addition” intent ───────────────────────────────────────
            if intent == "addition":
                # • company‐level addition
                if comps is not None:
                    merged = comps
                    st.session_state.last_companies = merged
                    # if we were graphing before, preserve that, otherwise compare
                    base = prev_intent if prev_intent in {"graph", "compare"} else "compare"
                    st.session_state.last_intent = base
                    # merged list of ≥2 always compares
                    result = self._handle_compare_intent({"companies": merged})
                    self.save_chat()
                    return result


                # • industry‐level addition
                if industries is not None:
                    merged_inds = industries
                    st.session_state.last_industries = merged_inds
                    st.session_state.last_intent     = "sector_comparison"
                    result = self._handle_sector_comparison_intent(merged_inds)
                    self.save_chat()
                    return result

            # ── 1️⃣ Fresh graph ───────────────────────────────────────────────────
            if intent == "graph":
                st.session_state.last_companies = [detected["company"]]
                st.session_state.last_intent    = "graph"
                result = self._handle_graph_intent({"company": detected["company"]})
                st.session_state["deep_analysis_pending"] = {
                    "summary": result["text"],
                    "context": f"{detected['company']} Stock Forecast"
                }
                self.save_chat()
                return result

            # ── 2️⃣ Fresh compare ─────────────────────────────────────────────────
            if intent == "compare":
                result = self._handle_compare_intent({"companies": detected["companies"]})
                st.session_state["deep_analysis_pending"] = {
                    "summary": result["text"],
                    "context": "Stock Comparison"
                }
                self.save_chat()
                return result

            # ── 3️⃣ Fresh industry values ─────────────────────────────────────────
            if intent == "industry_values":
                result = self._handle_industry_intent({"industry": detected["industry"]})
                st.session_state["deep_analysis_pending"] = {
                    "summary": result["text"],
                    "context": f"{detected['industry']} Industry"
                }
                self.save_chat()
                return result

            # ── 4️⃣ Fresh sector comparison ────────────────────────────────────────
            if intent == "sector_comparison":
                st.session_state.last_industries = detected["industries"]
                st.session_state["deep_analysis_pending"] = {
                    "summary": result["text"],
                    "context": f"{detected['industries']} Industries"
                }
                self.save_chat()
                return self._handle_sector_comparison_intent(detected["industries"])

        except Exception:
            pass

        # ── Fallback to LLM with chat history ───────────────────────────────────
        cleaned_history = clean_chat_history()
        messages = self.prompt.format_prompt(
            user_query=user_input,
            chat_history=cleaned_history
        ).to_messages()
        response = self.llm.invoke(messages)

        if hasattr(response, "tool_calls") and response.tool_calls:
            return self._handle_tool_call(response)

        output = {"text": response.content, "intent": "fallback", "raw_input": user_input}
        self.save_chat()
        return output


    def _stream_response(self, user_query):
        def convert_valid_message(m):
            if isinstance(m, (HumanMessage, AIMessage)):
                return m
            elif isinstance(m, dict):
                role = m.get("role")
                if role == "user" and "text" in m:
                    return HumanMessage(content=m["text"])
                elif role == "assistant" and "text" in m:
                    return AIMessage(content=m["text"])
                elif role == "assistant" and "deep_analysis" in m:
                    return AIMessage(content=m["deep_analysis"])
            return None

        # ✅ Language detection for streaming as well
        st.session_state["language"] = "he" if any('\u0590' <= c <= '\u05EA' for c in user_query) else "en"

        cleaned_history = [
            convert_valid_message(m)
            for m in st.session_state.chat_history
        ]
        cleaned_history = [m for m in cleaned_history if m is not None]

        return self.chain.stream({
            "user_query": user_query,
            "chat_history": cleaned_history
        })



    def _record_response(self, user_input, response):
        self.memory.chat_memory.add_user_message(user_input)
        self.memory.chat_memory.add_ai_message(response)

    # ====== Intent Handlers (static text, no stream) ======

    def _handle_industry_intent(self, intent_data):
        industry = intent_data["industry"]
        language = st.session_state.get("language", "en")
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 12, 31)

        actual_df = self.data_handler.get_industry_actuals(industry, start_date, end_date)
        predicted_df = self.data_handler.get_industry_predictions(industry, start_date, end_date)
        best_model_df = pd.read_csv(self.data_handler.best_model_path)

        forecast_dfs = []
        ticker_to_company = {v: k for k, v in self.data_handler.ticker_mapping.items()}

        for ticker in actual_df["Ticker"].dropna().unique():
            try:
                model = best_model_df.loc[best_model_df["Company"] == ticker, "Model"].values[0]
                ap_data = self.data_handler.extract_by_model(ticker, model, start_date, end_date)
                last_date = ap_data["Date"].max()
                forecast = self.data_handler.extract_forecasted_values(ticker, last_date, ap_data)
                forecast["Company"] = ticker_to_company.get(ticker.replace(".TA", ""), ticker)
                forecast["Ticker"] = ticker
                forecast_dfs.append(forecast)
            except:
                continue

        full_forecast_df = pd.concat(forecast_dfs) if forecast_dfs else pd.DataFrame()

        fig_actual = self.graph_generator.generate_industry_graph(actual_df, "Actual", industry, language=language)
        fig_pred = self.graph_generator.generate_industry_graph(predicted_df, "Predicted", industry, language=language)
        fig_forecast = self.graph_generator.generate_industry_graph(full_forecast_df, "Forecasted", industry, language=language)

        summary = self._generate_industry_summary(actual_df, predicted_df, full_forecast_df, ticker_to_company, industry, language=language)
        st.session_state.last_industry = industry

        return {"text": summary, "graphs": [fig_actual, fig_pred, fig_forecast]}


    def _handle_graph_intent(self, intent_data):
        company = intent_data["company"].upper()
        language = st.session_state.get("language", "en")
        ticker = self.intent_detector.resolve_ticker(company) + ".TA"
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 12, 31)

        best_model_df = pd.read_csv(self.data_handler.best_model_path)
        model = best_model_df.loc[best_model_df["Company"] == ticker, "Model"].values[0]

        data = self.data_handler.extract_by_model(ticker, model, start_date, end_date)
        last_date = data["Date"].max()
        forecast = self.data_handler.extract_forecasted_values(ticker, last_date, data)
        forecast["Date"] = pd.date_range(start=last_date + timedelta(days=1), periods=len(forecast))

        combined_df = pd.concat([
            data[["Date", "Actual", "Predicted"]],
            forecast[["Date", "Forecasted"]]
        ], ignore_index=True).sort_values("Date")

        fig = self.graph_generator.generate_actual_predicted_forecast_graph(combined_df, company, model, language=language)
        text = self._generate_single_stock_summary(company, data, forecast, model, language=language)
        return {"text": text, "graphs": [fig]}


    def _handle_compare_intent(self, intent_data):
        companies = intent_data["companies"]
        language = st.session_state.get("language", "en")
        tickers = [self.intent_detector.resolve_ticker(name.upper()) + ".TA" for name in companies]
        ticker_to_company = {v: k for k, v in self.data_handler.ticker_mapping.items()}

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 12, 31)
        best_model_df = pd.read_csv(self.data_handler.best_model_path)

        combined_dataframes = []
        summaries = []

        for ticker in tickers:
            try:
                model = best_model_df.loc[best_model_df["Company"] == ticker, "Model"].values[0]
                data = self.data_handler.extract_by_model(ticker, model, start_date, end_date)
                last_date = data["Date"].max()
                forecast = self.data_handler.extract_forecasted_values(ticker, last_date, data)
                forecast["Date"] = pd.date_range(start=last_date + timedelta(days=1), periods=len(forecast))

                combined = pd.concat([
                    data[["Date", "Actual", "Predicted"]],
                    forecast[["Date", "Forecasted"]]
                ], ignore_index=True).sort_values("Date")

                combined_dataframes.append((ticker, model, combined))
                summaries.append(self._generate_single_stock_summary(
                    ticker_to_company.get(ticker.replace(".TA", ""), ticker), data, forecast, model, language=language))
            except:
                continue

        if not combined_dataframes:
            return {"text": "No valid data found for comparison."}

        fig = self.graph_generator.generate_comparison_graph(combined_dataframes, ticker_to_company, language=language)
        summary = self._generate_comparison_summary(summaries, language=language)
        return {"text": summary, "graphs": [fig]}

    def _handle_sector_comparison_intent(self, industries):
        language = st.session_state.get("language", "en")
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2025, 3, 13)

        actual_frames, predicted_frames, forecasted_frames = [], [], []

        def get_common_dates(frames):
            if not frames:
                return []
            date_sets = [set(f["Date"]) for f in frames]
            return sorted(reduce(set.intersection, date_sets))

        for industry in industries:
            full_df = self.data_handler.extract_by_industry(industry, start_date, end_date)
            if full_df.empty:
                continue

            actual = full_df.dropna(subset=["Actual"])
            predicted = full_df.dropna(subset=["Predicted"])
            forecasted = full_df.dropna(subset=["Forecasted"])

            for df, label in zip([actual, predicted, forecasted], ["Actual", "Predicted", "Forecasted"]):
                df = df.copy()
                df["Industry"] = industry
                grouped = df.groupby("Date")[label].mean().reset_index()
                grouped["Type"] = label
                grouped["Industry"] = industry

                if label == "Actual":
                    actual_frames.append(grouped)
                elif label == "Predicted":
                    predicted_frames.append(grouped)
                elif label == "Forecasted":
                    forecasted_frames.append(grouped)

        common_actual_dates = get_common_dates(actual_frames)
        common_predicted_dates = get_common_dates(predicted_frames)
        common_forecasted_dates = get_common_dates(forecasted_frames)

        actual_df = pd.concat(actual_frames)
        predicted_df = pd.concat(predicted_frames)
        forecasted_df = pd.concat(forecasted_frames)

        actual_df = actual_df[actual_df["Date"].isin(common_actual_dates)]
        predicted_df = predicted_df[predicted_df["Date"].isin(common_predicted_dates)]
        forecasted_df = forecasted_df[forecasted_df["Date"].isin(common_forecasted_dates)]

        summaries = []
        for industry in industries:
            industry_df = actual_df[actual_df["Industry"] == industry].dropna(subset=["Actual"])
            if industry_df.empty:
                continue
            start_date = industry_df["Date"].iloc[0].strftime("%Y-%m-%d")
            end_date = industry_df["Date"].iloc[-1].strftime("%Y-%m-%d")
            start_val = industry_df["Actual"].iloc[0]
            end_val = industry_df["Actual"].iloc[-1]
            mean_val = industry_df["Actual"].mean()
            trend = "upward 📈" if end_val > start_val else "downward 📉"
            increase_rate = ((end_val - start_val) / start_val) * 100 if start_val != 0 else 0
            summaries.append(
                f"- {industry}: Start={start_val:.2f} on {start_date}, End={end_val:.2f} on {end_date}, "
                f"Mean={mean_val:.2f}, Increase={increase_rate:.2f}%, Trend={trend}"
            )

        if actual_df.empty:
            return {"text": "No data found for the selected industries."}

        fig_actual = self.graph_generator.generate_sector_comparison_graph(actual_df, label="Actual", language=language)
        fig_predicted = self.graph_generator.generate_sector_comparison_graph(predicted_df, label="Predicted", language=language)
        fig_forecast = self.graph_generator.generate_sector_comparison_graph(forecasted_df, label="Forecasted", language=language)

        summary_text = (
            "📊 Sector comparison between industries:\n\n" if language == "en"
            else "📊 השוואת תחומים בין תעשיות:\n\n"
        ) + "\n".join(summaries)

        deep_analysis = self._generate_deeper_analysis(summary_text, context_info="Sector Comparison")
        return {"text": summary_text + "\n\n🔍 **Deeper Analysis**\n\n" + deep_analysis, "graphs": [fig_actual, fig_predicted, fig_forecast]}


    def _generate_industry_summary(self, actual_df, predicted_df, forecast_df, name_map, industry, language="en"):
        def summarize(df, col):
            summary = []
            for ticker in df["Ticker"].unique():
                sub = df[df["Ticker"] == ticker].dropna(subset=[col])
                if sub.empty:
                    continue
                company = name_map.get(ticker.replace(".TA", ""), ticker)
                start_val = sub[col].iloc[0]
                end_val = sub[col].iloc[-1]
                min_, max_, mean_ = sub[col].min(), sub[col].max(), sub[col].mean()
                increase_rate = ((end_val - start_val) / start_val) * 100 if start_val != 0 else 0
                trend = "📈" if end_val > start_val else "📉"

                if language == "he":
                    summary.append(
                        f"- {company}: מינימום={min_:.2f}, מקסימום={max_:.2f}, ממוצע={mean_:.2f}, "
                        f"שינוי={increase_rate:.2f}%, מגמה={trend}"
                    )
                else:
                    summary.append(
                        f"- {company}: Min={min_:.2f}, Max={max_:.2f}, Mean={mean_:.2f}, "
                        f"Increase={increase_rate:.2f}%, Trend={trend}"
                    )
            return "\n".join(summary)

        if language == "he":
            return (
                f"📊 סיכום רחב עבור תחום {industry}:\n\n"
                f"**נתוני אמת:**\n{summarize(actual_df, 'Actual')}\n\n"
                f"**חיזויים:**\n{summarize(predicted_df, 'Predicted')}\n\n"
                f"**תחזיות:**\n{summarize(forecast_df, 'Forecasted')}"
            )
        else:
            return (
                f"📊 Industry-wide summary for {industry} industry:\n\n"
                f"**Actuals:**\n{summarize(actual_df, 'Actual')}\n\n"
                f"**Predictions:**\n{summarize(predicted_df, 'Predicted')}\n\n"
                f"**Forecasts:**\n{summarize(forecast_df, 'Forecasted')}"
            )



    def _generate_single_stock_summary(self, company, data, forecast_df, model, language="en"):
        def stats(col, df):
            start_val = df[col].iloc[0]
            end_val = df[col].iloc[-1]
            return {
                "min": df[col].min(),
                "max": df[col].max(),
                "mean": df[col].mean(),
                "trend": "📈" if end_val > start_val else "📉",
                "increase_rate": ((end_val - start_val) / start_val) * 100 if start_val != 0 else 0
            }

        actual = stats("Actual", data)
        predicted = stats("Predicted", data)
        forecast = stats("Forecasted", forecast_df)

        if language == "he":
            return (
                f"📈 ניתוח מניה עבור {company} במודל: **{model}**\n"
                f"- אמת: מינימום={actual['min']:.2f}, מקסימום={actual['max']:.2f}, ממוצע={actual['mean']:.2f}, "
                f"שינוי={actual['increase_rate']:.2f}%, מגמה={actual['trend']}\n"
                f"- חיזוי: מינימום={predicted['min']:.2f}, מקסימום={predicted['max']:.2f}, ממוצע={predicted['mean']:.2f}, "
                f"שינוי={predicted['increase_rate']:.2f}%, מגמה={predicted['trend']}\n"
                f"- תחזית: מינימום={forecast['min']:.2f}, מקסימום={forecast['max']:.2f}, ממוצע={forecast['mean']:.2f}, "
                f"שינוי={forecast['increase_rate']:.2f}%, מגמה={forecast['trend']}"
            )
        else:
            return (
                f"📈 Stock analysis for {company} using model: **{model}**\n"
                f"- Actual: Min={actual['min']:.2f}, Max={actual['max']:.2f}, Mean={actual['mean']:.2f}, "
                f"Increase={actual['increase_rate']:.2f}%, Trend={actual['trend']}\n"
                f"- Predicted: Min={predicted['min']:.2f}, Max={predicted['max']:.2f}, Mean={predicted['mean']:.2f}, "
                f"Increase={predicted['increase_rate']:.2f}%, Trend={predicted['trend']}\n"
                f"- Forecasted: Min={forecast['min']:.2f}, Max={forecast['max']:.2f}, Mean={forecast['mean']:.2f}, "
                f"Increase={forecast['increase_rate']:.2f}%, Trend={forecast['trend']}"
            )





    def _generate_comparison_summary(self, summaries, language="en"):
        if language == "he":
            return "📊 סיכום השוואה בין חברות:\n\n" + "\n\n".join(summaries)
        return "📊 Comparison Summary:\n\n" + "\n\n".join(summaries)

    def _generate_deeper_analysis(self, summary: str, context_info: str = "") -> str:
        """
        Uses Gemini to generate deeper analysis from an existing summary and optional metadata.
        Supports Hebrew or English output based on st.session_state["language"].
        """
        language = st.session_state.get("language", "en")

        if language == "he":
            prompt_text = f"""
        הנך אנליסט פיננסי. כתוב ניתוח מעמיק, ברור, מוסבר היטב ובשפה נגישה לקהל הרחב — עבור המידע הבא.

         תשתמש בידע חיצוני – אל תעבוד רק על בסיס הטקסט שלפניך. שמור על מבנה מאורגן:

        ### מה יש לנו (יסודות):
        - הסבר בסיסי על שם המניה.
        - מהו המודל החוזה.
        - מהי משמעות הערכים: אמת, חיזוי, תחזית.

        ### תבניות, מגמות ותובנות:
        - מה המגמה הכללית?
        - האם יש האטה בקצב הצמיחה?
        - האם המודל שומרני או אופטימי?

        ### סיכונים שיש לקחת בחשבון:
        - אילוצים של המודל.
        - תנודתיות.
        - אזהרת אי-ודאות.

        ### למידה למתחילים:
        - מה זה ניתוח מניות.
        - מה זה מגמות.
        - למה חשוב להבין תנודתיות.
        - הבנת מגבלות המודל.
        - חשיבות פיזור השקעות.

        ### לסיום:
        - מסקנה כוללת.
        - אזהרה לגבי ייעוץ מקצועי.

        — התחלה כאן —
        הקשר כללי:
        {context_info}

        סיכום הנתונים:
        {summary}
        """.strip()
            llm = get_llm_instance()
            chain = ChatPromptTemplate.from_template("{prompt}") | llm | StrOutputParser()
            result = chain.invoke({"prompt": prompt_text})

            # Ensure RTL rendering in Streamlit
            return f'<div dir="rtl" style="text-align: right;">{result}</div>'

        # 🔁 ENGLISH: default
        prompt_text = (
            "You are a financial analyst. Based on the given summary, values, and context, "
            "write a **deeper analysis** including patterns, anomalies, risks, and insights. "
            "Explain trends and make it educational for a beginner-level audience.\n\n"
            f"Context:\n{context_info}\n\nSummary:\n{summary}"
        )

        llm = get_llm_instance()
        chain = ChatPromptTemplate.from_template("{prompt}") | llm | StrOutputParser()
        result = chain.invoke({"prompt": prompt_text})
        return result


