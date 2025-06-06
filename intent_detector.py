import re
from typing import List, Dict
from spellchecker import SpellChecker
from difflib import get_close_matches
import streamlit as st
import string
from Levenshtein import distance as levenshtein_distance
import pandas as pd

class IntentDetector:
    def __init__(self, ticker_mapping: Dict[str, str], industry_mapping: Dict[str, str]):
        self.ticker_mapping = ticker_mapping
        self.industry_mapping = industry_mapping
        self.company_names = list(ticker_mapping.keys())
        self.intent_keywords = {
            "industry_values": ["actual values", "industry", "companies in industry", "all"],
            "compare": ["compare", "comparison", "between", "difference between", "versus", "vs", "with", "and", "&", ","],
            "graph": ["chart of", "plot of", "graph of", "graph", "plot", "visualize", "chart"]
        }

        self.keyword_vocab = sum(self.intent_keywords.values(), [])
        self.spell = SpellChecker()

        all_vocab = list(ticker_mapping.keys()) + list(ticker_mapping.values())
        all_vocab += list(industry_mapping.keys()) + self.keyword_vocab
        self.spell.word_frequency.load_words([w.lower() for w in all_vocab])

        # ===== Hebrew-to-English mapping =====
        self.hebrew_to_english_company = {}
        self.hebrew_to_english_industry = {}

        df_companies = pd.read_excel("company_name_to_ticker.xlsx")
        if "HebrewCompanyName" in df_companies.columns:
            for _, row in df_companies.iterrows():
                heb = str(row["HebrewCompanyName"]).strip()
                eng = str(row["CompanyName"]).strip().upper()
                if heb:
                    self.hebrew_to_english_company[heb] = eng

        df_sectors = pd.read_csv("sectors_df.csv")
        if "HebrewIndustryName" in df_sectors.columns:
            for _, row in df_sectors.iterrows():
                heb = str(row["HebrewIndustryName"]).strip()
                eng = str(row["Industry"]).strip()
                if heb:
                    self.hebrew_to_english_industry[heb] = eng

    def clean_name(self, name: str) -> str:
        return name.strip().translate(str.maketrans('', '', string.punctuation)).upper()

    def normalize_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[-:]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def spell_correct(self, word: str) -> str:
        return self.spell.correction(word.lower())

    def suggest_closest_matches(self, input_name: str, candidates: List[str], n=3):
        input_cleaned = self.clean_name(input_name)
        suggestions = sorted(
            candidates,
            key=lambda x: levenshtein_distance(input_cleaned, self.clean_name(x))
        )
        return suggestions[:n]

    def fuzzy_contains_keyword(self, user_input: str, keyword_list: List[str]) -> str:
        words = user_input.lower().split()
        for word in words:
            corrected = self.spell_correct(word)
            if corrected in keyword_list:
                return corrected
        return None

    def fuzzy_match_company(self, name: str) -> str:
        name = self.clean_name(name)
        all_names = list(set(self.company_names + list(self.ticker_mapping.values())))
        close_matches = get_close_matches(name, all_names, n=1, cutoff=0.8)
        if close_matches:
            return close_matches[0]
        corrected = self.spell.correction(name.lower()).upper()
        close_matches = get_close_matches(corrected, all_names, n=1, cutoff=0.8)
        if close_matches:
            return close_matches[0]
        suggestions = self.suggest_closest_matches(name, self.company_names)
        raise ValueError(f"Company '{name}' not found. Did you mean: {', '.join(suggestions)}?")

    def fuzzy_match_industry(self, name: str) -> str:
        name = name.lower().strip()
        name = re.sub(r"[-_:,&]", " ", name)
        name = re.sub(r"\s+", " ", name).strip()
        for industry_key, canonical in self.industry_mapping.items():
            if name in industry_key:
                return canonical
        suggestions = self.suggest_closest_matches(name, list(self.industry_mapping.keys()))
        raise ValueError(f"Industry '{name}' not recognized. Did you mean: {', '.join(suggestions)}?")

    def detect(self, user_input: str, last_intent: str = None) -> dict:
        # 1️⃣ Hebrew → English
        for heb, eng in self.hebrew_to_english_company.items():
            user_input = user_input.replace(heb, eng)
        for heb, eng in self.hebrew_to_english_industry.items():
            user_input = user_input.replace(heb, eng)

        # 2️⃣ Normalize once
        normalized          = self.normalize_text(user_input)     # e.g. "rami levi"
        normalized_no_space = normalized.replace(" ", "")        # e.g. "ramilevi"

        # 3️⃣ Company‐name substring matches (highest priority)
        matched_companies = []
        for comp_name, ticker in self.ticker_mapping.items():
            comp_key = self.normalize_text(comp_name)
            if comp_key in normalized:
                matched_companies.append(comp_name)

        # 4️⃣ Only if no names matched, check raw tickers
        if len(matched_companies) < 1:
            for comp_name, ticker in self.ticker_mapping.items():
                if ticker.lower() in normalized_no_space:
                    matched_companies.append(comp_name)

        # 5️⃣ Industry substring matches
        matched_industries = [
            canon for key, canon in self.industry_mapping.items()
            if self.normalize_text(key) in normalized
        ]

        # 6️⃣ Additive logic → check if new prompt is related to the previous one(wants to add company/industry to the prior prompt)
        additive_keywords = {
            "add", "also", "with", "vs", "versus", "too", "as well",
            "along", "plus", "include", "including", "alongside", "next to",
            "another", "more", "combine", "in addition","תשווה מול", "תשווה עם", "תוסיף","גם את","גם כן","לצד","תשווה ל","תשווה למול"
        }
        is_additive = any(kw in normalized for kw in additive_keywords)

        # a) company‐addition: if last was graph/compare and user used an additive keyword
        if last_intent in {"graph", "compare"} and is_additive and matched_companies:
            prev        = st.session_state.get("last_company")
            prev_list   = st.session_state.get("last_companies", [])
            # bring in single‐company state
            if prev and prev not in matched_companies:
                matched_companies.append(prev)
            # bring in multi‐company state
            for pc in prev_list:
                if pc not in matched_companies:
                    matched_companies.append(pc)
            unique = list(dict.fromkeys(matched_companies))
            # update state and return “addition”
            st.session_state.last_companies = unique
            return {"intent": "addition", "companies": unique}

        # b) industry‐addition: if last was industry_values/sector_comparison
        if last_intent in {"industry_values", "sector_comparison"} and is_additive and matched_industries:
            prev_inds = st.session_state.get("last_industries", [])
            prev_ind  = st.session_state.get("last_industry")
            if prev_ind and prev_ind not in matched_industries:
                matched_industries.append(prev_ind)
            for pi in prev_inds:
                if pi not in matched_industries:
                    matched_industries.append(pi)
            unique_i = list(dict.fromkeys(matched_industries))
            st.session_state.last_industries = unique_i
            return {"intent": "addition", "industries": unique_i}

        # # 7️⃣ Decide intent purely by count of substring‐matches (fresh logic)
        # comps = list(dict.fromkeys(matched_companies))
        # inds  = list(dict.fromkeys(matched_industries))

        # if len(comps) >= 2:
        #     st.session_state.last_companies = comps
        #     return {"intent": "compare", "companies": comps}
        # if len(comps) == 1:
        #     st.session_state.last_company = comps[0]
        #     st.session_state.last_companies = [comps[0]] 
        #     return {"intent": "graph", "company": comps[0]}
        # if len(inds) >= 2:
        #     st.session_state.last_industries = inds
        #     return {"intent": "sector_comparison", "industries": inds}
        # if len(inds) == 1:
        #     st.session_state.last_industry = inds[0]
        #     return {"intent": "industry_values", "industry": inds[0]}

        # 8️⃣ Fallback to fuzzy/keyword logic
        if self.fuzzy_contains_keyword(normalized, self.intent_keywords["compare"]):
            return {
                "intent":    "compare",
                "companies": self.extract_company_names(user_input)
            }
        if self.fuzzy_contains_keyword(normalized, self.intent_keywords["graph"]):
            return {
                "intent": "graph",
                "company": self.extract_company_name(user_input)
            }
        if self.fuzzy_contains_keyword(normalized, self.intent_keywords["industry_values"]):
            return {
                "intent":  "industry_values",
                "industry": self.extract_industry_name(user_input)
            }

        # 9️⃣ Default to free‐text
        return {"intent": "text"}



    def extract_industry_name(self, user_input: str) -> str:
        for keyword in self.intent_keywords["industry_values"]:
            if keyword in user_input:
                after = user_input.split(keyword)[-1].strip()
                return self.fuzzy_match_industry(after)
        raise ValueError("Could not recognize the industry name.")

    def extract_company_name(self, user_input: str) -> str:
        for keyword in self.intent_keywords["graph"]:
            if keyword in user_input:
                after = user_input.split(keyword)[-1].strip()
                return self.fuzzy_match_company(after)
        return self.fuzzy_match_company(user_input.split()[-1])

    def extract_company_names(self, user_input: str) -> list:
        for keyword in self.intent_keywords["compare"]:
            if keyword in user_input:
                user_input = user_input.split(keyword, 1)[-1]
                break
        for delim in self.intent_keywords["compare"]:
            user_input = user_input.replace(delim, ",")
        companies = [c.strip() for c in user_input.split(",") if c.strip()]
        matched = []
        for c in companies:
            c = self.clean_name(c)
            try:
                matched_name = self.fuzzy_match_company(c)
                matched.append(matched_name)
            except ValueError:
                continue
        if len(matched) < 2:
            raise ValueError("At least two valid companies required.")
        return list(set(matched))

    def resolve_ticker(self, company_or_ticker: str) -> str:
        key = company_or_ticker.strip().upper()
        if key in self.ticker_mapping:
            return self.ticker_mapping[key]
        reverse_map = {v: k for k, v in self.ticker_mapping.items()}
        if key in reverse_map:
            return key
        raise ValueError(f"Company or Ticker '{company_or_ticker}' not found in mapping.")
