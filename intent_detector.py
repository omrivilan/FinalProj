import re
from typing import List, Dict
from spellchecker import SpellChecker
from difflib import get_close_matches
import streamlit as st
import string
from Levenshtein import distance as levenshtein_distance
import pandas as pd

class IntentDetector:
    """Detects user intent and resolves company or industry names from natural language input.

    This class supports multilingual (Hebrew and English) intent recognition for financial chatbot queries.
    It uses keyword matching, fuzzy logic, and spell correction to determine whether a user wants to:
    - view a graph for a single company,
    - compare multiple companies,
    - analyze an entire industry,
    - or compare multiple sectors.

    It also maps Hebrew names to their English counterparts and prepares company/industry name lists
    for robust name resolution and correction.

    Attributes:
        ticker_mapping (dict): Maps company names to ticker symbols.
        industry_mapping (dict): Maps industry names to identifiers.
        company_names (list): All known company names (English + Hebrew).
        intent_keywords (dict): Trigger words for detecting supported intents.
        keyword_vocab (list): Flat list of all known intent-related keywords.
        spell (SpellChecker): Spell checker configured with custom domain-specific vocabulary.
        hebrew_to_english_company (dict): Mapping from Hebrew to English company names.
        hebrew_to_english_industry (dict): Mapping from Hebrew to English industry names.
    """
    def __init__(self, ticker_mapping: Dict[str, str], industry_mapping: Dict[str, str]):
        """Initializes the IntentDetector with company and industry mappings, keyword lists, and spell correction.
    
        This constructor loads mappings for ticker symbols and industry names, sets up keyword-based intent detection,
        and configures a spell checker with custom vocabulary. It also builds Hebrew-to-English mappings for both
        companies and industries, enabling bilingual support and fuzzy matching for user queries.
    
        Args:
            ticker_mapping (Dict[str, str]): Dictionary mapping company names to ticker symbols.
            industry_mapping (Dict[str, str]): Dictionary mapping industry names to their identifiers.
    
        Attributes:
            ticker_mapping (dict): Maps company names to ticker symbols.
            industry_mapping (dict): Maps industry names to industry identifiers.  
            company_names (list): List of valid company names (English and Hebrew).
            intent_keywords (dict): Keyword triggers for different intent categories.
            keyword_vocab (list): Flattened list of all intent keywords.
            spell (SpellChecker): A spell checker initialized with relevant vocabulary.
            hebrew_to_english_company (dict): Maps Hebrew company names to English.
            hebrew_to_english_industry (dict): Maps Hebrew industry names to English.
        """

        self.ticker_mapping = ticker_mapping
        self.industry_mapping = industry_mapping
        self.company_names = list(ticker_mapping.keys())
        self.intent_keywords = {
            "industry_values": ["actual values", "industry", "companies in industry", "all","בתעשיית","תעשיית","כל החברות ב","חברות ה","כל החברות של"],
            "compare": ["compare", "comparison", "between", "difference between", "versus", "vs", "with", "and", "&", ",", "השוואה", "להשוות את", "מול", "ו", "וגם", "גם", "לעומת", "בנוסף",
                        "תשווה מול", "תשווה עם", "תשווה בין", "השווה בין", "תשווה את", "תשווה ל", "תשווה למול"],
            "graph": ["chart of", "plot of", "graph of", "graph", "plot", "visualize", "chart","הגרף של","גרף של"]
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
        # Extend fuzzy matching with Hebrew names
        hebrew_names = df_companies["HebrewCompanyName"].dropna().astype(str).str.strip().tolist()
        self.company_names += hebrew_names
    def clean_name(self, name: str) -> str:
        """Cleans a company or industry name by stripping punctuation and converting to uppercase.
    
        Args:
            name (str): The raw name string to clean.
    
        Returns:
            str: The cleaned and normalized name.
        """
        return name.strip().translate(str.maketrans('', '', string.punctuation)).upper()

    def normalize_text(self, text: str) -> str:
        """Normalizes input text by lowercasing and standardizing spacing and punctuation.
    
        This method replaces hyphens and colons with spaces, collapses multiple spaces,
        and trims leading/trailing whitespace.
    
        Args:
            text (str): The input text to normalize.
    
        Returns:
            str: A clean, lowercase, space-normalized version of the text.
        """
        text = text.lower()
        text = re.sub(r"[-:]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def spell_correct(self, word: str) -> str:
        """Returns the most likely spelling correction for a given word.
    
        This method uses a preconfigured spell checker with domain-specific vocabulary.
    
        Args:
            word (str): The input word to correct.
    
        Returns:
            str: The corrected word with the highest probability.
        """
        return self.spell.correction(word.lower())

    def suggest_closest_matches(self, input_name: str, candidates: List[str], n=3):
        """Suggests the closest matches to a given input name using Levenshtein distance.
    
        This method compares the cleaned version of the input name against a list of candidate names
        and returns the top N most similar matches.
    
        Args:
            input_name (str): The name to match against candidate options.
            candidates (List[str]): A list of valid names to compare.
            n (int, optional): The maximum number of suggestions to return. Defaults to 3.
    
        Returns:
            List[str]: A list of the closest matching candidate names.
        """
        input_cleaned = self.clean_name(input_name)
        suggestions = sorted(
            candidates,
            key=lambda x: levenshtein_distance(input_cleaned, self.clean_name(x))
        )
        return suggestions[:n]

    def fuzzy_contains_keyword(self, user_input: str, keyword_list: List[str]) -> str:
        """Checks if any corrected word in the input matches a keyword from a given list.
    
        This method applies spell correction to each word in the input and returns
        the first matching keyword found in the provided list.
    
        Args:
            user_input (str): The raw user input text.
            keyword_list (List[str]): A list of keywords to match against.
    
        Returns:
            str or None: The first matched keyword if found; otherwise, None.
        """
        words = user_input.lower().split()
        for word in words:
            corrected = self.spell_correct(word)
            if corrected in keyword_list:
                return corrected
        return None

    def fuzzy_match_company(self, name: str) -> str:
        """Attempts to resolve a company name using fuzzy matching and spelling correction.
    
        This method first tries to match the cleaned name directly to known company names or tickers.
        If no match is found, it applies spell correction and tries again. Hebrew names are mapped
        to their English equivalents if found.
    
        Args:
            name (str): The company name to match.
    
        Returns:
            str or None: The best-matched English company name or ticker if found; otherwise, None.
        """
        name = self.clean_name(name)
        all_names = list(set(self.company_names + list(self.ticker_mapping.values())))
        close_matches = get_close_matches(name, all_names, n=1, cutoff=0.8)
        if close_matches:
            match = close_matches[0]
            if match in self.hebrew_to_english_company:
                return self.hebrew_to_english_company[match]
            return match

        corrected = self.spell.correction(name.lower()).upper()
        close_matches = get_close_matches(corrected, all_names, n=1, cutoff=0.8)
        if close_matches:
            match = close_matches[0]
            if match in self.hebrew_to_english_company:
                return self.hebrew_to_english_company[match]
            return match

        suggestions = self.suggest_closest_matches(name, self.company_names)
        raise ValueError(f"Company '{name}' not found. Did you mean: {', '.join(suggestions)}?")

    def fuzzy_match_industry(self, name: str) -> str:
        """Attempts to match an input string to a known industry using normalization and fuzzy logic.
    
        The method first cleans the input and checks for partial matches within the known industry keys.
        If no direct or partial match is found, it suggests close alternatives using Levenshtein distance.
    
        Args:
            name (str): The user-provided industry name to match.
    
        Returns:
            str: The canonical industry name from the industry mapping.
    
        Raises:
            ValueError: If no close match is found. Suggests the top candidates for correction.
        """
        name = name.lower().strip()
        name = re.sub(r"[-_:,&]", " ", name)
        name = re.sub(r"\s+", " ", name).strip()
        for industry_key, canonical in self.industry_mapping.items():
            if name in industry_key or industry_key in name:
                return canonical

        suggestions = self.suggest_closest_matches(name, list(self.industry_mapping.keys()))
        raise ValueError(f"Industry '{name}' not recognized. Did you mean: {', '.join(suggestions)}?")


    def detect(self, user_input: str, last_intent: str = None) -> dict:
        """Detects the user's intent from their input and extracts relevant entities.
    
        This method performs a multi-stage pipeline to:
        1. Translate Hebrew company/industry names to English.
        2. Normalize the input text.
        3. Detect companies and industries via substring matching and fuzzy logic.
        4. Identify additive queries (e.g., adding companies to a previous graph or comparison).
        5. Select the appropriate intent: 'graph', 'compare', 'industry_values', 'sector_comparison', 'addition', or 'text'.
    
        Args:
            user_input (str): The user's raw input query.
            last_intent (str, optional): The last detected intent to support context-aware decisions. Defaults to None.
    
        Returns:
            dict: A dictionary with an "intent" key and additional keys like "company", "companies", or "industry"/"industries"
                  depending on the detected intent.
        """
        # 1️⃣ Hebrew → English
        tokenized_input = user_input.split()
        keyword_index = None
        #USE-CASE: in companies comparison in hebrew, when 1 is typed correctly and other with typo
        # Find index of the first compare keyword (supports 1- and 2-token matches)
        for i in range(len(tokenized_input)):
            one = tokenized_input[i]
            two = f"{tokenized_input[i]} {tokenized_input[i+1]}" if i + 1 < len(tokenized_input) else None

            if two and two in self.intent_keywords["compare"]:
                keyword_index = i + 2
                break
                
            if one in self.intent_keywords["compare"]:
                keyword_index = i + 1
                break

        # ✅ Only process tokens AFTER the matched compare keyword
        if keyword_index is not None:
            tail_tokens = tokenized_input[keyword_index:]

            for token in tail_tokens:
                token_clean = token.strip(" ,.()")
                # Skip already translated or English tokens
                if token_clean in self.hebrew_to_english_company:
                    continue
                if re.search(r"[A-Za-z]", token_clean):
                    continue

                suggestions = self.suggest_closest_matches(token_clean, list(self.hebrew_to_english_company.keys()))
                if suggestions:
                    best = suggestions[0]
                    replacement = self.hebrew_to_english_company[best]
                    # Replace only exact occurrences
                    user_input = re.sub(rf"\b{re.escape(token_clean)}\b", replacement, user_input)
        
        #END OF USE CASE
        # Now safely replace all known exact company names (won't corrupt industry names)
        for heb, eng in self.hebrew_to_english_company.items():
            user_input = re.sub(rf"\b{re.escape(heb)}\b", eng, user_input)

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

        # 7️⃣ Decide intent purely by count of substring‐matches (fresh logic)
        comps = list(dict.fromkeys(matched_companies))
        inds  = list(dict.fromkeys(matched_industries))
        print(normalized)
        if self.fuzzy_contains_keyword(normalized, self.intent_keywords["compare"]):
            try:
                fuzzy_companies = self.extract_company_names(user_input)
                st.session_state.last_companies = fuzzy_companies
                return {"intent": "compare", "companies": fuzzy_companies}
            except:
                pass
        if len(inds) >= 2:
            st.session_state.last_industries = inds
            return {"intent": "sector_comparison", "industries": inds}
        if len(inds) == 1:
            st.session_state.last_industry = inds[0]
            return {"intent": "industry_values", "industry": inds[0]}
        if len(comps) >= 2:
            st.session_state.last_companies = comps
            return {"intent": "compare", "companies": comps}
        if len(comps) == 1:
            st.session_state.last_company = comps[0]
            st.session_state.last_companies = [comps[0]] 
            return {"intent": "graph", "company": comps[0]}


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
        """Extracts and resolves an industry name from user input based on intent keywords.
    
        This method looks for known industry-related keywords in the input and extracts the
        text following them as the candidate industry name. If no keyword match is found,
        it defaults to using the last word in the input.
    
        Args:
            user_input (str): The full user query text.
    
        Returns:
            str: The matched canonical industry name.
    
        Raises:
            ValueError: If no valid industry match is found.
        """
        for keyword in self.intent_keywords["industry_values"]:
            if keyword in user_input:
                after = user_input.split(keyword)[-1].strip()
                if after:
                    return self.fuzzy_match_industry(after)
        # Fallback: last token
        return self.fuzzy_match_industry(user_input.split()[-1])


    def extract_company_name(self, user_input: str) -> str:
        """Extracts and resolves a company name from user input based on graph-related keywords.
    
        This method searches for known graph-related keywords in the input and uses the text following
        the last keyword as the company name candidate. If no keyword is matched, it defaults to
        the last word in the input.
    
        Args:
            user_input (str): The full user query text.
    
        Returns:
            str: The best-matched company name or ticker.
        """
        for keyword in self.intent_keywords["graph"]:
            if keyword in user_input:
                after = user_input.split(keyword)[-1].strip()
                return self.fuzzy_match_company(after)
        return self.fuzzy_match_company(user_input.split()[-1])

    def extract_company_names(self, user_input: str) -> list:
        """Extracts and resolves multiple company names from a comparison-style user query.
    
        This method detects comparison keywords, replaces common delimiters with commas, and attempts
        to fuzzy match each extracted name. If a name cannot be matched directly, it uses suggestion
        logic to find the closest alternative. Hebrew names are also supported.
    
        Args:
            user_input (str): The user query containing multiple company references.
    
        Returns:
            list: A list of matched company names (in English), deduplicated and ordered.
    
        Raises:
            ValueError: If fewer than two valid company names are found.
        """
        for keyword in self.intent_keywords["compare"]:
            if keyword in user_input:
                user_input = user_input.split(keyword, 1)[-1]
                break
        for delim in self.intent_keywords["compare"]:
            user_input = user_input.replace(delim, ",")
        companies = [c.strip() for c in user_input.split(",") if c.strip()]
        matched = []
        unmatched = []

        for c in companies:
            c_clean = self.clean_name(c)
            try:
                matched_name = self.fuzzy_match_company(c_clean)
                matched.append(matched_name)
            except ValueError:
                unmatched.append(c_clean)

        # 🔧 Rescue unmatched companies using closest suggestions
        for c in unmatched:
            suggestions = self.suggest_closest_matches(c, self.company_names)
            print(c)
            print(suggestions)
            if suggestions:
                match = suggestions[0]
                if match in self.hebrew_to_english_company:
                    matched.append(self.hebrew_to_english_company[match])
                else:
                    matched.append(match)


        if len(matched) < 2:
            raise ValueError("At least two valid companies required.")
        return list(dict.fromkeys(matched))


    def resolve_ticker(self, company_or_ticker: str) -> str:
        """Resolves a company name or ticker symbol to its standardized ticker.
    
        This method checks both forward and reverse mappings to support inputs
        that may be either a company name or a ticker symbol.
    
        Args:
            company_or_ticker (str): The input name or ticker to resolve.
    
        Returns:
            str: The resolved ticker symbol.
    
        Raises:
            ValueError: If the input cannot be matched to any known company or ticker.
        """
        key = company_or_ticker.strip().upper()
        if key in self.ticker_mapping:
            return self.ticker_mapping[key]
        reverse_map = {v: k for k, v in self.ticker_mapping.items()}
        if key in reverse_map:
            return key
        raise ValueError(f"Company or Ticker '{company_or_ticker}' not found in mapping.")
