"""
Smart Search Engine for SwadeshiKart
====================================
Hybrid search system combining:
1. Exact & Keyword matching (BM25 / TF-IDF token overlap)
2. Fuzzy matching (handles spelling mistakes via sequence matching & edit distance)
3. Synonym matching (domain thesaurus & semantic term expansion)
4. Semantic search using text embeddings (Dense Latent Semantic Analysis / TruncatedSVD on TF-IDF space)
5. Product pairing preservation (Foreign <-> Indian alternative bi-directional linkage)
6. Dynamic real-time auto-suggestions with typo tolerance
"""

import re
import difflib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


SYNONYMS_MAP = {
    # Water & Hydration
    "water": ["mineral water", "packaged water", "drinking water", "hydration", "bottle", "pani", "aqua", "still water", "sparkling water"],
    "pani": ["water", "drinking water", "mineral water", "packaged water"],
    "sparkling": ["soda", "carbonated", "fizz", "bubble", "perrier", "san pellegrino"],
    "mineral": ["natural mineral", "spring water", "artesian", "himalayan", "evian"],
    
    # Chips & Snacks
    "chips": ["potato chips", "wafers", "crisps", "namkeen", "snacks", "sev", "bhujia", "nachos", "tortilla"],
    "wafers": ["chips", "potato wafers", "crisps", "salted wafers"],
    "crisps": ["chips", "potato crisps", "wafers", "pringles"],
    "namkeen": ["bhujia", "sev", "snacks", "aloo bhujia", "ratlami sev", "savouries"],
    "snack": ["chips", "namkeen", "wafers", "crisps", "sev", "bhujia"],
    "snacks": ["chips", "namkeen", "wafers", "crisps", "sev", "bhujia"],
    
    # Biscuits & Cookies
    "biscuit": ["biscuits", "cookies", "crackers", "bakery", "tea time", "chai"],
    "biscuits": ["biscuit", "cookies", "crackers", "bakery", "tea time", "chai"],
    "cookie": ["cookies", "biscuits", "biscuit", "bakery"],
    "cookies": ["cookie", "biscuits", "biscuit", "bakery"],
    "chocolate": ["choco", "cocoa", "dark chocolate", "choco fills", "chocolate chip", "chocolate biscuits", "dark fantasy", "hide seek", "tim tam", "milano"],
    "choco": ["chocolate", "cocoa", "dark chocolate", "choco fills", "chocolate chip"],
    "cracker": ["crackers", "salted crackers", "monaco", "bakery"],
    "crackers": ["cracker", "salted crackers", "monaco", "bakery"],
    
    # Cleaning & Household
    "cleaning": ["cleaner", "detergent", "dishwash", "floor cleaner", "toilet cleaner", "soap", "liquid", "surf excel", "ghadi", "lizol", "nimyle", "vim", "exo", "harpic", "colin"],
    "cleaner": ["cleaning", "detergent", "floor cleaner", "surface cleaner", "dishwash", "toilet cleaner", "glass cleaner"],
    "detergent": ["cleaning", "liquid detergent", "washing powder", "laundry", "surf excel", "ghadi", "clothes cleaner"],
    "dishwash": ["cleaning", "dish cleaner", "utensils", "bartan", "vim", "exo", "dishwash gel"],
    "floor cleaner": ["cleaning", "disinfectant", "surface cleaner", "lizol", "nimyle", "germ kill"],
    "toilet cleaner": ["cleaning", "bathroom cleaner", "harpic", "sanifresh", "disinfectant"],
    "glass cleaner": ["cleaning", "surface cleaner", "colin", "spray", "window cleaner"],
    
    # Shampoo & Hair Care
    "shampoo": ["hair wash", "hair cleanser", "cleanser", "hair care", "anti dandruff", "hair fall", "scalp", "conditioner"],
    "shampoos": ["shampoo", "hair wash", "hair cleanser", "hair care"],
    "hair": ["shampoo", "hair wash", "hair cleanser", "hair care", "scalp", "dandruff", "hair fall"],
    "cleanser": ["shampoo", "hair wash", "hair cleanser"],
    "dandruff": ["anti dandruff", "shampoo", "scalp care", "head shoulders", "meera"],
    
    # Types / Origins
    "foreign": ["imported", "international", "western", "expensive"],
    "indian": ["swadeshi", "desi", "local", "domestic", "india", "bharat"],
}


COMMON_TYPOS = {
    "watr": "water",
    "wtr": "water",
    "paani": "pani",
    "botle": "bottle",
    "choclate": "chocolate",
    "choclat": "chocolate",
    "choc": "chocolate",
    "biscut": "biscuit",
    "biscute": "biscuit",
    "biscuet": "biscuit",
    "biskit": "biscuit",
    "biscit": "biscuit",
    "cooki": "cookie",
    "cookey": "cookie",
    "chip": "chips",
    "chipss": "chips",
    "wafur": "wafers",
    "wafer": "wafers",
    "namkin": "namkeen",
    "namken": "namkeen",
    "detergnt": "detergent",
    "deturgent": "detergent",
    "cleen": "clean",
    "cleenar": "cleaner",
    "dishwsh": "dishwash",
    "florr": "floor",
    "shampo": "shampoo",
    "shmp": "shampoo",
    "shamphoo": "shampoo",
    "dandruf": "dandruff",
    "parly": "parle",
    "parleg": "parle-g",
    "pringls": "pringles",
    "pringl": "pringles",
    "oreoo": "oreo",
    "himalyan": "himalayan",
    "layss": "lays",
    "lizl": "lizol",
    "hrpic": "harpic",
    "ghdi": "ghadi",
    "exoo": "exo",
    "colln": "colin",
    "patnjali": "patanjali",
}


def normalize_text(text: str) -> str:
    """Normalize text by lowering, removing punctuation, and stripping extra whitespace."""
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^\w\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class SmartSearchEngine:
    def __init__(self, category_meta: dict):
        self.category_meta = category_meta
        self.df = None
        self.products = []
        self.product_lookup = {}
        self.pair_lookup = {}
        self.vocabulary_terms = set()
        
        # Semantic embedding models
        self.tfidf = None
        self.svd = None
        self.dense_embeddings = None
        self.corpus_docs = []

    def fit(self, df: pd.DataFrame):
        """Build search indexes and dense semantic embeddings from product dataframe."""
        if df.empty:
            return
        
        self.df = df.copy()
        self.products = df.to_dict("records")
        self.product_lookup = {str(p.get("id")): p for p in self.products}
        
        # Pair lookup
        self.pair_lookup = {}
        for p in self.products:
            pid = str(p.get("id"))
            pair_id = str(p.get("pair_id", "")).strip()
            if pair_id and pair_id in self.product_lookup:
                self.pair_lookup[pid] = pair_id
        
        # Build vocabulary dictionary for typo correction and suggestion matching
        self.vocabulary_terms = set()
        for p in self.products:
            for field in [p.get("name", ""), p.get("brand", ""), p.get("category", ""), p.get("country", "")]:
                for word in normalize_text(field).split():
                    if len(word) > 2:
                        self.vocabulary_terms.add(word)
            keywords = p.get("keywords", "").replace(";", " ")
            for word in normalize_text(keywords).split():
                if len(word) > 2:
                    self.vocabulary_terms.add(word)
        
        for cat_id, meta in self.category_meta.items():
            self.vocabulary_terms.add(cat_id)
            for word in normalize_text(meta.get("name", "")).split():
                if len(word) > 2:
                    self.vocabulary_terms.add(word)
            for kw in meta.get("keywords", []):
                for word in normalize_text(kw).split():
                    if len(word) > 2:
                        self.vocabulary_terms.add(word)

        # Build Semantic Corpus for Dense Vector Embeddings (LSA)
        self.corpus_docs = []
        for p in self.products:
            name = p.get("name", "")
            brand = p.get("brand", "")
            cat = p.get("category", "")
            cat_name = self.category_meta.get(cat, {}).get("name", "")
            country = p.get("country", "")
            keywords = p.get("keywords", "").replace(";", " ")
            ptype = p.get("type", "")
            
            # Weighted semantic representation: name and brand repeated for emphasis
            doc = f"{name} {name} {brand} {brand} {cat_name} {cat} {keywords} {country} {ptype}"
            self.corpus_docs.append(normalize_text(doc))
        
        # Fit TF-IDF Vectorizer
        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            token_pattern=r"(?u)\b[\w-]+\b"
        )
        tfidf_matrix = self.tfidf.fit_transform(self.corpus_docs)
        
        # Fit TruncatedSVD for Dense Latent Semantic Embeddings
        # Dimensionality: min(num_samples - 1, 20)
        n_components = min(tfidf_matrix.shape[0] - 1, 24)
        if n_components >= 2:
            self.svd = TruncatedSVD(n_components=n_components, random_state=42)
            self.dense_embeddings = self.svd.fit_transform(tfidf_matrix)
            # L2 normalize embeddings for cosine similarity
            norms = np.linalg.norm(self.dense_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.dense_embeddings = self.dense_embeddings / norms
        else:
            self.svd = None
            self.dense_embeddings = tfidf_matrix.toarray()

    def correct_spelling(self, query: str) -> str:
        """Correct misspelled query terms using known typos, dictionary, and vocabulary."""
        tokens = normalize_text(query).split()
        if not tokens:
            return query
        
        corrected_tokens = []
        changed = False
        
        for token in tokens:
            # Check known common typo map first
            if token in COMMON_TYPOS:
                corrected_tokens.append(COMMON_TYPOS[token])
                changed = True
                continue
            
            # If word is in vocabulary, keep it
            if token in self.vocabulary_terms or len(token) <= 2:
                corrected_tokens.append(token)
                continue
            
            # Fuzzy match with vocabulary terms
            matches = difflib.get_close_matches(token, list(self.vocabulary_terms), n=1, cutoff=0.72)
            if matches:
                corrected_tokens.append(matches[0])
                changed = True
            else:
                corrected_tokens.append(token)
                
        corrected = " ".join(corrected_tokens)
        return corrected if changed else query

    def expand_synonyms(self, query: str) -> list[str]:
        """Expand query with domain synonyms."""
        norm_q = normalize_text(query)
        expansions = [norm_q]
        
        tokens = norm_q.split()
        for token in tokens:
            if token in SYNONYMS_MAP:
                expansions.extend(SYNONYMS_MAP[token])
        
        # Check phrase matches
        for phrase, syns in SYNONYMS_MAP.items():
            if phrase in norm_q and phrase not in tokens:
                expansions.extend(syns)
                
        return list(dict.fromkeys(expansions))

    def compute_fuzzy_score(self, query: str, product: dict) -> float:
        """Compute fuzzy similarity between query and product fields."""
        q_norm = normalize_text(query)
        if not q_norm:
            return 0.0
            
        p_name = normalize_text(product.get("name", ""))
        p_brand = normalize_text(product.get("brand", ""))
        p_cat = normalize_text(product.get("category", ""))
        p_keywords = normalize_text(product.get("keywords", "").replace(";", " "))
        
        # Direct phrase ratios
        r_name = difflib.SequenceMatcher(None, q_norm, p_name).ratio()
        r_brand = difflib.SequenceMatcher(None, q_norm, p_brand).ratio()
        
        # Token-level best match
        q_tokens = q_norm.split()
        target_tokens = (p_name + " " + p_brand + " " + p_cat + " " + p_keywords).split()
        
        token_scores = []
        for qt in q_tokens:
            if len(qt) <= 2:
                continue
            best_t = 0.0
            for tt in target_tokens:
                if len(tt) <= 2:
                    continue
                score = difflib.SequenceMatcher(None, qt, tt).ratio()
                if score > best_t:
                    best_t = score
            token_scores.append(best_t)
            
        avg_token_score = float(np.mean(token_scores)) if token_scores else 0.0
        return max(r_name, r_brand * 1.05, avg_token_score)

    def compute_semantic_score(self, query: str) -> np.ndarray:
        """Compute cosine similarity between query and product dense embeddings."""
        if self.tfidf is None or self.dense_embeddings is None:
            return np.zeros(len(self.products))
            
        q_norm = normalize_text(query)
        q_vec = self.tfidf.transform([q_norm])
        
        if self.svd is not None:
            q_emb = self.svd.transform(q_vec)
            norm = np.linalg.norm(q_emb)
            if norm > 0:
                q_emb = q_emb / norm
            scores = cosine_similarity(q_emb, self.dense_embeddings)[0]
        else:
            q_dense = q_vec.toarray()
            scores = cosine_similarity(q_dense, self.dense_embeddings)[0]
            
        # Ensure scores are within [0, 1]
        scores = np.clip(scores, 0.0, 1.0)
        return scores

    def search(self, query: str, category_filter: str = "") -> dict:
        """
        Execute hybrid search combining:
        - Exact & Brand matching
        - Keyword & Token overlap
        - Fuzzy typo matching
        - Synonym expansion
        - Semantic embedding similarity
        - Automatic pairing between Foreign and Indian products
        """
        if self.df is None or self.df.empty:
            return {"foreign": [], "indian": [], "stats": {}, "category": {}, "query": query}
            
        raw_query = query.strip()
        corrected_query = self.correct_spelling(raw_query) if raw_query else ""
        norm_query = normalize_text(corrected_query if corrected_query else raw_query)
        
        # Check if query is explicitly a known category
        resolved_category = category_filter.strip().lower()
        if not resolved_category and norm_query:
            for cat_id, meta in self.category_meta.items():
                if norm_query == cat_id or norm_query == normalize_text(meta.get("name", "")):
                    resolved_category = cat_id
                    break
                if norm_query in [normalize_text(k) for k in meta.get("keywords", [])]:
                    resolved_category = cat_id
                    break

        # Calculate scores across all products
        n_prods = len(self.products)
        scores = np.zeros(n_prods)
        match_details = [{} for _ in range(n_prods)]
        
        # If no query and category filter given, load full category
        if not norm_query and resolved_category:
            for i, p in enumerate(self.products):
                if p.get("category") == resolved_category:
                    scores[i] = 1.0
        elif norm_query:
            # 1. Semantic embeddings score
            sem_scores = self.compute_semantic_score(norm_query)
            
            # 2. Synonym expansion list
            expanded_terms = self.expand_synonyms(norm_query)
            q_tokens = set(norm_query.split())
            
            for i, p in enumerate(self.products):
                p_name = normalize_text(p.get("name", ""))
                p_brand = normalize_text(p.get("brand", ""))
                p_cat = normalize_text(p.get("category", ""))
                p_keywords = normalize_text(p.get("keywords", "").replace(";", " "))
                p_all = f"{p_name} {p_brand} {p_cat} {p_keywords}"
                p_tokens = set(p_all.split())
                
                # --- A. Exact Match Scoring ---
                exact_score = 0.0
                if norm_query == p_name:
                    exact_score = 1.0
                elif norm_query == p_brand:
                    exact_score = 0.95
                elif norm_query in p_name:
                    exact_score = 0.85
                elif norm_query in p_brand:
                    exact_score = 0.80
                elif any(norm_query == kw for kw in p_keywords.split()):
                    exact_score = 0.75
                
                # --- B. Keyword / Token Overlap Scoring ---
                keyword_score = 0.0
                if q_tokens:
                    intersect = q_tokens.intersection(p_tokens)
                    keyword_score = len(intersect) / len(q_tokens)
                    # Extra boost if brand or primary product name token matches
                    if p_brand in q_tokens:
                        keyword_score = max(keyword_score, 0.9)
                
                # --- C. Fuzzy Matching Scoring ---
                fuzzy_score = self.compute_fuzzy_score(norm_query, p)
                
                # --- D. Synonym Scoring ---
                synonym_score = 0.0
                for syn in expanded_terms:
                    syn_tokens = set(syn.split())
                    if syn == p_cat or syn == p_brand:
                        synonym_score = max(synonym_score, 0.85)
                    elif syn_tokens.issubset(p_tokens):
                        synonym_score = max(synonym_score, 0.75)
                    elif syn in p_all:
                        synonym_score = max(synonym_score, 0.65)
                
                # --- E. Semantic Embedding Score ---
                semantic_score = float(sem_scores[i])
                
                # --- F. Category Affinity Bonus ---
                cat_bonus = 0.0
                if resolved_category and p.get("category") == resolved_category:
                    cat_bonus = 0.20
                
                # Combined Hybrid Score
                # Prioritize exact/keyword matches while rewarding semantic and fuzzy relevance
                if exact_score >= 0.85:
                    final_score = 0.70 * exact_score + 0.15 * keyword_score + 0.15 * semantic_score
                else:
                    final_score = (
                        0.25 * exact_score +
                        0.30 * keyword_score +
                        0.15 * fuzzy_score +
                        0.15 * synonym_score +
                        0.15 * semantic_score +
                        cat_bonus
                    )
                
                scores[i] = final_score
                match_details[i] = {
                    "exact": exact_score,
                    "keyword": keyword_score,
                    "fuzzy": fuzzy_score,
                    "synonym": synonym_score,
                    "semantic": semantic_score,
                    "final": final_score,
                }

        # Filter out irrelevant results below threshold
        # Minimum threshold: 0.20 when searching with query, 0.50 when browsing a category
        THRESHOLD = 0.20 if norm_query else 0.50
        
        matched_foreign = []
        matched_indian = []
        
        for i, p in enumerate(self.products):
            score = scores[i]
            if score >= THRESHOLD:
                item_copy = dict(p)
                item_copy["_score"] = round(float(score), 4)
                if p.get("type") == "foreign":
                    matched_foreign.append(item_copy)
                else:
                    matched_indian.append(item_copy)
                    
        # Sort matched lists descending by score
        matched_foreign.sort(key=lambda x: x["_score"], reverse=True)
        matched_indian.sort(key=lambda x: x["_score"], reverse=True)
        
        # --- Bi-Directional Alternative Pairing Preservation ---
        # 1. If Foreign products matched, ensure their Indian alternatives are included and prioritized
        indian_ids_present = {str(item.get("id")) for item in matched_indian}
        indian_pair_boosts = []
        
        for f_item in matched_foreign:
            f_id = str(f_item.get("id"))
            paired_indian_id = self.pair_lookup.get(f_id)
            if paired_indian_id and paired_indian_id in self.product_lookup:
                if paired_indian_id not in indian_ids_present:
                    paired_prod = dict(self.product_lookup[paired_indian_id])
                    # Give paired alternative score derived from matched product
                    paired_prod["_score"] = round(f_item["_score"] * 0.95, 4)
                    paired_prod["_paired_match"] = True
                    indian_pair_boosts.append(paired_prod)
                    indian_ids_present.add(paired_indian_id)
        
        matched_indian.extend(indian_pair_boosts)
        matched_indian.sort(key=lambda x: x["_score"], reverse=True)

        # 2. If Indian products matched, ensure their Foreign counterparts are included and prioritized
        foreign_ids_present = {str(item.get("id")) for item in matched_foreign}
        foreign_pair_boosts = []
        
        for i_item in matched_indian:
            i_id = str(i_item.get("id"))
            paired_foreign_id = self.pair_lookup.get(i_id)
            if paired_foreign_id and paired_foreign_id in self.product_lookup:
                if paired_foreign_id not in foreign_ids_present:
                    paired_prod = dict(self.product_lookup[paired_foreign_id])
                    paired_prod["_score"] = round(i_item["_score"] * 0.95, 4)
                    paired_prod["_paired_match"] = True
                    foreign_pair_boosts.append(paired_prod)
                    foreign_ids_present.add(paired_foreign_id)
                    
        matched_foreign.extend(foreign_pair_boosts)
        matched_foreign.sort(key=lambda x: x["_score"], reverse=True)

        # Detect primary category for banner metadata
        primary_cat_id = resolved_category
        if not primary_cat_id and (matched_foreign or matched_indian):
            top_cat = (matched_foreign[0] if matched_foreign else matched_indian[0]).get("category", "")
            primary_cat_id = top_cat

        category_meta = self.category_meta.get(primary_cat_id, {
            "id": primary_cat_id or "search",
            "name": raw_query.title() if raw_query else "Products",
            "icon": "⚡",
            "tagline": f"Showing Indian alternatives and foreign comparisons for '{raw_query}'" if raw_query else "Smart grocery alternatives",
        })

        # Calculate live price savings statistics
        def get_prices(items):
            prices = []
            for item in items:
                try:
                    val = float(str(item.get("price", "0")).replace("₹", "").replace(",", "").strip())
                    prices.append(val)
                except ValueError:
                    pass
            return prices

        f_prices = get_prices(matched_foreign)
        i_prices = get_prices(matched_indian)

        avg_f = round(float(np.mean(f_prices)), 1) if f_prices else 0
        avg_i = round(float(np.mean(i_prices)), 1) if i_prices else 0
        min_i = int(min(i_prices)) if i_prices else 0
        max_f = int(max(f_prices)) if f_prices else 0

        savings_pct = 0
        if avg_f > 0 and avg_i > 0:
            savings_pct = max(int(round((1 - (avg_i / avg_f)) * 100)), 0)

        stats = {
            "avg_foreign": avg_f,
            "avg_indian": avg_i,
            "min_indian": min_i,
            "max_foreign": max_f,
            "savings_pct": savings_pct,
        }

        # User feedback message if query had spelling correction
        message = ""
        if corrected_query and corrected_query.lower() != raw_query.lower():
            message = f"Showing results for \"{corrected_query}\" (auto-corrected from \"{raw_query}\")"

        return {
            "query": raw_query,
            "corrected_query": corrected_query,
            "message": message,
            "category": category_meta,
            "stats": stats,
            "foreign": matched_foreign,
            "indian": matched_indian,
            "total_matches": len(matched_foreign) + len(matched_indian),
        }

    def get_suggestions(self, query: str, limit: int = 8) -> list[dict]:
        """Generate smart suggestions for typing input including categories, products, and brand matches."""
        raw_query = query.strip()
        if not raw_query:
            # Return primary categories when search box is empty
            primary = []
            for cat_id, meta in self.category_meta.items():
                primary.append({
                    "label": meta["name"],
                    "query": meta["name"],
                    "category": cat_id,
                    "categoryName": meta["name"],
                    "icon": meta.get("icon", "🛒"),
                    "type": "Category",
                    "isPrimary": True,
                })
            return primary

        norm_q = normalize_text(raw_query)
        corrected_query = self.correct_spelling(raw_query)
        norm_corr = normalize_text(corrected_query)
        suggestions = []
        seen = set()

        # Check if query is an exact prefix of common category keywords
        prefix_completions = []
        for cat_id, meta in self.category_meta.items():
            for kw in meta.get("keywords", []):
                norm_kw = normalize_text(kw)
                if norm_kw.startswith(norm_q) and norm_kw != norm_q:
                    prefix_completions.append((kw, cat_id, meta))

        # 1. Category matching (direct prefix or contains)
        for cat_id, meta in self.category_meta.items():
            name = meta["name"]
            cat_keywords = [normalize_text(k) for k in meta.get("keywords", [])]
            match = False
            is_prefix = False
            
            if name.lower().startswith(norm_q) or any(k.startswith(norm_q) for k in cat_keywords):
                match = True
                is_prefix = True
            elif norm_q in name.lower() or norm_corr in name.lower():
                match = True
            elif any(norm_q in kw or norm_corr in kw for kw in cat_keywords):
                match = True
            elif difflib.SequenceMatcher(None, norm_q, cat_id).ratio() > 0.75:
                match = True

            if match and name.lower() not in seen:
                sug_item = {
                    "label": name,
                    "query": name,
                    "category": cat_id,
                    "categoryName": name,
                    "icon": meta.get("icon", "🛒"),
                    "type": "Category",
                    "isPrimary": True,
                    "_is_prefix": is_prefix,
                }
                suggestions.append(sug_item)
                seen.add(name.lower())

        # Sort prefix matching categories to the top
        suggestions.sort(key=lambda x: 0 if x.get("_is_prefix") else 1)

        # 2. Spelling correction suggestion (only if query is not a clean prefix of a category)
        if not prefix_completions and norm_corr != norm_q and len(norm_corr) > 2:
            if corrected_query.lower() not in seen:
                suggestions.insert(0, {
                    "label": f"Did you mean: {corrected_query}?",
                    "query": corrected_query,
                    "category": "",
                    "categoryName": "Spell Check",
                    "icon": "✨",
                    "type": "Correction",
                    "isPrimary": False,
                })
                seen.add(corrected_query.lower())

        # 3. Product & Brand Matches
        scored_products = []
        for p in self.products:
            p_name = p.get("name", "")
            p_brand = p.get("brand", "")
            p_cat = p.get("category", "")
            cat_meta = self.category_meta.get(p_cat, {})
            icon = cat_meta.get("icon", "🛒")
            ptype = p.get("type", "")
            badge = "🇮🇳 Indian Brand" if ptype == "indian" else "⚠️ Foreign Brand"
            
            # Match score
            norm_name = normalize_text(p_name)
            norm_brand = normalize_text(p_brand)
            
            score = 0.0
            if norm_brand.startswith(norm_q) or norm_name.startswith(norm_q):
                score = 0.95
            elif norm_q == norm_brand or norm_corr == norm_brand:
                score = 0.92
            elif norm_q in norm_brand or norm_q in norm_name:
                score = 0.85
            elif norm_corr in norm_brand or norm_corr in norm_name:
                score = 0.80
            else:
                fuzzy = max(
                    difflib.SequenceMatcher(None, norm_q, norm_brand).ratio(),
                    difflib.SequenceMatcher(None, norm_q, norm_name).ratio()
                )
                if fuzzy > 0.68:
                    score = fuzzy * 0.75

            if score > 0.50:
                scored_products.append((score, {
                    "label": p_name,
                    "query": p_name,
                    "category": p_cat,
                    "categoryName": cat_meta.get("name", p_cat.title()),
                    "icon": icon,
                    "type": badge,
                    "isPrimary": False,
                }))

        scored_products.sort(key=lambda x: x[0], reverse=True)
        for _, item in scored_products:
            if item["label"].lower() not in seen:
                suggestions.append(item)
                seen.add(item["label"].lower())
            if len(suggestions) >= limit:
                break

        return suggestions[:limit]
