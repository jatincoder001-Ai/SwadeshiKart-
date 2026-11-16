import os
import pathlib
from flask import Flask, render_template, request, jsonify
import pandas as pd
from search_engine import SmartSearchEngine

app = Flask(__name__)

BASE_DIR = pathlib.Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "products.csv"

CATEGORY_META = {
    "water": {
        "id": "water",
        "name": "Water & Hydration",
        "icon": "💧",
        "tagline": "Pure mineral & packaged drinking water",
        "keywords": ["water", "mineral water", "packaged water", "drinking water", "hydration", "pani", "bottle", "aqua"],
    },
    "chips": {
        "id": "chips",
        "name": "Chips & Namkeen",
        "icon": "🍟",
        "tagline": "Crispy potato chips, wafers & Indian savouries",
        "keywords": ["chips", "potato chips", "wafers", "crisps", "namkeen", "snacks", "sev", "bhujia"],
    },
    "biscuits": {
        "id": "biscuits",
        "name": "Biscuits & Cookies",
        "icon": "🍪",
        "tagline": "Tea-time biscuits, cookies & crackers",
        "keywords": ["biscuits", "cookies", "biscuit", "biscuet", "biscute", "crackers", "bakery", "chai", "chocolate"],
    },
    "cleaning": {
        "id": "cleaning",
        "name": "Cleaning & Household",
        "icon": "🧼",
        "tagline": "Detergents, floor cleaners & dishwash essentials",
        "keywords": ["cleaning", "cleaner", "detergent", "dishwash", "floor cleaner", "toilet cleaner", "soap", "liquid", "surf", "vim", "lizol", "harpic", "colin", "ghadi", "exo"],
    },
    "shampoo": {
        "id": "shampoo",
        "name": "Shampoo & Hair Care",
        "icon": "🧴",
        "tagline": "Shampoos, cleansers & herbal hair care",
        "keywords": ["shampoo", "hair", "hair wash", "hair care", "cleanser", "anti dandruff", "conditioner"],
    },
}

_cache = {
    "mtime": 0,
    "df": None,
    "engine": None,
}


def load_data():
    """Load product data with cached reload on CSV modification and sync search engine."""
    if not CSV_PATH.exists():
        return pd.DataFrame(), None

    mtime = os.path.getmtime(CSV_PATH)
    if _cache["df"] is not None and _cache["mtime"] == mtime and _cache["engine"] is not None:
        return _cache["df"], _cache["engine"]

    df = pd.read_csv(CSV_PATH, dtype=str).fillna("")
    for col in ["type", "category", "keywords"]:
        if col in df.columns:
            df[col] = df[col].str.strip().str.lower()

    engine = SmartSearchEngine(CATEGORY_META)
    engine.fit(df)

    _cache["mtime"] = mtime
    _cache["df"] = df
    _cache["engine"] = engine
    return df, engine


def get_categories_list(df):
    """Return structured list of all categories with item counts."""
    categories = []
    for cat_id, meta in CATEGORY_META.items():
        subset = df[df["category"] == cat_id]
        foreign_count = len(subset[subset["type"] == "foreign"])
        indian_count = len(subset[subset["type"] == "indian"])
        categories.append({
            "id": cat_id,
            "name": meta["name"],
            "icon": meta["icon"],
            "tagline": meta["tagline"],
            "foreign_count": foreign_count,
            "indian_count": indian_count,
            "total_count": len(subset),
        })
    return categories


@app.route("/")
def home():
    df, engine = load_data()
    categories = get_categories_list(df)
    suggestions = engine.get_suggestions("", limit=10) if engine else []
    return render_template("index.html", suggestions=suggestions, categories=categories)


@app.route("/categories")
def categories():
    df, _ = load_data()
    return jsonify(get_categories_list(df))


@app.route("/api/suggestions")
def suggestions():
    """Real-time autocomplete and smart suggestion API with typo-tolerance."""
    q = request.args.get("q", "").strip()
    limit = int(request.args.get("limit", 8))
    _, engine = load_data()
    if not engine:
        return jsonify([])
    return jsonify(engine.get_suggestions(q, limit=limit))


@app.route("/search")
def search():
    """
    Intelligent Hybrid Search Endpoint.
    Supports:
    - ?q=query (hybrid keyword, fuzzy, synonym, semantic search)
    - ?category=cat_id (category browse or filter)
    - Preserves side-by-side Foreign and Indian product pairing
    """
    query = request.args.get("q", "").strip()
    category = request.args.get("category", "").strip().lower()
    df, engine = load_data()

    if not engine:
        return jsonify(
            category=CATEGORY_META.get(category, {"id": category, "name": category.title(), "icon": "🛒", "tagline": ""}),
            stats={"avg_foreign": 0, "avg_indian": 0, "min_indian": 0, "max_foreign": 0, "savings_pct": 0},
            foreign=[],
            indian=[],
            message="",
            total_matches=0,
        )

    # If query is not passed, but category is passed
    results = engine.search(query=query, category_filter=category)
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True)
