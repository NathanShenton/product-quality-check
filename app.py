import streamlit as st
import pandas as pd
import json
import requests
import base64
import plotly.graph_objects as go
from streamlit_cropper import st_cropper
from PIL import Image
import io
from openai import OpenAI

# ── Project modules ────────────────────────────────────────────────────────────
from sidebar import render_sidebar
from style import inject_css

from prompts.prompts import PROMPT_OPTIONS

from prompts.artwork_processing_ingredients import process_artwork
from prompts.artwork_processing_directions import process_artwork_directions
from prompts.artwork_processing_warnings_advisory import process_artwork_warnings_advisory
from prompts.artwork_processing_packsize_nutrition import (
    process_artwork_packsize,
    process_artwork_nutrition,
)
from prompts.artwork_processing_supplier_addresses import process_artwork as process_artwork_suppliers

from prompts.competitor_match import (
    parse_sku,
    top_candidates,
    build_match_prompt,
    # load_competitor_db,  # available if you need it later
)

import prompts.banningredients as banningredients
from prompts.banningredients import build_banned_prompt  # keep if you need it elsewhere

from prompts.hfss import (
    build_pass_1_prompt,
    build_pass_2_prompt,
    build_pass_3_prompt,
    build_pass_4_prompt
)

from prompts.green_claims import (
    load_green_claims_db,
    screen_candidates,
    build_green_claims_prompt,
    LANG_TO_COL
)

# at the top of the file (if not already)
import numpy as np
import pandas as pd


# --- Text normalisation helpers (HTML → plain; lowercase; tidy whitespace)
import html, re, unicodedata
def strip_html(s: str) -> str:
    s = html.unescape(str(s or ""))
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def norm_basic(s: str) -> str:
    s = strip_html(s)
    s = ''.join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return s.lower()

# ─── Streamlit page config ─────────────────────────────────────────────────────
st.set_page_config(page_title="Flexible AI Product Data Checker", layout="wide")
inject_css()
render_sidebar()

#############################
#   Helper: Approx. Tokens  #
#############################
def approximate_tokens(text: str) -> int:
    """Approximate token count from text length."""
    return max(1, len(text) // 4)

#############################
#   Cost Estimation         #
#############################
def estimate_cost(model: str, df: pd.DataFrame, user_prompt: str, cols_to_use: list) -> float:
    """
    Estimate cost based on the chosen model, #rows, approximate tokens in prompt + row data.
    Costs are per 1k tokens (input, output).
    """
    model_costs_per_1k = {
        "gpt-3.5-turbo": (0.0005, 0.002),
        "gpt-4.1-mini":  (0.0004, 0.0016),
        "gpt-4.1-nano":  (0.0001, 0.0004),
        "gpt-4o-mini":   (0.00015, 0.0006),
        "gpt-4o":        (0.005,  0.015),  # baseline as of 2024-05
        "gpt-4-turbo":   (0.01,   0.03)
    }
    cost_in, cost_out = model_costs_per_1k.get(model, (0.001, 0.003))
    total_input_tokens = 0
    total_output_tokens = 0

    for _, row in df.iterrows():
        system_tokens = 30
        row_dict = {c: row.get(c, "") for c in cols_to_use}
        row_json_str = json.dumps(row_dict, ensure_ascii=False)
        prompt_tokens = approximate_tokens(user_prompt) + approximate_tokens(row_json_str)
        total_input_tokens += (system_tokens + prompt_tokens)
        # Rough output per row
        total_output_tokens += 100

    input_ktokens = total_input_tokens / 1000
    output_ktokens = total_output_tokens / 1000
    return (input_ktokens * cost_in) + (output_ktokens * cost_out)

# ─── JSON Cleaner ──────────────────────────────────────────────────────────────
def clean_gpt_json_block(text: str) -> str:
    """
    Strip ``` wrappers and any preamble before the first '{' so json.loads() doesn't choke.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"```$", "", text.strip(), flags=re.IGNORECASE)
    json_start = text.find("{")
    if json_start != -1:
        text = text[json_start:]
    return text.strip()

# ─── Arrow-safe DataFrame helper (with unique-columns + dtype guards) ─────────
from pandas.api.types import (
    is_datetime64_any_dtype, is_object_dtype, is_bool_dtype, is_integer_dtype
)

def _dedupe_columns(cols):
    seen = {}
    out = []
    for c in map(str, cols):
        if c not in seen:
            seen[c] = 1
            out.append(c)
        else:
            out.append(f"{c}.{seen[c]}")
            seen[c] += 1
    return out

def make_arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce df into Arrow/Streamlit-friendly types with per-column guards."""
    df = df.copy()

    # 0) Ensure unique, string column names (critical for Arrow)
    try:
        df.columns = _dedupe_columns(df.columns)
    except Exception:
        df.columns = _dedupe_columns([str(c) for c in df.columns])

    # 1) Normalize index (Arrow doesn't need fancy indexes)
    df = df.reset_index(drop=True)

    # 2) Per-column sanitation
    for c in list(df.columns):
        s = df[c]

        # 2a) Datetimes → tz-naive
        try:
            if is_datetime64_any_dtype(s):
                df[c] = pd.to_datetime(s, errors="coerce").dt.tz_localize(None)
            elif is_object_dtype(s) and s.map(
                lambda x: isinstance(x, pd.Timestamp) and getattr(x, "tz", None) is not None
                if pd.notna(x) else False
            ).any():
                df[c] = pd.to_datetime(s, errors="coerce").dt.tz_localize(None)
        except Exception:
            df[c] = s.astype(str)
            continue

        # 2b) Nullable ints/bools → plain or string (Arrow sometimes chokes on extension dtypes)
        try:
            if is_integer_dtype(df[c].dtype) and str(df[c].dtype).startswith("Int"):  # Pandas nullable Int64/Int32
                df[c] = df[c].astype("float64")  # safe preview; preserves NaN
            elif is_bool_dtype(df[c].dtype) and str(df[c].dtype) == "boolean":        # Pandas nullable BooleanDtype
                df[c] = df[c].astype("object").astype(str)
        except Exception:
            df[c] = df[c].astype(str)

        # 2c) Container/bytes → JSON/string
        if is_object_dtype(df[c]):
            def _to_serializable(x):
                try:
                    if isinstance(x, (dict, list, tuple, set)):
                        return json.dumps(x, ensure_ascii=False)
                    if isinstance(x, (bytes, bytearray)):
                        try:
                            return x.decode("utf-8", "ignore")
                        except Exception:
                            return str(x)
                    return x
                except Exception:
                    return str(x)

            try:
                needs_map = df[c].map(lambda v: isinstance(v, (dict, list, tuple, set, bytes, bytearray))).any()
            except Exception:
                needs_map = True

            if needs_map:
                try:
                    df[c] = df[c].map(_to_serializable)
                except Exception:
                    df[c] = df[c].astype(str)

        # 2d) Mixed-type object → string
        if is_object_dtype(df[c]):
            try:
                types = df[c].dropna().map(lambda v: type(v).__name__).unique()
                if len(types) > 1:
                    df[c] = df[c].astype(str)
            except Exception:
                df[c] = df[c].astype(str)

    # 3) Inf/NaN → None (nicer for Arrow)
    try:
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.where(pd.notna(df), None)
    except Exception:
        pass

    return df

def st_dataframe_safe(df: pd.DataFrame, **kwargs):
    return st.dataframe(make_arrow_safe(df), **kwargs)


def _flatten(x):
    """
    Turn lists/dicts/tuples into JSON strings so PyArrow can serialize.
    """
    if isinstance(x, (list, dict, tuple)):
        try:
            return json.dumps(x, ensure_ascii=False)
        except Exception:
            return str(x)
    return x

#############################
# Model Descriptions + UI   #
#############################
MODEL_OPTIONS = {
    "gpt-3.5-turbo": "Cheapest, good for basic tasks with acceptable quality.",
    "gpt-4.1-mini":  "Balanced cost and intelligence, great for language tasks.",
    "gpt-4.1-nano":  "Ultra-cheap and fast, best for very lightweight checks.",
    "gpt-4o-mini":   "Higher quality than 4.1-mini, still affordable.",
    "gpt-4o":        "The latest multimodal GPT-4 (vision + text).",
    "gpt-4-turbo":   "Very powerful and expensive — best for complex, high-value use cases."
}

# ---- Main Page Layout ----
st.markdown("<h1>📄 Flexible AI Product Data Checker With Cost Estimate</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; font-size:16px; color:#4A4443;'>"
    "Process your CSV row by row with OpenAI. Configure your columns, select (or write) a prompt, and choose a model."
    "</p>",
    unsafe_allow_html=True
)

# ── API key ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    api_key_input = st.text_input("🔑 Enter your OpenAI API Key", type="password")
    if not api_key_input:
        st.warning("Please enter your OpenAI API key to proceed.")
        st.stop()
    client = OpenAI(api_key=api_key_input)

@st.cache_data(show_spinner=False)
def _bulk_prescreen_banned(texts: list[str], threshold: int) -> dict[int, list[dict]]:
    # texts should be a plain list of strings; threshold is your slider value
    return banningredients.bulk_find_banned_candidates(texts=texts, threshold=threshold)

# ------------------------------------------------------------------
# Three-pass image ingredients extractor (OCR → Allergen HTML → QC)
# ------------------------------------------------------------------
def three_pass_extract(image_bytes: bytes) -> str:
    """
    Run GPT-4o three times:
      • Pass-1: OCR the ingredients panel (verbatim)
      • Pass-2: Format to HTML + <b>bold</b> allergens (UK list)
      • Pass-3: Correct obvious OCR misreads, keep HTML
    """
    import base64, textwrap
    data_url = f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode()}"

    # ---- PASS 1 ----
    pass1_sys = (
        "You are a specialist OCR engine. Extract the EXACT text of the INGREDIENTS "
        "panel on a UK food label image. Preserve punctuation, %, brackets. "
        "If the section is unreadable, output IMAGE_UNREADABLE."
    )
    resp1 = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": pass1_sys},
            {"role": "user",   "content": [
                {"type": "text", "text": "Label image incoming."},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]}
        ],
        temperature=temperature_val, top_p=0
    )
    raw = resp1.choices[0].message.content.strip()
    if "IMAGE_UNREADABLE" in raw.upper():
        return "IMAGE_UNREADABLE"

    # ---- PASS 2 ----
    allergens = (
        "celery,wheat,rye,barley,oats,spelt,kamut,crustaceans,eggs,fish,lupin,"
        "milk,molluscs,mustard,almond,hazelnut,walnut,cashew,pecan,pistachio,"
        "macadamia,brazil nut,peanut,sesame,soy,soya,sulphur dioxide,sulphites"
    )
    pass2_sys = textwrap.dedent(f"""
        You are a food-label compliance agent. Format the INGREDIENT string
        exactly as HTML and bold (<b>…</b>) every word that matches this UK-FIC allergen list:

        {allergens}

        • Bold ONLY the allergen token(s); keep all other text unchanged.
        • Do NOT re-order, translate, or summarise.
        • Return the HTML string only – no markdown, no commentary.
    """).strip()
    resp2 = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": pass2_sys},
            {"role": "user",   "content": raw}
        ],
        temperature=temperature_val, top_p=0
    )
    html_out = resp2.choices[0].message.content.strip()

    # ---- PASS 3 ----
    pass3_sys = (
        "You previously extracted an ingredient list from a UK food label image. "
        "Double-check for spelling errors/OCR misreads. If corrections are needed, return the corrected string, "
        "preserving existing HTML. Otherwise return the input unchanged."
    )
    resp3 = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": pass3_sys},
            {"role": "user",   "content": html_out}
        ],
        temperature=temperature_val, top_p=0
    )
    return resp3.choices[0].message.content.strip()

def fetch_image_as_base64(url: str) -> str | None:
    """Fetch an image and return it base64-encoded. Returns None on failure."""
    try:
        if not url.startswith("http"):
            url = "https://" + url.strip().lstrip("/")
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        return base64.b64encode(response.content).decode("utf-8")
    except Exception:
        return None

# ── Prompt picker ──────────────────────────────────────────────────────────────
st.subheader("💬 Choose a Prompt")

ARTWORK_AUTO_PROMPT                = "Artwork: Ingredient Statement (PDF/JPEG)"
ARTWORK_DIRECTIONS_PROMPT          = "Artwork: Directions for Use (PDF/JPEG)"
ARTWORK_PACKSIZE_PROMPT            = "Artwork: Pack Size / Net & Gross Weight (PDF/JPEG)"
ARTWORK_NUTRITION_PROMPT           = "Artwork: Nutrition Facts (PDF/JPEG)"
ARTWORK_SUPPLIER_PROMPT            = "Artwork: Supplier Addresses (UK/EU) (PDF/JPEG)"
ARTWORK_RUN_ALL_PROMPT             = "Artwork: Run ALL Packaging Pipelines (PDF/JPEG)"
ARTWORK_WARNINGS_ADVISORY_PROMPT   = "Artwork: Warnings & Advisory (PDF/JPEG)"
GREEN_CLAIMS_PROMPT                = "Green Claims Checker (Language-aware)"

prompt_names = list(PROMPT_OPTIONS.keys()) + [
    ARTWORK_AUTO_PROMPT,
    ARTWORK_DIRECTIONS_PROMPT,
    ARTWORK_PACKSIZE_PROMPT,
    ARTWORK_NUTRITION_PROMPT,
    ARTWORK_SUPPLIER_PROMPT,
    ARTWORK_WARNINGS_ADVISORY_PROMPT,
    ARTWORK_RUN_ALL_PROMPT,
    GREEN_CLAIMS_PROMPT,
]

prompt_choice = st.selectbox("Select a pre-written prompt or 'Custom':", prompt_names, index=0)

# ── Competitor DB upload (only if needed) ──────────────────────────────────────
COMP_DB = None
if prompt_choice == "Competitor SKU Match":
    comp_file = st.file_uploader("🔍 Upload competitor CSV", type=["csv"], key="comp_csv")
    if comp_file:
        comp_df = pd.read_csv(comp_file, dtype=str).fillna("")
        COMP_DB = [
            parse_sku(row["Retailer Product Name"], uid=row.get("UID", ""))
            for _, row in comp_df.iterrows()
        ]
    else:
        st.warning("Please upload a competitor CSV to enable SKU matching.")

# ── Prompt metadata → default model & description ──────────────────────────────
if prompt_choice == ARTWORK_AUTO_PROMPT:
    selected_prompt_text = "SYSTEM MESSAGE: handled by artwork_processing module"
    prompt_description   = "Auto-finds Ingredients panel; returns exact text + HTML with allergens bolded."
    recommended_model    = "gpt-4o"
elif prompt_choice == ARTWORK_DIRECTIONS_PROMPT:
    selected_prompt_text = "SYSTEM MESSAGE: handled by artwork_processing (directions) module"
    prompt_description   = "Auto-finds Directions/Usage/Preparation; extracts text, structures steps, tags pictograms."
    recommended_model    = "gpt-4o"
elif prompt_choice == ARTWORK_PACKSIZE_PROMPT:
    selected_prompt_text = "SYSTEM MESSAGE: handled by artwork_processing (pack size/weights) module"
    prompt_description   = "Parses net quantity/pack size; extracts No. items, base quantity, UoM, net/gross/drained weight, ℮."
    recommended_model    = "gpt-4o"
elif prompt_choice == ARTWORK_NUTRITION_PROMPT:
    selected_prompt_text = "SYSTEM MESSAGE: handled by artwork_processing (nutrition) module"
    prompt_description   = "Auto-locates nutrition panel → structured JSON + flat rows."
    recommended_model    = "gpt-4o"
elif prompt_choice == ARTWORK_SUPPLIER_PROMPT:
    selected_prompt_text = "SYSTEM MESSAGE: handled by artwork_processing (supplier addresses) module"
    prompt_description   = "Extract UK/EU supplier/responsible person blocks + bboxes."
    recommended_model    = "gpt-4o"
elif prompt_choice == ARTWORK_RUN_ALL_PROMPT:
    selected_prompt_text = "SYSTEM MESSAGE: handled by run-all packaging pipelines module"
    prompt_description   = "Runs Ingredients → Directions → Pack Size → Nutrition → Supplier → Warnings in sequence."
    recommended_model    = "gpt-4o"
elif prompt_choice == GREEN_CLAIMS_PROMPT:
    selected_prompt_text = "SYSTEM MESSAGE: handled by green_claims module"
    prompt_description   = "Matches text to a Green Claims library using fuzzy screen + AI adjudicator (language-aware)."
    recommended_model    = "gpt-4.1-mini"
elif prompt_choice == ARTWORK_WARNINGS_ADVISORY_PROMPT:
    selected_prompt_text = "SYSTEM MESSAGE: handled by artwork_processing (warnings/advisory) module"
    prompt_description   = "Finds warnings/advisory statements; outputs two verbatim lists + QA."
    recommended_model    = "gpt-4o"
else:
    selected = PROMPT_OPTIONS[prompt_choice]
    selected_prompt_text = selected["prompt"]
    recommended_model    = selected["recommended_model"]
    prompt_description   = selected["description"]

st.markdown(f"**Prompt Info:** {prompt_description}")

# ── Session state for crop resets ──────────────────────────────────────────────
if "last_prompt" not in st.session_state:
    st.session_state["last_prompt"] = prompt_choice
if "cropped_bytes" not in st.session_state:
    st.session_state["cropped_bytes"] = None
if st.session_state["last_prompt"] != prompt_choice:
    st.session_state["last_prompt"] = prompt_choice
    st.session_state["cropped_bytes"] = None

# ── Prompt-specific controls ───────────────────────────────────────────────────
fuzzy_threshold = 87
if prompt_choice == "Novel Food Checker (EU)":
    fuzzy_threshold = st.slider("Novel-food fuzzy threshold", 70, 100, 87,
                                help="Lower = catch more variants (but more false positives).")

if prompt_choice == GREEN_CLAIMS_PROMPT:
    gc_threshold = st.slider("Green-claims fuzzy threshold", 70, 100, 85,
                             help="Lower = catch more variants (but more false positives).")

    gc_language = st.selectbox(
        "Claim language",
        options=list(LANG_TO_COL.keys()),
        index=(list(LANG_TO_COL.keys()).index("Dutch (NL)") if "Dutch (NL)" in LANG_TO_COL else 0),
        help="Only candidates from this language column will be considered."
    )

    gc_upload = st.file_uploader("Upload your green-claims-database.csv (⚠️ not your product CSV)", type=["csv"], key="gc_db")

    try:
        GC_DB = load_green_claims_db(uploaded_file=gc_upload)
    except Exception as e:
        st.error(f"Could not load green-claims database: {e}")
        st.stop()

    # Catch wrong file
    PRODUCT_MARKERS = {"SKU ID", "SKU Name", "Product Name"}
    if PRODUCT_MARKERS.issubset(set(GC_DB.columns)):
        st.error("Looks like you uploaded a PRODUCT CSV. Please upload green-claims-database.csv here instead.")
        st.stop()

    lang_col = LANG_TO_COL.get(gc_language)
    if not lang_col:
        st.error(f"No mapping found in LANG_TO_COL for '{gc_language}'.")
        st.stop()

    if lang_col not in GC_DB.columns:
        st.error(f"Column '{lang_col}' not found for {gc_language}. Available: {list(GC_DB.columns)}")
        st.stop()

    non_empty = (GC_DB[lang_col].astype(str).str.strip() != "").sum()
    st.caption(f"Green-claims DB: {len(GC_DB):,} rows • {gc_language} '{lang_col}' non-empty: {non_empty:,}")
    if non_empty == 0:
        st.warning(f"No text in '{lang_col}'. If your CSV uses semicolons, ensure the loader detected it.")
        st.stop()

    gc_debug = st.checkbox("🔍 Show Green Claims matcher debug (first 10 rows)", value=False)
    if gc_debug:
        st.dataframe(GC_DB[[lang_col]].head(10))

# Slider for the Banned/Restricted checker
if prompt_choice == "Banned/Restricted Checker":
    banned_fuzzy_threshold = st.slider("Banned/Restricted fuzzy threshold", 80, 100, 90,
                                       help="Lower = catch more variants (but more false positives).")

# --- Determine if image-based (manual crop prompts only) ---
single_image_prompts = {
    "Image: Ingredient Scrape (HTML)",
    "Image: Directions for Use",
    "Image: Storage Instructions",
    "Image: Warnings and Advisory (JSON)",
}
multi_image_url_prompts = {
    "Image: Multi-Image Ingredient Extract & Compare",
    "GHS Pictogram Detector"
}

is_image_prompt = prompt_choice in single_image_prompts
uploaded_image = None
uploaded_file = None

# Force gpt-4o for image prompts
if is_image_prompt:
    recommended_model = "gpt-4o"

# 5) Model & Temperature
all_model_keys = list(MODEL_OPTIONS.keys())
default_index = all_model_keys.index(recommended_model) if recommended_model in all_model_keys else 0
model_choice = st.selectbox("🧠 Choose GPT model", all_model_keys, index=default_index)
st.markdown(f"**Model Info:** {MODEL_OPTIONS[model_choice]}")
temperature_val = st.slider("🎛️ Temperature (0 = deterministic)", 0.0, 1.0, 0.0, 0.05)
st.markdown("---")

# 6) Prompt textarea
user_prompt = st.text_area("✍️ Your prompt for GPT", value=selected_prompt_text, height=200)

# ===========================
#   AUTO ARTWORK FLOWS
# ===========================
def _auto_artwork_gate():
    st.stop()  # used to short-circuit after these flows

if prompt_choice == ARTWORK_AUTO_PROMPT:
    st.markdown("### 📄 Upload artwork (PDF/JPG/PNG) – no manual crop")
    art_file = st.file_uploader("Choose file", type=["pdf","jpg","jpeg","png"], key="art_auto")
    if model_choice != "gpt-4o":
        st.warning("This prompt is designed for **gpt-4o** (vision).")

    if art_file and st.button("🚀 Extract Ingredients (Auto)"):
        with st.spinner("Locating INGREDIENTS…"):
            try:
                res = process_artwork(client=client, file_bytes=art_file.read(), filename=art_file.name, render_dpi=350, model="gpt-4o")
            except Exception as e:
                res = {"ok": False, "error": f"Processing failed: {e}"}
        if not res.get("ok"):
            st.error(res.get("error", "Failed"))
        else:
            st.success("✅ Extracted INGREDIENTS")
            st.write(f"**Page:** {res['page_index']}")
            st.write(f"**BBox (pixels):** {res['bbox_pixels']}")
            st.code(res["ingredients_text"], language="text")
            st.code(res["ingredients_html"], language="html")
            st.json(res["qa"])
            st.download_button("⬇️ Download JSON", data=json.dumps(res, ensure_ascii=False, indent=2).encode("utf-8"),
                               file_name="ingredients_result.json", mime="application/json")
    _auto_artwork_gate()

if prompt_choice == ARTWORK_WARNINGS_ADVISORY_PROMPT:
    st.markdown("### 📄 Upload artwork (PDF/JPG/PNG) – auto-locate Warnings & Advisory")
    art_file = st.file_uploader("Choose file", type=["pdf","jpg","jpeg","png"], key="art_warnadv_auto")
    if model_choice != "gpt-4o":
        st.warning("This prompt is designed for **gpt-4o** (vision).")

    if art_file and st.button("🚀 Extract Warnings & Advisory (Auto)"):
        with st.spinner("Locating WARNINGS/ADVISORY…"):
            try:
                res = process_artwork_warnings_advisory(client=client, file_bytes=art_file.read(),
                                                        filename=art_file.name, render_dpi=350, model="gpt-4o")
            except Exception as e:
                res = {"ok": False, "error": f"Processing failed: {e}"}
        if not res.get("ok"):
            st.error(res.get("error", "Failed"))
        else:
            st.success("✅ Extracted Warnings/Advisory")
            st.write(f"**Page:** {res.get('page_index')}")
            if res.get("bbox_pixels"):
                st.write(f"**BBox (pixels):** {res.get('bbox_pixels')}")
            st.markdown("#### ⚠️ Warnings")
            if res.get("warning_info"):
                for w in res["warning_info"]:
                    st.markdown(f"- {w}")
            else:
                st.info("No warnings detected.")
            st.markdown("#### ℹ️ Advisory")
            if res.get("advisory_info"):
                for a in res["advisory_info"]:
                    st.markdown(f"- {a}")
            else:
                st.info("No advisory statements detected.")
            st.markdown("#### 🧪 QA")
            st.json(res.get("qa", {}))
            st.download_button("⬇️ Download JSON", data=json.dumps(res, ensure_ascii=False, indent=2).encode("utf-8"),
                               file_name="warnings_advisory_result.json", mime="application/json")
    _auto_artwork_gate()

if prompt_choice == ARTWORK_RUN_ALL_PROMPT:
    st.markdown("### 📄 Upload artwork (PDF/JPG/PNG) – run **all** packaging pipelines")
    art_file = st.file_uploader("Choose file", type=["pdf","jpg","jpeg","png"], key="art_run_all")
    if model_choice != "gpt-4o":
        st.warning("This prompt is designed for **gpt-4o** (vision).")

    output_mode = st.radio("Output format", ["Single JSON (everything together)", "6 JSONs (zipped)"], index=0)

    if art_file and st.button("🚀 Run ALL Pipelines (Ingredients → Directions → Pack Size → Nutrition → Supplier → Warnings)"):
        file_bytes = art_file.read()
        filename   = art_file.name

        def run_step(fn, label):
            try:
                with st.spinner(f"Running {label}…"):
                    res = fn(client=client, file_bytes=file_bytes, filename=filename, render_dpi=350, model="gpt-4o")
                ok = bool(res.get("ok", True))
                st.success(f"✅ {label} complete" if ok else f"❌ {label} failed: {res.get('error','Unknown error')}")
                return res
            except Exception as e:
                err = {"ok": False, "error": f"{label} exception: {e}"}
                st.error(f"❌ {label} crashed: {e}")
                return err

        ingredients_res   = run_step(process_artwork,                 "Ingredients")
        directions_res    = run_step(process_artwork_directions,      "Directions")
        packsize_res      = run_step(process_artwork_packsize,        "Pack Size / Weights")
        nutrition_res     = run_step(process_artwork_nutrition,       "Nutrition")
        supplier_res      = run_step(process_artwork_suppliers,       "Supplier Addresses")
        warnings_adv_res  = run_step(process_artwork_warnings_advisory,"Warnings & Advisory")

        combined = {
            "source_file": filename,
            "model": "gpt-4o",
            "pipelines": {
                "ingredients": ingredients_res,
                "directions":  directions_res,
                "packsize":    packsize_res,
                "nutrition":   nutrition_res,
                "suppliers":   supplier_res,
                "warnings_advisory": warnings_adv_res
            }
        }

        st.markdown("#### 🧭 Run summary")
        st.json({
            "ingredients_ok":        bool(ingredients_res.get("ok", True)),
            "directions_ok":         bool(directions_res.get("ok", True)),
            "packsize_ok":           bool(packsize_res.get("ok", True)),
            "nutrition_ok":          bool(nutrition_res.get("ok", True)),
            "suppliers_ok":          bool(supplier_res.get("ok", True)),
            "warnings_advisory_ok":  bool(warnings_adv_res.get("ok", True)),
        })

        if output_mode.startswith("Single"):
            data = json.dumps(combined, ensure_ascii=False, indent=2).encode("utf-8")
            st.download_button("⬇️ Download Combined JSON", data=data,
                               file_name="packaging_pipelines_all.json", mime="application/json")
        else:
            import zipfile
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("ingredients_result.json",        json.dumps(ingredients_res, ensure_ascii=False, indent=2))
                zf.writestr("directions_result.json",         json.dumps(directions_res,  ensure_ascii=False, indent=2))
                zf.writestr("packsize_result.json",           json.dumps(packsize_res,    ensure_ascii=False, indent=2))
                zf.writestr("nutrition_result.json",          json.dumps(nutrition_res,   ensure_ascii=False, indent=2))
                zf.writestr("supplier_addresses_result.json", json.dumps(supplier_res,    ensure_ascii=False, indent=2))
                zf.writestr("warnings_advisory_result.json",  json.dumps(warnings_adv_res,ensure_ascii=False, indent=2))
                zf.writestr("packaging_pipelines_all.json",   json.dumps(combined,        ensure_ascii=False, indent=2))
            buf.seek(0)
            st.download_button("⬇️ Download ZIP (5 JSONs + combined)",
                               data=buf.getvalue(), file_name="packaging_pipelines_all.zip", mime="application/zip")
    _auto_artwork_gate()

if prompt_choice == ARTWORK_SUPPLIER_PROMPT:
    st.markdown("### 📄 Upload artwork (PDF/JPG/PNG) – auto-locate Supplier/Responsible Person addresses")
    art_file = st.file_uploader("Choose file", type=["pdf","jpg","jpeg","png"], key="art_supplier_auto")
    if model_choice != "gpt-4o":
        st.warning("This prompt is designed for **gpt-4o** (vision).")

    if art_file and st.button("🚀 Extract Supplier Addresses (Auto)"):
        with st.spinner("Locating supplier/Responsible Person…"):
            try:
                res = process_artwork_suppliers(client=client, file_bytes=art_file.read(),
                                                filename=art_file.name, render_dpi=350, model="gpt-4o")
            except Exception as e:
                res = {"ok": False, "error": f"Processing failed: {e}"}
        if not res.get("ok"):
            st.error(res.get("error", "Failed"))
        else:
            st.success("✅ Extracted Supplier Addresses")
            st.write(f"**Page:** {res.get('page_index')}")
            st.markdown("#### 🇬🇧 UK Address")
            if res.get("uk_address_text"):
                st.code(res["uk_address_text"], language="text")
                st.write(f"**UK BBox (pixels):** {res.get('uk_bbox_pixels')}")
            else:
                st.info("No UK address detected.")
            st.markdown("#### 🇪🇺 EU Address")
            if res.get("eu_address_text"):
                st.code(res["eu_address_text"], language="text")
                st.write(f"**EU BBox (pixels):** {res.get('eu_bbox_pixels')}")
            else:
                st.info("No EU address detected.")
            st.markdown("#### 🧪 QA signals")
            st.json(res.get("qa", {}))
            st.download_button("⬇️ Download JSON", data=json.dumps(res, ensure_ascii=False, indent=2).encode("utf-8"),
                               file_name="supplier_addresses_result.json", mime="application/json")
    _auto_artwork_gate()

if prompt_choice == ARTWORK_PACKSIZE_PROMPT:
    st.markdown("### 📄 Upload artwork (PDF/JPG/PNG) – auto-locate Pack Size / Weights")
    art_file = st.file_uploader("Choose file", type=["pdf","jpg","jpeg","png"], key="art_packsize_auto")
    if model_choice != "gpt-4o":
        st.warning("This prompt is designed for **gpt-4o** (vision).")

    if art_file and st.button("🚀 Extract Pack Size / Weights (Auto)"):
        with st.spinner("Locating PACK SIZE / NET QUANTITY…"):
            try:
                res = process_artwork_packsize(client=client, file_bytes=art_file.read(),
                                               filename=art_file.name, render_dpi=350, model="gpt-4o")
            except Exception as e:
                res = {"ok": False, "error": f"Processing failed: {e}"}

        if not res.get("ok"):
            st.error(res.get("error", "Failed"))
        else:
            st.success("✅ Extracted Pack Size / Weights")
            st.write(f"**Page:** {res.get('page_index')}")
            st.write(f"**BBox (pixels):** {res.get('bbox_pixels')}")
            parsed = (res.get("parsed") or {})
            st.markdown("#### 🧾 Parsed Summary")
            colA, colB, colC = st.columns(3)
            with colA:
                st.metric("Number of items", str(parsed.get("number_of_items") or "—"))
            with colB:
                st.metric("Base quantity", str(parsed.get("base_quantity") or "—"))
            with colC:
                st.metric("Unit of measure", parsed.get("unit_of_measure") or "—")

            st.markdown("#### ⚖️ Weights")
            nw = parsed.get("net_weight") or {}
            gw = parsed.get("gross_weight") or {}
            dw = parsed.get("drained_weight") or {}
            e_flag = parsed.get("e_mark_present")
            st.write({
                "Net weight":     f"{nw.get('value')} {nw.get('unit')}" if nw.get("value") is not None else None,
                "Gross weight":   f"{gw.get('value')} {gw.get('unit')}" if gw.get("value") is not None else None,
                "Drained weight": f"{dw.get('value')} {dw.get('unit')}" if dw.get("value") is not None else None,
                "℮ present":      e_flag if e_flag is not None else None
            })

            st.markdown("#### 🧩 Raw OCR lines considered")
            st.code("\n".join(parsed.get("raw_candidates") or []), language="text")
            st.markdown("#### 🪵 Raw text (crop OCR)")
            st.code(res.get("raw_text", ""), language="text")
            st.markdown("#### 🧪 QA signals")
            st.json(res.get("qa", {}))
            st.download_button("⬇️ Download JSON", data=json.dumps(res, ensure_ascii=False, indent=2).encode("utf-8"),
                               file_name="packsize_result.json", mime="application/json")
    _auto_artwork_gate()

if prompt_choice == ARTWORK_NUTRITION_PROMPT:
    st.markdown("### 📄 Upload artwork (PDF/JPG/PNG) – auto-locate Nutrition panel")
    art_file = st.file_uploader("Choose file", type=["pdf","jpg","jpeg","png"], key="art_nutrition_auto")
    if model_choice != "gpt-4o":
        st.warning("This prompt is designed for **gpt-4o**.")
    if art_file and st.button("🚀 Extract Nutrition (Auto)"):
        with st.spinner("Finding nutrition table…"):
            try:
                res = process_artwork_nutrition(client=client, file_bytes=art_file.read(),
                                                filename=art_file.name, render_dpi=350, model="gpt-4o")
            except Exception as e:
                res = {"ok": False, "error": f"Processing failed: {e}"}
        if not res.get("ok"):
            st.error(res.get("error","Failed"))
        else:
            st.success("✅ Nutrition extracted")
            st.write(f"**Page:** {res['page_index']}")
            st.write(f"**BBox (pixels):** {res['bbox_pixels']}")
            st.markdown("#### Flat rows (easy export)")
            st_dataframe_safe(pd.DataFrame(res.get("flat", [])), use_container_width=True)
            st.markdown("#### Structured JSON")
            st.json(res.get("parsed", {}))
            st.markdown("#### QA")
            st.json(res.get("qa", {}))
            st.download_button("⬇️ Download Nutrition JSON",
                               data=json.dumps(res, ensure_ascii=False, indent=2).encode("utf-8"),
                               file_name="nutrition_result.json", mime="application/json")
    _auto_artwork_gate()

if prompt_choice == ARTWORK_DIRECTIONS_PROMPT:
    st.markdown("### 📄 Upload artwork (PDF/JPG/PNG) – auto-locate Directions/Usage/Preparation")
    art_file = st.file_uploader("Choose file", type=["pdf","jpg","jpeg","png"], key="art_directions_auto")
    if model_choice != "gpt-4o":
        st.warning("This prompt is designed for **gpt-4o** (vision).")

    if art_file and st.button("🚀 Extract Directions (Auto)"):
        with st.spinner("Locating DIRECTIONS/USAGE/PREPARATION…"):
            try:
                res = process_artwork_directions(client=client, file_bytes=art_file.read(),
                                                 filename=art_file.name, render_dpi=350, model="gpt-4o")
            except Exception as e:
                res = {"ok": False, "error": f"Processing failed: {e}"}
        if not res.get("ok"):
            st.error(res.get("error", "Failed"))
        else:
            st.success("✅ Extracted DIRECTIONS/USAGE/PREPARATION")
            st.write(f"**Page:** {res['page_index']}")
            st.write(f"**BBox (pixels):** {res['bbox_pixels']}")
            st.markdown("**Raw directions text (exact OCR):**")
            st.code(res["directions_text"], language="text")
            st.markdown("**Steps (HTML):**")
            if res.get("steps_html"):
                st.markdown(res["steps_html"], unsafe_allow_html=True)
            else:
                st.info("No ordered steps found.")
            st.markdown("**Structured extraction:**")
            st.json(res.get("structured", {}))
            st.markdown("**Pictograms (icons):**")
            st.json(res.get("pictograms", {}))
            st.markdown("**QA signals:**")
            st.json(res.get("qa", {}))
            st.download_button("⬇️ Download JSON", data=json.dumps(res, ensure_ascii=False, indent=2).encode("utf-8"),
                               file_name="directions_result.json", mime="application/json")
    _auto_artwork_gate()

# ------------------------------------------------------------------------------
# Manual crop (single-image prompts)
# ------------------------------------------------------------------------------
if is_image_prompt:
    st.markdown("### 🖼️ Upload Product Image & crop just the relevant panel")
    uploaded_image = st.file_uploader("Choose JPG or PNG", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        img = Image.open(uploaded_image).convert("RGB")
        st.markdown("### ✂️ Crop the label to the relevant section below:")
        with st.spinner("🖼️ Loading crop tool..."):
            cropped_img = st_cropper(
                img, box_color="#C2EA46", realtime_update=True, aspect_ratio=None, return_type="image"
            )
        if st.button("✅ Use this crop →"):
            buf = io.BytesIO()
            cropped_img.save(buf, format="PNG")
            st.session_state["cropped_bytes"] = buf.getvalue()
            st.session_state["cropped_preview"] = cropped_img
            st.success("✅ Crop captured! Preview below:")
            st.image(cropped_img, use_container_width=True, caption="Cropped Area Sent to GPT")
            st.download_button("⬇️ Download Cropped Image Sent to GPT",
                               data=st.session_state["cropped_bytes"], file_name="cropped_label.png", mime="image/png")

# Non-image prompts → CSV uploader
if not is_image_prompt:
    uploaded_file = st.file_uploader("📁 Upload your product CSV", type=["csv"], key="data_csv")

# ------------------------------------------------------------------------------
# Image-prompt processing (manual crop)
# ------------------------------------------------------------------------------
if is_image_prompt and st.session_state.get("cropped_bytes"):
    st.markdown("### 📤 Processing image…")
    with st.spinner("Running high-accuracy image extraction"):
        if model_choice != "gpt-4o":
            st.error("🛑 Image prompts require the **gpt-4o** model. Please select it and try again.")
            st.stop()

        try:
            if "Ingredient Scrape" in prompt_choice:
                html_out = three_pass_extract(st.session_state["cropped_bytes"])
            else:
                data_url = "data:image/jpeg;base64," + base64.b64encode(st.session_state["cropped_bytes"]).decode()
                system_msg = user_prompt.replace("SYSTEM MESSAGE:", "").strip()
                response = client.chat.completions.create(
                    model=model_choice,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": [
                            {"type": "text", "text": "Cropped label image below."},
                            {"type": "image_url", "image_url": {"url": data_url}}
                        ]}
                    ],
                    temperature=temperature_val, top_p=0
                )
                html_out = response.choices[0].message.content.strip()

            if html_out == "IMAGE_UNREADABLE":
                st.error("🛑 The image was unreadable or missing the required section.")
            else:
                st.success("✅ GPT image processing complete!")
                output_type = "html"
                if "Directions" in prompt_choice or "Storage" in prompt_choice:
                    output_type = "text"
                elif "Warnings and Advisory" in prompt_choice:
                    output_type = "json"
                st.code(html_out, language=output_type)
        except Exception as e:
            st.error(f"Image processing failed: {e}")

# ------------------------------------------------------------------------------
# CSV-driven processing
# ------------------------------------------------------------------------------
if uploaded_file and (
    user_prompt.strip() or
    prompt_choice in {"Novel Food Checker (EU)", "Competitor SKU Match", "GHS Pictogram Detector", "Banned/Restricted Checker"}
):
    df = pd.read_csv(uploaded_file, dtype=str)
    st.markdown("### 📄 CSV Preview")
    st_dataframe_safe(df.head(), use_container_width=True)

    # Dynamic Column Selector (up to 10)
    st.subheader("📊 Select up to 10 CSV columns to pass to GPT")
    selected_columns = st.multiselect(
        "Use in Processing",
        options=df.columns.tolist(),
        default=df.columns.tolist()[:3],
        help="Pick between 1 and 10 columns."
    )

    if not selected_columns:
        st.error("⚠️ Please select at least one column.")
        st.stop()
    if len(selected_columns) > 10:
        st.error("⚠️ You can select at most 10 columns. Please deselect some.")
        st.stop()

    cols_to_use = selected_columns

    # Prompt-specific extra selectors
    if prompt_choice == "Competitor SKU Match":
        sku_col = st.selectbox(
            "Which column contains *your* product name / volume?",
            options=cols_to_use,
            help="e.g. 'Product_Name' or 'SKU Title'"
        )

    if prompt_choice == "Banned/Restricted Checker":
        banned_ing_col = st.selectbox(
            "Which column contains the full ingredients text?",
            options=cols_to_use,
            help="Pick the column with the product’s ingredient statement."
        )
        banned_texts = df[banned_ing_col].fillna("").astype(str).tolist() if banned_ing_col in df.columns else ["" for _ in range(len(df))]
        with st.spinner("Pre-screening banned/restricted candidates across all rows…"):
            prescreen_map = _bulk_prescreen_banned(banned_texts, banned_fuzzy_threshold)
        rows_with_candidates = sum(1 for v in prescreen_map.values() if v)
        st.info(f"Prescreen found candidates in {rows_with_candidates} of {len(df)} rows. "
                "Only those rows will be sent to GPT for adjudication.")

    # Prescreen-aware cost estimate
    est_df = df
    if prompt_choice == "Banned/Restricted Checker" and isinstance(globals().get('prescreen_map'), dict):
        rows_to_send = [i for i, c in prescreen_map.items() if c]
        est_df = df.iloc[rows_to_send] if rows_to_send else df.iloc[0:0]

    st.markdown(
        f"""
        <div style='padding:10px; background-color:#FFFFFF; color:#4A4443; border-radius:5px; margin-bottom:1rem;'>
            <strong>Estimated Cost:</strong> ${estimate_cost(model_choice, est_df, user_prompt, cols_to_use):0.4f}
            <br/>(rows counted: {len(est_df)}/{len(df)})
        </div>
        """,
        unsafe_allow_html=True
    )

    gauge_placeholder = st.empty()

    if st.button("🚀 Run GPT on CSV"):
        if prompt_choice == "Competitor SKU Match" and not COMP_DB:
            st.error("Cannot run SKU match—no competitor CSV uploaded.")
            st.stop()

        with st.spinner("Processing with GPT..."):
            progress_bar = st.progress(0)
            progress_text = st.empty()
            n_rows = len(df)
            results = []
            failed_rows = []
            log_placeholder = st.empty()
            status_placeholder = st.empty()

            for idx, row in df.iterrows():
                row_data = {c: row.get(c, "") for c in cols_to_use}
                content = ""

                # GHS pictogram detector
                if prompt_choice == "GHS Pictogram Detector":
                    image_urls = row_data.get("image_link", "")
                    image_list = [image_urls.strip()] if image_urls.strip() else []
                    pictograms_found = set()
                    debug_notes_all = []
                    for url in image_list:
                        encoded = fetch_image_as_base64(url)
                        if not encoded:
                            debug_notes_all.append(f"⚠️ Could not fetch image: {url}")
                            continue
                        try:
                            gpt_response = client.chat.completions.create(
                                model="gpt-4o",
                                messages=[
                                    {"role": "system", "content": selected_prompt_text},
                                    {"role": "user", "content": [
                                        {"type": "text", "text": "Check this image for GHS pictograms."},
                                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}
                                    ]}
                                ],
                                temperature=temperature_val, top_p=0
                            )
                            result = json.loads(gpt_response.choices[0].message.content.strip())
                            icons = [i.strip() for i in result.get("pictograms", "").split(",") if i.strip()]
                            pictograms_found.update(icons)
                            debug_notes_all.append(result.get("debug_notes", ""))
                        except Exception as e:
                            failed_rows.append(idx)
                            results.append({"error": f"Error in GPT call for image: {url}", "debug_notes": str(e)})
                            break
                    results.append({"pictograms": ", ".join(sorted(pictograms_found)), "debug_notes": " | ".join(debug_notes_all)})
                    # progress + gauge update for this row
                    progress = (idx + 1) / n_rows
                    progress_bar.progress(progress)
                    progress_text.markdown(
                        f"<h4 style='text-align:center; color:#4A4443;'>Processed {idx + 1} of {n_rows} rows ({progress*100:.1f}%)</h4>",
                        unsafe_allow_html=True
                    )
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=progress * 100,
                        number={'font': {'color': '#4A4443'}},
                        title={'text': "Progress", 'font': {'color': '#4A4443'}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickcolor': '#4A4443', 'tickfont': {'color': '#4A4443'},
                                     'tickwidth': 2, 'ticklen': 8},
                            'bar': {'color': "#C2EA46"},
                            'bgcolor': "#E1FAD1",
                            'borderwidth': 1,
                            'steps': [{'range': [0, 50], 'color': "#E1FAD1"},
                                      {'range': [50, 100], 'color': "#F2FAF4"}]
                        },
                        domain={'x': [0, 1], 'y': [0, 1]}
                    ))
                    gauge_placeholder.plotly_chart(fig, use_container_width=True)
                    continue

                # Green Claims
                if prompt_choice == GREEN_CLAIMS_PROMPT:
                    try:
                        row_text_raw = " ".join(str(row.get(c, "")) for c in cols_to_use if str(row.get(c, "")).strip()).strip()
                        row_text = norm_basic(row_text_raw)
                        if not row_text:
                            results.append({
                                "green_claims_any": False,
                                "green_claims_candidates": [],
                                "green_claims_ai": {},
                                "green_claims_language": gc_language,
                                "explanation": "No text found in selected columns."
                            })
                            pass
                        else:
                            lang_col = LANG_TO_COL[gc_language]
                            candidates = screen_candidates(
                                text=row_text, db=GC_DB, language_col=lang_col,
                                threshold=gc_threshold, max_per_section=5
                            )
                            if not candidates:
                                results.append({
                                    "green_claims_any": False,
                                    "green_claims_candidates": [],
                                    "green_claims_ai": {},
                                    "green_claims_language": gc_language
                                })
                            else:
                                sys_txt, user_txt = build_green_claims_prompt(
                                    candidates=candidates, product_text=row_text, language_name=gc_language
                                )
                                resp = client.chat.completions.create(
                                    model=model_choice,
                                    messages=[{"role": "system", "content": sys_txt},
                                              {"role": "user",   "content": user_txt}],
                                    temperature=temperature_val, top_p=0
                                )
                                content = resp.choices[0].message.content.strip()
                                try:
                                    parsed = json.loads(clean_gpt_json_block(content))
                                except Exception as e:
                                    parsed = {"error": f"JSON parse failed: {e}", "raw_output": content}

                                matched_strings = sorted({s for s in (c.get("evidence_snippet") for c in candidates) if s})
                                results.append({
                                    "green_claims_any": str(parsed.get("overall", {}).get("any_green_claim_detected")).lower() == "true",
                                    "green_claims_matched_strings": matched_strings,
                                    "green_claims_candidates": candidates,
                                    "green_claims_ai": parsed,
                                    "green_claims_language": gc_language
                                })
                    except Exception as e:
                        failed_rows.append(idx)
                        results.append({"error": f"Row {idx} (Green Claims): {e}"})
                    # progress update
                    progress = (idx + 1) / n_rows
                    progress_bar.progress(progress)
                    progress_text.markdown(
                        f"<h4 style='text-align:center; color:#4A4443;'>Processed {idx + 1} of {n_rows} rows ({progress*100:.1f}%)</h4>",
                        unsafe_allow_html=True
                    )
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number", value=progress*100,
                        number={'font': {'color': '#4A4443'}},
                        title={'text': "Progress", 'font': {'color': '#4A4443'}},
                        gauge={'axis': {'range': [0,100], 'tickcolor':'#4A4443','tickfont':{'color':'#4A4443'},'tickwidth':2,'ticklen':8},
                               'bar': {'color': "#C2EA46"}, 'bgcolor': "#E1FAD1", 'borderwidth':1,
                               'steps':[{'range':[0,50],'color':"#E1FAD1"},{'range':[50,100],'color':"#F2FAF4"}]},
                        domain={'x':[0,1],'y':[0,1]}
                    ))
                    gauge_placeholder.plotly_chart(fig, use_container_width=True)
                    continue

                # HFSS Checker
                if prompt_choice == "HFSS Checker":
                    try:
                        p1 = client.chat.completions.create(
                            model=model_choice, messages=[{"role": "system", "content": build_pass_1_prompt(row_data)}]
                        ).choices[0].message.content
                        parsed_1 = json.loads(clean_gpt_json_block(p1))

                        p2 = client.chat.completions.create(
                            model=model_choice, messages=[{"role": "system", "content": build_pass_2_prompt(parsed_1)}]
                        ).choices[0].message.content
                        parsed_2 = json.loads(clean_gpt_json_block(p2))

                        p3 = client.chat.completions.create(
                            model=model_choice, messages=[{"role": "system", "content": build_pass_3_prompt({
                                **parsed_2, "is_drink": parsed_1.get("is_drink", False)
                            })}]
                        ).choices[0].message.content
                        parsed_3 = json.loads(clean_gpt_json_block(p3))

                        all_passes = {"parsed_nutrients": parsed_1, "npm_scoring": parsed_2, "hfss_classification": parsed_3}
                        p4 = client.chat.completions.create(
                            model=model_choice, messages=[{"role": "system", "content": build_pass_4_prompt(all_passes)}]
                        ).choices[0].message.content
                        parsed_4 = json.loads(clean_gpt_json_block(p4))

                        full_result = {**parsed_1, **parsed_2, **parsed_3, **parsed_4}
                        results.append(full_result)
                    except Exception as e:
                        failed_rows.append(idx)
                        results.append({"error": f"Row {idx}: {e}", "raw_output": "Check individual passes for debug info"})
                    # progress update
                    progress = (idx + 1) / n_rows
                    progress_bar.progress(progress)
                    progress_text.markdown(
                        f"<h4 style='text-align:center; color:#4A4443;'>Processed {idx + 1} of {n_rows} rows ({progress*100:.1f}%)</h4>",
                        unsafe_allow_html=True
                    )
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number", value=progress*100,
                        number={'font': {'color': '#4A4443'}},
                        title={'text': "Progress", 'font': {'color': '#4A4443'}},
                        gauge={'axis': {'range': [0,100], 'tickcolor':'#4A4443','tickfont':{'color':'#4A4443'},'tickwidth':2,'ticklen':8},
                               'bar': {'color': "#C2EA46"}, 'bgcolor': "#E1FAD1", 'borderwidth':1,
                               'steps':[{'range':[0,50],'color':"#E1FAD1"},{'range':[50,100],'color':"#F2FAF4"}]},
                        domain={'x':[0,1],'y':[0,1]}
                    ))
                    gauge_placeholder.plotly_chart(fig, use_container_width=True)
                    continue

                # Multi-image ingredient extract & compare
                if prompt_choice == "Image: Multi-Image Ingredient Extract & Compare":
                    image_urls = row.get("image URLs", "")
                    image_list = [url.strip().replace('"', '') for url in image_urls.split(",") if url.strip()]
                    extracted = []
                    for url in image_list:
                        encoded_img = fetch_image_as_base64(url)
                        if not encoded_img:
                            continue
                        try:
                            response = client.chat.completions.create(
                                model="gpt-4o",
                                messages=[
                                    {"role": "system", "content": selected_prompt_text},
                                    {"role": "user", "content": [
                                        {"type": "text", "text": "Extract the INGREDIENTS section only."},
                                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_img}"}}
                                    ]}
                                ],
                                temperature=temperature_val
                            )
                            result = response.choices[0].message.content.strip()
                            if result and "IMAGE_UNREADABLE" not in result.upper():
                                extracted.append(result)
                        except Exception as e:
                            extracted.append(f"[ERROR processing image: {e}]")

                    combined_html = "\n".join(extracted).strip()
                    reference = row.get("full_ingredients", "")
                    match_flag = "Pass" if combined_html in reference else "Needs Review"

                    diff_prompt = [
                        {"role": "system", "content":
                            "You compare two ingredient strings (OCR vs. reference). "
                            "Return JSON only: {\"severity\":\"Minor|Major\",\"diff_explanation\":\"...\"} "
                            "Flag missing/incorrect allergens if present (<b> tags in OCR)."},
                        {"role": "user", "content": f"OCR_OUTPUT:\n{combined_html}\n\nREFERENCE:\n{reference}"}
                    ]
                    try:
                        diff_resp = client.chat.completions.create(model="gpt-4.1-mini",
                                                                    messages=diff_prompt,
                                                                    temperature=temperature_val)
                        diff_content = diff_resp.choices[0].message.content.strip()
                        diff_json = json.loads(diff_content)
                    except Exception as e:
                        diff_json = {"severity": "Major", "diff_explanation": f"[Error comparing: {e}]"}

                    results.append({
                        "extracted_ingredients": combined_html,
                        "comparison_result": match_flag,
                        "severity": diff_json.get("severity", ""),
                        "diff_explanation": diff_json.get("diff_explanation", "")
                    })
                    # progress update
                    progress = (idx + 1) / n_rows
                    progress_bar.progress(progress)
                    progress_text.markdown(
                        f"<h4 style='text-align:center; color:#4A4443;'>Processed {idx + 1} of {n_rows} rows ({progress*100:.1f}%)</h4>",
                        unsafe_allow_html=True
                    )
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number", value=progress*100,
                        number={'font': {'color': '#4A4443'}},
                        title={'text': "Progress", 'font': {'color': '#4A4443'}},
                        gauge={'axis': {'range': [0,100], 'tickcolor':'#4A4443','tickfont':{'color':'#4A4443'},'tickwidth':2,'ticklen':8},
                               'bar': {'color': "#C2EA46"}, 'bgcolor': "#E1FAD1", 'borderwidth':1,
                               'steps':[{'range':[0,50],'color':"#E1FAD1"},{'range':[50,100],'color':"#F2FAF4"}]},
                        domain={'x':[0,1],'y':[0,1]}
                    ))
                    gauge_placeholder.plotly_chart(fig, use_container_width=True)
                    continue

                # ── General CSV branches ─────────────────────────────────────
                try:
                    # Competitor SKU Match
                    if prompt_choice == "Competitor SKU Match":
                        try:
                            my_sku    = parse_sku(row[sku_col])
                            cands_raw = top_candidates(my_sku, db=COMP_DB, k=8)   # [(ParsedSKU, score)]
                            cand_list = [c for c, _ in cands_raw]
                            status_placeholder.info(f"Row {idx+1}/{n_rows}: running fuzzy match…")
                            status_placeholder.success(f"Row {idx+1}/{n_rows}: found {len(cands_raw)} candidate(s)")

                            if not cand_list:
                                results.append({
                                    "match_found": "No",
                                    "best_match_uid": "",
                                    "best_match_name": "",
                                    "confidence_pct": 0,
                                    "reason": "No candidate met minimum fuzzy+size rules"
                                })
                                # progress update
                                progress = (idx + 1) / n_rows
                                progress_bar.progress(progress)
                                progress_text.markdown(
                                    f"<h4 style='text-align:center; color:#4A4443;'>Processed {idx + 1} of {n_rows} rows ({progress*100:.1f}%)</h4>",
                                    unsafe_allow_html=True
                                )
                                fig = go.Figure(go.Indicator(
                                    mode="gauge+number", value=progress*100,
                                    number={'font': {'color': '#4A4443'}},
                                    title={'text': "Progress", 'font': {'color': '#4A4443'}},
                                    gauge={'axis': {'range': [0,100], 'tickcolor':'#4A4443','tickfont':{'color':'#4A4443'},'tickwidth':2,'ticklen':8},
                                           'bar': {'color': "#C2EA46"}, 'bgcolor': "#E1FAD1", 'borderwidth':1,
                                           'steps':[{'range':[0,50],'color':"#E1FAD1"},{'range':[50,100],'color':"#F2FAF4"}]},
                                    domain={'x':[0,1],'y':[0,1]}
                                ))
                                gauge_placeholder.plotly_chart(fig, use_container_width=True)
                                continue

                            system_prompt = build_match_prompt(my_sku, cand_list)
                            status_placeholder.info(f"Row {idx+1}/{n_rows}: calling GPT to pick best match…")
                            resp = client.chat.completions.create(
                                model=model_choice,
                                messages=[{"role": "system", "content": system_prompt}],
                                temperature=temperature_val, top_p=0
                            )
                            gpt_json = json.loads(resp.choices[0].message.content)
                            status_placeholder.success(f"Row {idx+1}/{n_rows}: GPT done (match_found={gpt_json.get('match_found')})")

                            best = next((c for c in cand_list if c.uid == gpt_json.get("best_match_uid")), None)
                            results.append({
                                **gpt_json,
                                "best_match_uid": getattr(best, "uid", ""),
                                "best_match_name": getattr(best, "raw_name", ""),
                                "candidate_debug": [(c.raw_name, s) for c, s in cands_raw]
                            })
                        except Exception as e:
                            failed_rows.append(idx)
                            results.append({"error": f"Row {idx}: {e}"})
                        # progress update
                        progress = (idx + 1) / n_rows
                        progress_bar.progress(progress)
                        progress_text.markdown(
                            f"<h4 style='text-align:center; color:#4A4443;'>Processed {idx + 1} of {n_rows} rows ({progress*100:.1f}%)</h4>",
                            unsafe_allow_html=True
                        )
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number", value=progress*100,
                            number={'font': {'color': '#4A4443'}},
                            title={'text': "Progress", 'font': {'color': '#4A4443'}},
                            gauge={'axis': {'range': [0,100], 'tickcolor':'#4A4443','tickfont':{'color':'#4A4443'},'tickwidth':2,'ticklen':8},
                                   'bar': {'color': "#C2EA46"}, 'bgcolor': "#E1FAD1", 'borderwidth':1,
                                   'steps':[{'range':[0,50],'color':"#E1FAD1"},{'range':[50,100],'color':"#F2FAF4"}]},
                            domain={'x':[0,1],'y':[0,1]}
                        ))
                        gauge_placeholder.plotly_chart(fig, use_container_width=True)
                        continue

                    # Banned/Restricted Checker (prescreened)
                    if prompt_choice == "Banned/Restricted Checker":
                        try:
                            ing_text = str(row_data.get(banned_ing_col, "") or "").strip()
                            if not ing_text:
                                results.append({
                                    "overall": {"banned_present": False, "restricted_present": False},
                                    "items": [],
                                    "explanation": f"No ingredients in '{banned_ing_col}'",
                                    "candidates_debug": []
                                })
                            else:
                                cands = prescreen_map.get(idx, []) if isinstance(globals().get('prescreen_map'), dict) else []
                                if not cands:
                                    results.append({
                                        "overall": {"banned_present": False, "restricted_present": False},
                                        "items": [],
                                        "explanation": "No candidates found via substring/fuzzy pre-screen.",
                                        "candidates_debug": []
                                    })
                                else:
                                    system_txt = build_banned_prompt(cands, ing_text)
                                    resp = client.chat.completions.create(
                                        model=model_choice,
                                        messages=[{"role": "system", "content": system_txt}, {"role": "user", "content": ""}],
                                        temperature=temperature_val, top_p=0
                                    )
                                    content = resp.choices[0].message.content.strip()
                                    try:
                                        parsed = json.loads(clean_gpt_json_block(content))
                                    except Exception as e:
                                        parsed = {"error": f"JSON parse failed: {e}", "raw_output": content}
                                    parsed["candidates_debug"] = cands
                                    results.append(parsed)
                        except Exception as e:
                            failed_rows.append(idx)
                            results.append({"error": f"Row {idx} (Banned/Restricted): {e}"})
                        # progress update
                        progress = (idx + 1) / n_rows
                        progress_bar.progress(progress)
                        progress_text.markdown(
                            f"<h4 style='text-align:center; color:#4A4443;'>Processed {idx + 1} of {n_rows} rows ({progress*100:.1f}%)</h4>",
                            unsafe_allow_html=True
                        )
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number", value=progress*100,
                            number={'font': {'color': '#4A4443'}},
                            title={'text': "Progress", 'font': {'color': '#4A4443'}},
                            gauge={'axis': {'range': [0,100], 'tickcolor':'#4A4443','tickfont':{'color':'#4A4443'},'tickwidth':2,'ticklen':8},
                                   'bar': {'color': "#C2EA46"}, 'bgcolor': "#E1FAD1", 'borderwidth':1,
                                   'steps':[{'range':[0,50],'color':"#E1FAD1"},{'range':[50,100],'color':"#F2FAF4"}]},
                            domain={'x':[0,1],'y':[0,1]}
                        ))
                        gauge_placeholder.plotly_chart(fig, use_container_width=True)
                        continue

                    # Novel Food Checker (EU)
                    if prompt_choice == "Novel Food Checker (EU)":
                        from prompts.novel_check_utils import find_novel_matches, build_novel_food_prompt
                        ing_text = row_data.get("full_ingredients", "")
                        if not ing_text:
                            results.append({
                                "novel_food_flag": "No",
                                "confirmed_matches": [],
                                "explanation": "No 'full_ingredients' field provided.",
                                "fuzzy_debug_matches": []
                            })
                        else:
                            matches_with_scores = find_novel_matches(ing_text, threshold=fuzzy_threshold, return_scores=True)
                            candidate_matches = [term for term, _ in matches_with_scores]
                            debug_scores = matches_with_scores
                            if candidate_matches:
                                system_txt = build_novel_food_prompt(candidate_matches, ing_text)
                                user_txt = ""
                            else:
                                results.append({
                                    "novel_food_flag": "No",
                                    "confirmed_matches": [],
                                    "explanation": "No potential matches found via fuzzy/substring match.",
                                    "fuzzy_debug_matches": debug_scores
                                })
                                # progress update
                                progress = (idx + 1) / n_rows
                                progress_bar.progress(progress)
                                progress_text.markdown(
                                    f"<h4 style='text-align:center; color:#4A4443;'>Processed {idx + 1} of {n_rows} rows ({progress*100:.1f}%)</h4>",
                                    unsafe_allow_html=True
                                )
                                fig = go.Figure(go.Indicator(
                                    mode="gauge+number", value=progress*100,
                                    number={'font': {'color': '#4A4443'}},
                                    title={'text': "Progress", 'font': {'color': '#4A4443'}},
                                    gauge={'axis': {'range': [0,100], 'tickcolor':'#4A4443','tickfont':{'color':'#4A4443'},'tickwidth':2,'ticklen':8},
                                           'bar': {'color': "#C2EA46"}, 'bgcolor': "#E1FAD1", 'borderwidth':1,
                                           'steps':[{'range':[0,50],'color':"#E1FAD1"},{'range':[50,100],'color':"#F2FAF4"}]},
                                    domain={'x':[0,1],'y':[0,1]}
                                ))
                                gauge_placeholder.plotly_chart(fig, use_container_width=True)
                                continue

                            # fallthrough to common call below
                    # Common JSON prompt (incl. Novel Food if candidate_matches exist)
                    if "USER MESSAGE:" in user_prompt:
                        system_txt, user_txt = user_prompt.split("USER MESSAGE:", 1)
                    else:
                        system_txt, user_txt = user_prompt, ""
                    system_txt = system_txt.replace("SYSTEM MESSAGE:", "").strip()
                    user_txt = user_txt.strip().format(**row_data)
                    user_txt += f"\n\nSelected fields:\n{json.dumps(row_data, ensure_ascii=False)}"

                    response = client.chat.completions.create(
                        model=model_choice,
                        messages=[{"role": "system", "content": system_txt},
                                  {"role": "user",   "content": user_txt}],
                        temperature=temperature_val, top_p=0
                    )
                    content = response.choices[0].message.content.strip()
                    if content.startswith("```"):
                        parts = content.split("```", maxsplit=2)
                        content = parts[1].lstrip("json").strip().split("```")[0].strip()

                    parsed = json.loads(content)
                    if prompt_choice == "Novel Food Checker (EU)":
                        # Attach debug scores if they exist in scope
                        try:
                            parsed["fuzzy_debug_matches"] = debug_scores  # noqa
                        except Exception:
                            pass
                    results.append(parsed)

                except Exception as e:
                    failed_rows.append(idx)
                    results.append({"error": f"Failed to process row {idx}: {e}", "raw_output": content or "No content returned"})

                # ── progress + rolling log/gauge ───────────────────────────
                progress = (idx + 1) / n_rows
                progress_bar.progress(progress)
                progress_text.markdown(
                    f"<h4 style='text-align:center; color:#4A4443;'>Processed {idx + 1} of {n_rows} rows ({progress*100:.1f}%)</h4>",
                    unsafe_allow_html=True
                )
                if "rolling_log_dicts" not in st.session_state:
                    st.session_state.rolling_log_dicts = []
                st.session_state.rolling_log_dicts.append(results[-1])
                st.session_state.rolling_log_dicts = st.session_state.rolling_log_dicts[-20:]

                log_placeholder.empty()
                log_placeholder.markdown("<h4 style='color:#4A4443;'>📝 Recent Outputs (Last 20)</h4>", unsafe_allow_html=True)
                for entry in st.session_state.rolling_log_dicts[-3:]:
                    log_placeholder.json(entry)
                for i, entry in enumerate(st.session_state.rolling_log_dicts):
                    row_num = (idx + 1) - (len(st.session_state.rolling_log_dicts) - i)
                    with log_placeholder.expander(f"Row {row_num} output", expanded=True):
                        st.json(entry)

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=progress * 100,
                    number={'font': {'color': '#4A4443'}},
                    title={'text': "Progress", 'font': {'color': '#4A4443'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': '#4A4443', 'tickfont': {'color': '#4A4443'},
                                 'tickwidth': 2, 'ticklen': 8},
                        'bar': {'color': "#C2EA46"},
                        'bgcolor': "#E1FAD1",
                        'borderwidth': 1,
                        'steps': [{'range': [0, 50], 'color': "#E1FAD1"},
                                  {'range': [50, 100], 'color': "#F2FAF4"}]
                    },
                    domain={'x': [0, 1], 'y': [0, 1]}
                ))
                gauge_placeholder.plotly_chart(fig, use_container_width=True)

            # ---------- end loop ----------
            results_df = pd.DataFrame(results)
            final_df   = pd.concat([df.reset_index(drop=True), results_df], axis=1)
            st.success("✅ GPT processing complete!")

            st.markdown("<h3 style='color:#005A3F;'>🔍 Final Result</h3>", unsafe_allow_html=True)
            final_df = final_df.applymap(_flatten).astype(str)

            max_preview = st.number_input(
                "How many rows would you like to preview?",
                min_value=1, max_value=min(1000, len(final_df)), value=min(20, len(final_df)), step=1
            )
            preview_df = final_df.head(int(max_preview))
            st_dataframe_safe(preview_df, use_container_width=True)

            # After building preview_df but before st_dataframe_safe()
            if len(set(preview_df.columns)) != len(preview_df.columns):
                st.warning("Detected duplicate column names; suffixes have been added for display.")
            
            st.download_button("⬇️ Download Full Results CSV",
                               final_df.to_csv(index=False).encode("utf-8"),
                               "gpt_output.csv", "text/csv")

            if failed_rows:
                failed_df = df.iloc[failed_rows].copy()
                st.warning(f"{len(failed_rows)} rows failed to process. You can download them and retry.")
                st.download_button("⬇️ Download Failed Rows CSV",
                                   failed_df.to_csv(index=False).encode("utf-8"),
                                   "gpt_failed_rows.csv", "text/csv")
