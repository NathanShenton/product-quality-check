# app.py
import streamlit as st
import pandas as pd
import json
import os
import requests
import base64
import plotly.graph_objects as go
from streamlit_cropper import st_cropper
from PIL import Image
import io
from openai import OpenAI

# ── H&B prompt modules (unchanged imports) ─────────────────────────────────────
from prompts.artwork_processing_ingredients import process_artwork
from prompts.artwork_processing_directions import process_artwork_directions
from prompts.artwork_processing_warnings_advisory import process_artwork_warnings_advisory
from prompts.artwork_processing_packsize_nutrition import (
    process_artwork_packsize,
    process_artwork_nutrition,
)
from sidebar import render_sidebar
from style import inject_css
from prompts.bannedingredients import find_banned_matches, build_banned_prompt
from prompts.artwork_processing_supplier_addresses import process_artwork as process_artwork_suppliers

from prompts.prompts import PROMPT_OPTIONS
from prompts.competitor_match import (
    parse_sku,
    top_candidates,
    build_match_prompt,
)

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

# ── New: resilience helpers ────────────────────────────────────────────────────
from utils.utils_retries import safe_chat_completion   # retrying OpenAI wrapper
from utils.durability import (                         # checkpoint + resume
    new_job_dir, write_json, append_csv, read_existing_results, heartbeat
)
import streamlit.components.v1 as components           # keep-awake
from functools import partial

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

# ─── Streamlit page config ─────────────────────
st.set_page_config(page_title="Flexible AI Product Data Checker (Durable)", layout="wide")

# ── Keep the browser awake & ping the server (reduces silent disconnects)
components.html("""
<script>
let wakeLock=null;
async function req(){try{wakeLock=await navigator.wakeLock.request('screen');}catch(e){}}
document.addEventListener('visibilitychange',()=>{if(document.visibilityState==='visible')req();});
req();
setInterval(()=>{fetch(window.location.href,{method:'HEAD',cache:'no-store'}).catch(()=>{});},25000);
</script>
""", height=0)

inject_css()
render_sidebar()

#############################
#   Helper: Approx. Tokens  #
#############################
def approximate_tokens(text: str) -> int:
    """Approximate the number of tokens based on text length."""
    return max(1, len(text) // 4)

#############################
#   Cost Estimation         #
#############################
def estimate_cost(model: str, df: pd.DataFrame, user_prompt: str, cols_to_use: list) -> float:
    """
    Estimate cost based on the chosen model, number of rows,
    approximate tokens in user prompt, and row data.
    """
    model_costs_per_1k = {
        "gpt-3.5-turbo": (0.0005, 0.002),
        "gpt-4.1-mini":  (0.0004, 0.0016),
        "gpt-4.1-nano":  (0.0001, 0.0004),
        "gpt-4o-mini":   (0.00015, 0.0006),
        "gpt-4o":        (0.005,  0.015),  # May 2024 reference
        "gpt-4-turbo":   (0.01,   0.03)
    }

    # Default fallback costs
    cost_in, cost_out = model_costs_per_1k.get(model, (0.001, 0.003))
    total_input_tokens = 0
    total_output_tokens = 0

    for _, row in df.iterrows():
        system_tokens = 30
        row_dict = {c: row.get(c, "") for c in cols_to_use}
        row_json_str = json.dumps(row_dict, ensure_ascii=False)
        prompt_tokens = approximate_tokens(user_prompt) + approximate_tokens(row_json_str)
        total_input_tokens += (system_tokens + prompt_tokens)
        # Estimate ~100 tokens output per row
        total_output_tokens += 100

    input_ktokens = total_input_tokens / 1000
    output_ktokens = total_output_tokens / 1000
    return (input_ktokens * cost_in) + (output_ktokens * cost_out)

# JSON CLEANER #
def clean_gpt_json_block(text: str) -> str:
    """
    Strips markdown-style ``` wrappers and leading prose to return clean JSON string.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"```$", "", text.strip(), flags=re.IGNORECASE)
    json_start = text.find("{")
    if json_start != -1:
        text = text[json_start:]
    return text.strip()

def _flatten(x):
    """Make values serializable for CSV download preview (PyArrow friendliness)."""
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
    "gpt-4o":        "Multimodal GPT-4 (images + text).",
    "gpt-4-turbo":   "Powerful & pricier — use for complex tasks."
}

# ---- Main Page Layout ----
st.markdown(
    "<h1>📄 Flexible AI Product Data Checker — Durable & Resumable</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; font-size:16px; color:#4A4443;'>"
    "Now with checkpointing, retries, resume, and keep-awake for long runs."
    "</p>",
    unsafe_allow_html=True
)

# Using columns to separate the API key entry and file upload
col1, col2 = st.columns(2)
with col1:
    # 1. API Key Entry
    api_key_input = st.text_input("🔑 Enter your OpenAI API Key", type="password")
    if not api_key_input:
        st.warning("Please enter your OpenAI API key to proceed.")
        st.stop()
    # Initialize the OpenAI client and bind the retry helper
    client = OpenAI(api_key=api_key_input)
    ai = partial(safe_chat_completion, client=client)  # <-- bind client ONCE

# ------------------------------------------------------------------
# Two/Three-pass image ingredient extractor (uses bound `ai`)
# ------------------------------------------------------------------
def two_pass_extract(image_bytes: bytes, temperature_val: float = 0.0) -> str:
    """
    Run GPT-4o three times:
      • Pass-1: OCR of ingredients panel
      • Pass-2: Format and bold allergens
      • Pass-3: Double-check and correct OCR misreads
    """
    import base64, textwrap
    data_url = f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode()}"

    # ---- PASS 1 ----
    pass1_sys = (
        "You are a specialist OCR engine. Extract the EXACT text of the INGREDIENTS "
        "panel on a UK food label image. Preserve punctuation, %, brackets. "
        "If the section is unreadable, output IMAGE_UNREADABLE."
    )
    content = ai(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": pass1_sys},
            {"role": "user",   "content": [
                {"type": "text", "text": "Label image incoming."},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]}
        ],
        temperature=temperature_val, top_p=0, timeout=90
    )
    raw = (content or "").strip()
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
        provided by the user exactly as HTML and bold (**<b>…</b>**) every word
        that matches this UK-FIC allergen list:

        {allergens}

        • Bold ONLY the allergen token(s); keep all other text unchanged.
        • Do NOT re-order, translate, or summarise.
        • Return the HTML string only – no markdown, no commentary.
    """).strip()
    html_out = ai(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": pass2_sys},
            {"role": "user",   "content": raw}
        ],
        temperature=temperature_val, top_p=0, timeout=90
    )

    # ---- PASS 3 ----
    pass3_sys = (
        "You previously extracted an ingredient list from a UK food label image. "
        "Please double-check for spelling errors or OCR misreads in these items. "
        "If any corrections are needed, return the corrected string, preserving original HTML formatting. "
        "Otherwise return the input unchanged."
    )
    final_out = ai(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": pass3_sys},
            {"role": "user",   "content": html_out}
        ],
        temperature=temperature_val, top_p=0, timeout=90
    )
    return (final_out or "").strip()

def fetch_image_as_base64(url: str) -> str | None:
    """Fetch an image (with timeout) and return base64 or None on failure."""
    try:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url.strip().lstrip("/")
        response = requests.get(url, timeout=20)  # timeout added
        response.raise_for_status()
        return base64.b64encode(response.content).decode("utf-8")
    except Exception:
        return None

# 4. Choose a Pre-Written Prompt
st.subheader("💬 Choose a Prompt")

ARTWORK_AUTO_PROMPT = "Artwork: Ingredient Statement (PDF/JPEG)"
ARTWORK_DIRECTIONS_PROMPT = "Artwork: Directions for Use (PDF/JPEG)"
ARTWORK_PACKSIZE_PROMPT = "Artwork: Pack Size / Net & Gross Weight (PDF/JPEG)"
ARTWORK_NUTRITION_PROMPT = "Artwork: Nutrition Facts (PDF/JPEG)"
ARTWORK_SUPPLIER_PROMPT = "Artwork: Supplier Addresses (UK/EU) (PDF/JPEG)"
ARTWORK_RUN_ALL_PROMPT = "Artwork: Run ALL Packaging Pipelines (PDF/JPEG)"
ARTWORK_WARNINGS_ADVISORY_PROMPT = "Artwork: Warnings & Advisory (PDF/JPEG)"

GREEN_CLAIMS_PROMPT = "Green Claims Checker (Language-aware)"

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

prompt_choice = st.selectbox(
    "Select a pre-written prompt or 'Custom':",
    prompt_names,
    index=0
)

# ─ Extract chosen prompt details ─
if prompt_choice == ARTWORK_AUTO_PROMPT:
    selected_prompt_text = "SYSTEM MESSAGE: handled by artwork_processing module"
    prompt_description   = "Upload PDF/JPG/PNG label artwork. Auto-finds the Ingredients panel, returns exact text + HTML with allergens bolded."
    recommended_model    = "gpt-4o"
elif prompt_choice == ARTWORK_DIRECTIONS_PROMPT:
    selected_prompt_text = "SYSTEM MESSAGE: handled by artwork_processing (directions) module"
    prompt_description   = "Upload PDF/JPG/PNG artwork. Auto-finds Directions/Usage/Preparation, extracts exact text, structures steps/timings/temps/volumes, and tags pictograms."
    recommended_model    = "gpt-4o"
elif prompt_choice == ARTWORK_PACKSIZE_PROMPT:
    selected_prompt_text = "SYSTEM MESSAGE: handled by artwork_processing (pack size/weights) module"
    prompt_description   = (
        "Upload PDF/JPG/PNG artwork. Auto-finds the main net quantity/pack-size statement, "
        "parses Number of items, Base quantity, Unit of measure, and extracts Net/Gross/Drained weight + ℮ if present."
    )
    recommended_model    = "gpt-4o"
elif prompt_choice == ARTWORK_NUTRITION_PROMPT:
    selected_prompt_text = "SYSTEM MESSAGE: handled by artwork_processing (nutrition) module"
    prompt_description   = "Upload PDF/JPG/PNG artwork. Auto-locates the nutrition panel and returns structured JSON + flat key/value rows."
    recommended_model    = "gpt-4o"
elif prompt_choice == ARTWORK_SUPPLIER_PROMPT:
    selected_prompt_text = "SYSTEM MESSAGE: handled by artwork_processing (supplier addresses) module"
    prompt_description   = (
        "Upload PDF/JPG/PNG artwork. Auto-finds Supplier/Responsible Person blocks and "
        "extracts exact text for **UK** and **EU** addresses separately, plus bounding boxes."
    )
    recommended_model    = "gpt-4o"
elif prompt_choice == ARTWORK_RUN_ALL_PROMPT:
    selected_prompt_text = "SYSTEM MESSAGE: handled by run-all packaging pipelines module"
    prompt_description   = (
        "Upload PDF/JPG/PNG artwork. Runs Ingredients → Directions → Pack Size → "
        "Nutrition → Supplier (and Warnings). Choose combined JSON or ZIP bundle."
    )
    recommended_model    = "gpt-4o"
elif prompt_choice == GREEN_CLAIMS_PROMPT:
    selected_prompt_text = "SYSTEM MESSAGE: handled by green_claims module"
    prompt_description   = (
        "Checks product marketing text against a curated Green Claims library using fuzzy matching "
        "and an AI adjudicator. Language selection filters candidates to the chosen variant only."
    )
    recommended_model    = "gpt-4.1-mini"
elif prompt_choice == ARTWORK_WARNINGS_ADVISORY_PROMPT:
    selected_prompt_text = "SYSTEM MESSAGE: handled by artwork_processing (warnings/advisory) module"
    prompt_description   = (
        "Upload PDF/JPG/PNG artwork. Auto-locates warnings/safety/advisory text and returns two verbatim lists."
    )
    recommended_model    = "gpt-4o"
else:
    selected = PROMPT_OPTIONS[prompt_choice]
    selected_prompt_text = selected["prompt"]
    recommended_model    = selected["recommended_model"]
    prompt_description   = selected["description"]

st.markdown(f"**Prompt Info:** {prompt_description}")

# ─ Session-state to reset crops when the prompt changes ─
if "last_prompt" not in st.session_state:
    st.session_state["last_prompt"] = prompt_choice
if "cropped_bytes" not in st.session_state:
    st.session_state["cropped_bytes"] = None
if st.session_state["last_prompt"] != prompt_choice:
    st.session_state["last_prompt"] = prompt_choice
    st.session_state["cropped_bytes"] = None

# ─ Ensure sliders exist for specific prompts ─
fuzzy_threshold = 87
if prompt_choice == "Novel Food Checker (EU)":
    fuzzy_threshold = st.slider(
        "Novel-food fuzzy threshold",
        min_value=70, max_value=100, value=87,
        help="Lower = catch more variants (but watch for false positives)."
    )

if prompt_choice == GREEN_CLAIMS_PROMPT:
    gc_threshold = st.slider(
        "Green-claims fuzzy threshold",
        min_value=70, max_value=100, value=85,
        help="Lower = catch more variants (but watch for false positives)."
    )

    gc_language = st.selectbox(
        "Claim language",
        options=list(LANG_TO_COL.keys()),
        index=list(LANG_TO_COL.keys()).index("Dutch (NL)") if "Dutch (NL)" in LANG_TO_COL else 0,
        help="Only candidates from this language column will be considered."
    )

    gc_upload = st.file_uploader(
        "Upload your green-claims-database.csv (⚠️ not your product CSV)",
        type=["csv"],
        key="gc_db"
    )

    # Load DB
    try:
        GC_DB = load_green_claims_db(uploaded_file=gc_upload)
    except Exception as e:
        st.error(f"Could not load green-claims database: {e}")
        st.stop()

    # Detect mis-upload
    PRODUCT_MARKERS = {"SKU ID", "SKU Name", "Product Name"}
    if PRODUCT_MARKERS.issubset(set(GC_DB.columns)):
        st.error("It looks like you uploaded your PRODUCT CSV into the Green-Claims DB uploader. Please upload the DB here instead.")
        st.stop()

    # Validate language col
    lang_col = LANG_TO_COL.get(gc_language)
    if not lang_col:
        st.error(f"No mapping found in LANG_TO_COL for '{gc_language}'.")
        st.stop()
    if lang_col not in GC_DB.columns:
        st.error(f"Language column '{lang_col}' not found. Available: {list(GC_DB.columns)}")
        st.stop()

    non_empty = (GC_DB[lang_col].astype(str).str.strip() != "").sum()
    st.caption(f"Green-claims DB loaded: {len(GC_DB):,} rows • {gc_language} '{lang_col}' non-empty: {non_empty:,}")
    if non_empty == 0:
        st.warning(f"No text present in '{lang_col}'. If your CSV uses semicolons, ensure the loader detects it.")
        st.stop()

    gc_debug = st.checkbox("🔍 Show Green Claims matcher debug (first 10 rows)", value=False)
    if gc_debug:
        st.dataframe(GC_DB[[lang_col]].head(10))

# Slider for the Banned/Restricted checker
if prompt_choice == "Banned/Restricted Checker":
    banned_fuzzy_threshold = st.slider(
        "Banned/Restricted fuzzy threshold",
        min_value=80, max_value=100, value=90,
        help="Lower = catch more variants (but watch false positives)."
    )

# --- Determine if image-based (single-image cropping prompts only) ---
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

# Force gpt-4o if image prompt is selected
if is_image_prompt:
    recommended_model = "gpt-4o"

# 5. Model & Temperature Selector (default model comes from the prompt metadata)
all_model_keys  = list(MODEL_OPTIONS.keys())
default_index   = all_model_keys.index(recommended_model) if recommended_model in all_model_keys else 0

model_choice = st.selectbox(
    "🧠 Choose GPT model",
    all_model_keys,
    index=default_index
)

st.markdown(f"**Model Info:** {MODEL_OPTIONS[model_choice]}")

# 🔥 Temperature slider – default 0.00 (fully deterministic)
temperature_val = st.slider(
    "🎛️ Model temperature (0 = deterministic, 1 = very creative)",
    min_value=0.0, max_value=1.0, value=0.0, step=0.05
)

st.markdown("---")

# 6. User Prompt Text Area
user_prompt = st.text_area(
    "✍️ Your prompt for GPT",
    value=selected_prompt_text,
    height=200
)

# =========================================
# Automatic Artwork processing (unchanged behaviours; still early-return)
# =========================================
if prompt_choice == ARTWORK_AUTO_PROMPT:
    st.markdown("### 📄 Upload artwork (PDF/JPG/PNG) – no manual crop")
    art_file = st.file_uploader("Choose file", type=["pdf","jpg","jpeg","png"], key="art_auto")

    if model_choice != "gpt-4o":
        st.warning("This prompt is designed for **gpt-4o** (vision). Please switch the model above.")

    run_auto = st.button("🚀 Extract Ingredients (Auto)")
    if art_file and run_auto:
        with st.spinner("Locating INGREDIENTS panel and extracting…"):
            try:
                res = process_artwork(
                    client=client,
                    file_bytes=art_file.read(),
                    filename=art_file.name,
                    render_dpi=350,
                    model="gpt-4o"
                )
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

            st.download_button(
                "⬇️ Download JSON",
                data=json.dumps(res, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="ingredients_result.json",
                mime="application/json"
            )
    st.stop()

if prompt_choice == ARTWORK_WARNINGS_ADVISORY_PROMPT:
    st.markdown("### 📄 Upload artwork (PDF/JPG/PNG) – auto-locate Warnings & Advisory")
    art_file = st.file_uploader("Choose file", type=["pdf","jpg","jpeg","png"], key="art_warnadv_auto")

    if model_choice != "gpt-4o":
        st.warning("This prompt is designed for **gpt-4o** (vision). Please switch the model above.")

    run_auto = st.button("🚀 Extract Warnings & Advisory (Auto)")
    if art_file and run_auto:
        with st.spinner("Locating WARNINGS/ADVISORY text and extracting…"):
            try:
                res = process_artwork_warnings_advisory(
                    client=client,
                    file_bytes=art_file.read(),
                    filename=art_file.name,
                    render_dpi=350,
                    model="gpt-4o"
                )
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

            st.download_button(
                "⬇️ Download JSON",
                data=json.dumps(res, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="warnings_advisory_result.json",
                mime="application/json"
            )
    st.stop()

if prompt_choice == ARTWORK_RUN_ALL_PROMPT:
    st.markdown("### 📄 Upload artwork (PDF/JPG/PNG) – run **all** packaging pipelines in sequence")
    art_file = st.file_uploader("Choose file", type=["pdf","jpg","jpeg","png"], key="art_run_all")
    if model_choice != "gpt-4o":
        st.warning("This prompt is designed for **gpt-4o** (vision). Please switch the model above.")
    output_mode = st.radio("Output format", ["Single JSON (everything together)", "6 JSONs (zipped)"], index=0)

    run_all = st.button("🚀 Run ALL Pipelines (Ingredients → Directions → Pack Size → Nutrition → Supplier)")
    if art_file and run_all:
        file_bytes = art_file.read()
        filename   = art_file.name

        def run_step(fn, label):
            try:
                with st.spinner(f"Running {label}…"):
                    res = fn(client=client, file_bytes=file_bytes, filename=filename, render_dpi=350, model="gpt-4o")
                ok = bool(res.get("ok", True))
                if ok: st.success(f"✅ {label} complete")
                else:  st.error(f"❌ {label} failed: {res.get('error','Unknown error')}")
                return res
            except Exception as e:
                err = {"ok": False, "error": f"{label} exception: {e}"}
                st.error(f"❌ {label} crashed: {e}")
                return err

        ingredients_res   = run_step(process_artwork,               "Ingredients")
        directions_res    = run_step(process_artwork_directions,    "Directions")
        packsize_res      = run_step(process_artwork_packsize,      "Pack Size / Weights")
        nutrition_res     = run_step(process_artwork_nutrition,     "Nutrition")
        supplier_res      = run_step(process_artwork_suppliers,     "Supplier Addresses")
        warnings_adv_res  = run_step(process_artwork_warnings_advisory, "Warnings & Advisory")

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
            "ingredients_ok": bool(ingredients_res.get("ok", True)),
            "directions_ok":  bool(directions_res.get("ok", True)),
            "packsize_ok":    bool(packsize_res.get("ok", True)),
            "nutrition_ok":   bool(nutrition_res.get("ok", True)),
            "suppliers_ok":   bool(supplier_res.get("ok", True)),
            "warnings_advisory_ok": bool(warnings_adv_res.get("ok", True)),
        })

        if output_mode.startswith("Single"):
            data = json.dumps(combined, ensure_ascii=False, indent=2).encode("utf-8")
            st.download_button("⬇️ Download Combined JSON", data=data, file_name="packaging_pipelines_all.json", mime="application/json")
        else:
            import zipfile
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("ingredients_result.json", json.dumps(ingredients_res, ensure_ascii=False, indent=2))
                zf.writestr("directions_result.json",  json.dumps(directions_res,  ensure_ascii=False, indent=2))
                zf.writestr("packsize_result.json",    json.dumps(packsize_res,    ensure_ascii=False, indent=2))
                zf.writestr("nutrition_result.json",   json.dumps(nutrition_res,   ensure_ascii=False, indent=2))
                zf.writestr("supplier_addresses_result.json", json.dumps(supplier_res, ensure_ascii=False, indent=2))
                zf.writestr("warnings_advisory_result.json",  json.dumps(warnings_adv_res,ensure_ascii=False, indent=2))
                zf.writestr("packaging_pipelines_all.json", json.dumps(combined, ensure_ascii=False, indent=2))
            buf.seek(0)
            st.download_button("⬇️ Download ZIP (5 JSONs + combined)", data=buf.getvalue(), file_name="packaging_pipelines_all.zip", mime="application/zip")
    st.stop()

if prompt_choice == ARTWORK_SUPPLIER_PROMPT:
    st.markdown("### 📄 Upload artwork (PDF/JPG/PNG) – auto-locate Supplier/Responsible Person addresses")
    art_file = st.file_uploader("Choose file", type=["pdf","jpg","jpeg","png"], key="art_supplier_auto")

    if model_choice != "gpt-4o":
        st.warning("This prompt is designed for **gpt-4o** (vision). Please switch the model above.")

    run_auto = st.button("🚀 Extract Supplier Addresses (Auto)")
    if art_file and run_auto:
        with st.spinner("Locating supplier/Responsible Person address blocks and extracting…"):
            try:
                res = process_artwork_suppliers(client=client, file_bytes=art_file.read(), filename=art_file.name, render_dpi=350, model="gpt-4o")
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

            st.download_button(
                "⬇️ Download JSON",
                data=json.dumps(res, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="supplier_addresses_result.json",
                mime="application/json"
            )
    st.stop()

if prompt_choice == ARTWORK_PACKSIZE_PROMPT:
    st.markdown("### 📄 Upload artwork (PDF/JPG/PNG) – auto-locate Pack Size / Weights")
    art_file = st.file_uploader("Choose file", type=["pdf","jpg","jpeg","png"], key="art_packsize_auto")

    if model_choice != "gpt-4o":
        st.warning("This prompt is designed for **gpt-4o** (vision). Please switch the model above.")

    run_auto = st.button("🚀 Extract Pack Size / Weights (Auto)")
    if art_file and run_auto:
        with st.spinner("Locating PACK SIZE / NET QUANTITY and extracting…"):
            try:
                res = process_artwork_packsize(client=client, file_bytes=art_file.read(), filename=art_file.name, render_dpi=350, model="gpt-4o")
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
            with colA: st.metric("Number of items", str(parsed.get("number_of_items") or "—"))
            with colB: st.metric("Base quantity", str(parsed.get("base_quantity") or "—"))
            with colC: st.metric("Unit of measure", parsed.get("unit_of_measure") or "—")

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

            st.download_button(
                "⬇️ Download JSON",
                data=json.dumps(res, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="packsize_result.json",
                mime="application/json"
            )
    st.stop()

if prompt_choice == ARTWORK_NUTRITION_PROMPT:
    st.markdown("### 📄 Upload artwork (PDF/JPG/PNG) – auto-locate Nutrition panel")
    art_file = st.file_uploader("Choose file", type=["pdf","jpg","jpeg","png"], key="art_nutrition_auto")
    if model_choice != "gpt-4o":
        st.warning("This prompt is designed for **gpt-4o**.")
    if art_file and st.button("🚀 Extract Nutrition (Auto)"):
        with st.spinner("Finding nutrition table and extracting…"):
            try:
                res = process_artwork_nutrition(client=client, file_bytes=art_file.read(), filename=art_file.name, render_dpi=350, model="gpt-4o")
            except Exception as e:
                res = {"ok": False, "error": f"Processing failed: {e}"}

        if not res.get("ok"):
            st.error(res.get("error","Failed"))
        else:
            st.success("✅ Nutrition extracted")
            st.write(f"**Page:** {res['page_index']}")
            st.write(f"**BBox (pixels):** {res['bbox_pixels']}")
            st.markdown("#### Flat rows (easy export)")
            st.dataframe(pd.DataFrame(res.get("flat", [])))
            st.markdown("#### Structured JSON")
            st.json(res.get("parsed", {}))
            st.markdown("#### QA")
            st.json(res.get("qa", {}))
            st.download_button(
                "⬇️ Download Nutrition JSON",
                data=json.dumps(res, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="nutrition_result.json",
                mime="application/json"
            )
    st.stop()

if prompt_choice == ARTWORK_DIRECTIONS_PROMPT:
    st.markdown("### 📄 Upload artwork (PDF/JPG/PNG) – auto-locate Directions/Usage/Preparation")
    art_file = st.file_uploader("Choose file", type=["pdf","jpg","jpeg","png"], key="art_directions_auto")

    if model_choice != "gpt-4o":
        st.warning("This prompt is designed for **gpt-4o** (vision). Please switch the model above.")

    run_auto = st.button("🚀 Extract Directions (Auto)")
    if art_file and run_auto:
        with st.spinner("Locating DIRECTIONS/USAGE/PREPARATION panel and extracting…"):
            try:
                res = process_artwork_directions(client=client, file_bytes=art_file.read(), filename=art_file.name, render_dpi=350, model="gpt-4o")
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

            st.download_button(
                "⬇️ Download JSON",
                data=json.dumps(res, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="directions_result.json",
                mime="application/json"
            )
    st.stop()

# -------------------------------
# Image uploader and crop logic
# -------------------------------
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

            st.download_button(
                label="⬇️ Download Cropped Image Sent to GPT",
                data=st.session_state["cropped_bytes"],
                file_name="cropped_label.png",
                mime="image/png"
            )

# ────────────────────────────────────────────────
# Non-image prompts get the product-CSV uploader
# ────────────────────────────────────────────────
if not is_image_prompt:
    uploaded_file = st.file_uploader("📁 Upload your product CSV", type=["csv"], key="data_csv")

# ---------------------------------------------------------------
# Image-prompt flow – two/three-pass high-accuracy extraction (single-image)
# ---------------------------------------------------------------
if is_image_prompt and st.session_state.get("cropped_bytes"):
    st.markdown("### 📤 Processing image…")
    with st.spinner("Running high-accuracy multi-pass extraction"):
        if model_choice != "gpt-4o":
            st.error("🛑  Image prompts require the **gpt-4o** model. Please choose it above and try again.")
            st.stop()
        try:
            if "Ingredient Scrape" in prompt_choice:
                html_out = two_pass_extract(st.session_state["cropped_bytes"], temperature_val)
            else:
                data_url = "data:image/jpeg;base64," + base64.b64encode(st.session_state["cropped_bytes"]).decode()
                system_msg = user_prompt.replace("SYSTEM MESSAGE:", "").strip()
                content = ai(
                    model=model_choice,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": [
                            {"type": "text", "text": "Cropped label image below."},
                            {"type": "image_url", "image_url": {"url": data_url}}
                        ]}
                    ],
                    temperature=temperature_val, top_p=0, timeout=90
                )
                html_out = (content or "").strip()

            if html_out == "IMAGE_UNREADABLE":
                st.error("🛑  The image was unreadable or missing the required section.")
            else:
                st.success("✅ GPT image processing complete!")
                output_type = "html"
                if "Directions" in prompt_choice or "Storage" in prompt_choice: output_type = "text"
                elif "Warnings and Advisory" in prompt_choice: output_type = "json"
                st.code(html_out, language=output_type)
        except Exception as e:
            st.error(f"Image processing failed: {e}")

# ---------- Main Execution Logic: DURABLE CSV RUN ----------
if (not is_image_prompt) and uploaded_file and (
    user_prompt.strip() or
    prompt_choice in {"Novel Food Checker (EU)", "Competitor SKU Match", "GHS Pictogram Detector", "Banned/Restricted Checker", GREEN_CLAIMS_PROMPT}
):
    df = pd.read_csv(uploaded_file, dtype=str).fillna("")
    st.markdown("### 📄 CSV Preview")
    st.dataframe(df.head())

    # 3. Dynamic Column Selector (up to 10 columns)
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

    # Special per-prompt UI
    if prompt_choice == "Competitor SKU Match":
        sku_col = st.selectbox(
            "Which column contains *your* product name / volume?",
            options=selected_columns,
            help="e.g. 'Product_Name' or 'SKU Title'"
        )

    if prompt_choice == "Banned/Restricted Checker":
        banned_ing_col = st.selectbox(
            "Which column contains the full ingredients text?",
            options=selected_columns,
            help="Pick the column with the product’s ingredient statement."
        )

    # Cost estimate
    st.markdown(
        f"""
        <div style='padding:10px; background-color:#FFFFFF; color:#4A4443; border-radius:5px; margin-bottom:1rem;'>
            <strong>Estimated Cost:</strong> ${estimate_cost(model_choice, df, user_prompt, selected_columns):0.4f}
        </div>
        """, unsafe_allow_html=True
    )

    # Create gauge placeholder before starting the loop
    gauge_placeholder = st.empty()

    if st.button("🚀 Run GPT on CSV (durable)"):
        # Build a run folder + resume state
        job_dir = new_job_dir()  # e.g. runs/20251003_141530
        df.to_csv(os.path.join(job_dir, "input_snapshot.csv"), index=False, encoding="utf-8")
        cfg = {"model": model_choice, "temperature": float(temperature_val), "prompt_choice": prompt_choice, "selected_columns": selected_columns}
        write_json(os.path.join(job_dir, "config.json"), cfg)

        results_path   = os.path.join(job_dir, "results.csv")
        failures_path  = os.path.join(job_dir, "failures.csv")
        heartbeat_path = os.path.join(job_dir, "heartbeat.txt")
        log_path       = os.path.join(job_dir, "rolling_log.jsonl")

        existing = read_existing_results(results_path)
        done_idx = set(existing["__row_index"].astype(int)) if "__row_index" in existing.columns else set()
        st.info(f"📦 Run folder: {job_dir}" + (f" • Resuming: {len(done_idx)} row(s) already done." if done_idx else ""))

        # Placeholders
        progress_bar = st.progress(0)
        progress_text = st.empty()
        log_placeholder = st.empty()
        status_placeholder = st.empty()

        n_rows = len(df)
        BATCH_SIZE = 20
        batch_rows, batch_fail = [], []

        abort_toggle = st.checkbox("🛑 Allow manual abort (saves progress)")
        abort_now = st.empty()

        # Processing loop
        for idx, row in df.iterrows():
            if idx in done_idx:
                # Already processed
                progress = (idx + 1) / n_rows
                progress_bar.progress(progress)
                progress_text.markdown(f"<h4 style='text-align:center; color:#4A4443;'>Skipped {idx + 1} of {n_rows} (already done)</h4>", unsafe_allow_html=True)
                continue

            # heartbeat
            heartbeat(heartbeat_path)

            row_data = {c: row.get(c, "") for c in selected_columns}

            try:
                # ──────────────────────────────────────────────────────────
                # Branches
                # ──────────────────────────────────────────────────────────

                # GHS Pictogram Detector (expects image URLs in selected columns, typically 'image_link')
                if prompt_choice == "GHS Pictogram Detector":
                    image_urls = row_data.get("image_link", "")
                    image_list = [u.strip() for u in ([image_urls] if image_urls.strip() else [])]
                    pictograms_found = set()
                    debug_notes_all = []

                    for url in image_list:
                        encoded = fetch_image_as_base64(url)
                        if not encoded:
                            debug_notes_all.append(f"⚠️ Could not fetch image: {url}")
                            continue
                        out = ai(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": selected_prompt_text},
                                {"role": "user", "content": [
                                    {"type": "text", "text": "Check this image for GHS pictograms."},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}
                                ]}
                            ],
                            temperature=temperature_val, top_p=0, timeout=90
                        )
                        try:
                            parsed = json.loads((out or "").strip())
                            icons = [i.strip() for i in parsed.get("pictograms", "").split(",") if i.strip()]
                            pictograms_found.update(icons)
                            debug_notes_all.append(parsed.get("debug_notes", ""))
                        except Exception as e:
                            debug_notes_all.append(f"[parse-error: {e}]")

                    flat = {
                        "__row_index": idx,
                        "pictograms": ", ".join(sorted(pictograms_found)),
                        "debug_notes": " | ".join(debug_notes_all)
                    }
                    batch_rows.append(flat)

                elif prompt_choice == GREEN_CLAIMS_PROMPT:
                    row_text_raw = " ".join(str(row.get(c, "")) for c in selected_columns if str(row.get(c, "")).strip()).strip()
                    row_text = norm_basic(row_text_raw)

                    if not row_text:
                        flat = {"__row_index": idx, "green_claims_any": False, "green_claims_candidates": [], "green_claims_ai": {}, "green_claims_language": gc_language, "explanation": "No text in selected columns."}
                        batch_rows.append(flat)
                    else:
                        lang_col = LANG_TO_COL[gc_language]
                        candidates = screen_candidates(text=row_text, db=GC_DB, language_col=lang_col, threshold=gc_threshold, max_per_section=5)
                        if not candidates:
                            flat = {"__row_index": idx, "green_claims_any": False, "green_claims_candidates": [], "green_claims_ai": {}, "green_claims_language": gc_language}
                            batch_rows.append(flat)
                        else:
                            sys_txt, user_txt = build_green_claims_prompt(candidates=candidates, product_text=row_text, language_name=gc_language)
                            content = ai(
                                model=model_choice,
                                messages=[{"role": "system", "content": sys_txt}, {"role": "user", "content": user_txt}],
                                temperature=temperature_val, top_p=0, timeout=90
                            )
                            try:
                                parsed = json.loads(clean_gpt_json_block(content or ""))
                            except Exception as e:
                                parsed = {"error": f"JSON parse failed: {e}", "raw_output": content}

                            matched_strings = sorted({s for s in (c.get("evidence_snippet") for c in candidates) if s})
                            flat = {
                                "__row_index": idx,
                                "green_claims_any": str(parsed.get("overall", {}).get("any_green_claim_detected")).lower() == "true",
                                "green_claims_matched_strings": matched_strings,
                                "green_claims_candidates": candidates,
                                "green_claims_ai": parsed,
                                "green_claims_language": gc_language
                            }
                            batch_rows.append(flat)

                elif prompt_choice == "HFSS Checker":
                    # Pass 1 – nutrients
                    p1 = ai(model=model_choice, messages=[{"role": "system", "content": build_pass_1_prompt(row_data)}], temperature=temperature_val, top_p=0, timeout=90)
                    parsed_1 = json.loads(clean_gpt_json_block(p1 or "{}"))

                    # Pass 2 – NPM score
                    p2 = ai(model=model_choice, messages=[{"role": "system", "content": build_pass_2_prompt(parsed_1)}], temperature=temperature_val, top_p=0, timeout=90)
                    parsed_2 = json.loads(clean_gpt_json_block(p2 or "{}"))

                    # Pass 3 – HFSS status
                    p3 = ai(model=model_choice, messages=[{"role": "system", "content": build_pass_3_prompt({**parsed_2, "is_drink": parsed_1.get("is_drink", False)})}], temperature=temperature_val, top_p=0, timeout=90)
                    parsed_3 = json.loads(clean_gpt_json_block(p3 or "{}"))

                    # Pass 4 – validator
                    all_passes = {"parsed_nutrients": parsed_1, "npm_scoring": parsed_2, "hfss_classification": parsed_3}
                    p4 = ai(model=model_choice, messages=[{"role": "system", "content": build_pass_4_prompt(all_passes)}], temperature=temperature_val, top_p=0, timeout=90)
                    parsed_4 = json.loads(clean_gpt_json_block(p4 or "{}"))

                    full_result = {**parsed_1, **parsed_2, **parsed_3, **parsed_4}
                    flat = {"__row_index": idx, **{k: _flatten(v) for k, v in full_result.items()}}
                    batch_rows.append(flat)

                elif prompt_choice == "Banned/Restricted Checker":
                    ing_text = row_data.get(banned_ing_col, "")
                    if not ing_text.strip():
                        flat = {"__row_index": idx, "overall": {"banned_present": False, "restricted_present": False}, "items": [], "explanation": f"No ingredients in '{banned_ing_col}'", "candidates_debug": []}
                        batch_rows.append(flat)
                    else:
                        cands = find_banned_matches(ing_text, threshold=banned_fuzzy_threshold, return_details=True)
                        if not cands:
                            flat = {"__row_index": idx, "overall": {"banned_present": False, "restricted_present": False}, "items": [], "explanation": "No candidates via substring/fuzzy screen.", "candidates_debug": []}
                            batch_rows.append(flat)
                        else:
                            system_txt = build_banned_prompt(cands, ing_text)
                            content = ai(
                                model=model_choice,
                                messages=[{"role": "system", "content": system_txt}, {"role": "user", "content": ""}],
                                temperature=temperature_val, top_p=0, timeout=90
                            )
                            try:
                                parsed = json.loads(clean_gpt_json_block(content or ""))
                            except Exception as e:
                                parsed = {"error": f"JSON parse failed: {e}", "raw_output": content}
                            parsed["candidates_debug"] = cands
                            flat = {"__row_index": idx, **{k: _flatten(v) for k, v in parsed.items()}}
                            batch_rows.append(flat)

                elif prompt_choice == "Competitor SKU Match":
                    # NOTE: For brevity, not fully wired in this durable demo.
                    # If you want this branch active, load COMP_DB + use ai(...) to adjudicate exactly like your original flow.
                    flat = {"__row_index": idx, "error": "Competitor match not wired in durable demo"}
                    batch_rows.append(flat)

                elif prompt_choice == "Image: Multi-Image Ingredient Extract & Compare":
                    image_urls = row.get("image URLs", "")
                    image_list = [url.strip().replace('"', '') for url in image_urls.split(",") if url.strip()]
                    extracted = []
                    for url in image_list:
                        encoded_img = fetch_image_as_base64(url)
                        if not encoded_img:
                            continue
                        content = ai(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": selected_prompt_text},
                                {"role": "user", "content": [
                                    {"type": "text", "text": "Extract the INGREDIENTS section only."},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_img}"}}
                                ]}
                            ],
                            temperature=temperature_val, top_p=0, timeout=90
                        )
                        if content and "IMAGE_UNREADABLE" not in content.upper():
                            extracted.append(content.strip())
                    combined_html = "\n".join(extracted).strip()
                    reference = row.get("full_ingredients", "")
                    match_flag = "Pass" if combined_html in reference else "Needs Review"

                    diff_sys = (
                        "You are a detailed comparison assistant for UK food label ingredients. "
                        "You will be given two strings: OCR_OUTPUT (HTML-bolded ingredients from the image) "
                        "and REFERENCE (expected full_ingredients). Identify differences and return JSON "
                        '{"severity":"Minor|Major","diff_explanation":"..."}'
                    )
                    diff_user = f"OCR_OUTPUT:\n{combined_html}\n\nREFERENCE:\n{reference}"
                    diff_content = ai(model="gpt-4.1-mini", messages=[{"role":"system","content":diff_sys},{"role":"user","content":diff_user}], temperature=temperature_val, top_p=0, timeout=90)
                    try:
                        diff_json = json.loads(diff_content or "{}")
                    except Exception as e:
                        diff_json = {"severity": "Major", "diff_explanation": f"[Error comparing: {e}]"}

                    flat = {"__row_index": idx, "extracted_ingredients": combined_html, "comparison_result": match_flag, "severity": diff_json.get("severity",""), "diff_explanation": diff_json.get("diff_explanation","")}
                    batch_rows.append(flat)

                else:
                    # Generic JSON-returning prompt
                    if "USER MESSAGE:" in user_prompt:
                        system_txt, user_txt = user_prompt.split("USER MESSAGE:", 1)
                    else:
                        system_txt, user_txt = user_prompt, ""
                    system_txt = system_txt.replace("SYSTEM MESSAGE:", "").strip()
                    user_txt = (user_txt or "").strip().format(**row_data)
                    user_txt += f"\n\nSelected fields:\n{json.dumps(row_data, ensure_ascii=False)}"

                    content = ai(
                        model=model_choice,
                        messages=[{"role": "system", "content": system_txt}, {"role": "user", "content": user_txt}],
                        temperature=temperature_val, top_p=0, timeout=90
                    )
                    out = (content or "").strip()
                    if out.startswith("```"):
                        parts = out.split("```", maxsplit=2)
                        out = parts[1].lstrip("json").strip().split("```")[0].strip()
                    try:
                        parsed = json.loads(out)
                    except Exception:
                        parsed = {"raw_output": out}

                    flat = {"__row_index": idx, **{k: _flatten(v) for k, v in parsed.items()}}
                    batch_rows.append(flat)

            except Exception as e:
                batch_fail.append({"__row_index": idx, "error": str(e), "row_snapshot": json.dumps(row_data, ensure_ascii=False)})

            # Flush regularly or on abort
            should_flush = ((len(batch_rows) + len(batch_fail)) >= BATCH_SIZE) or (abort_toggle and abort_now.button("Stop after this row and save"))
            if should_flush:
                if batch_rows: append_csv(results_path, pd.DataFrame(batch_rows)); batch_rows.clear()
                if batch_fail: append_csv(failures_path, pd.DataFrame(batch_fail)); batch_fail.clear()
                if abort_toggle:
                    st.warning("Run stopped by user. Partial results saved.")
                    break

            # Progress + mini log
            progress = (idx + 1) / n_rows
            progress_bar.progress(progress)
            progress_text.markdown(f"<h4 style='text-align:center; color:#4A4443;'>Processed {idx + 1} of {n_rows} rows ({progress*100:.1f}%)</h4>", unsafe_allow_html=True)

            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=progress * 100,
                number={'font': {'color': '#4A4443'}},
                title={'text': "Progress", 'font': {'color': '#4A4443'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#4A4443', 'tickfont': {'color': '#4A4443'}, 'tickwidth': 2, 'ticklen': 8},
                    'bar': {'color': "#C2EA46"},
                    'bgcolor': "#E1FAD1",
                    'borderwidth': 1,
                    'steps': [{'range': [0, 50], 'color': "#E1FAD1"}, {'range': [50, 100], 'color': "#F2FAF4"}]
                },
                domain={'x': [0, 1], 'y': [0, 1]}
            ))
            gauge_placeholder.plotly_chart(fig, use_container_width=True)

        # Final flush
        if batch_rows: append_csv(results_path, pd.DataFrame(batch_rows))
        if batch_fail: append_csv(failures_path, pd.DataFrame(batch_fail))

        st.success("✅ Run finished (or aborted). Files saved in:")
        st.code(job_dir)

        # Convenience download buttons for the current session
        if os.path.exists(results_path):
            st.download_button("⬇️ Download results.csv", data=open(results_path, "rb").read(), file_name="results.csv", mime="text/csv")
        if os.path.exists(failures_path):
            st.download_button("⬇️ Download failures.csv", data=open(failures_path, "rb").read(), file_name="failures.csv", mime="text/csv")
