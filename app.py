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
from rapidfuzz import fuzz

from prompts.prompts import PROMPT_OPTIONS
from prompts.competitor_match import (
    parse_sku,
    top_candidates,
    build_match_prompt,
    load_competitor_db,   # if you need it
)

from prompts.hfss import (
    build_pass_1_prompt,
    build_pass_2_prompt,
    build_pass_3_prompt,
    build_pass_4_prompt
)

# ─── Streamlit page config ─────────────────────
st.set_page_config(page_title="Flexible AI Product Data Checker", layout="wide")


#############################
#  Custom CSS Styling Block! #
#############################
st.markdown(
    """
    <style>
    /* ---------------------------------------------- */
    /* Core Brand Colours (for reference):
       • Brand Green   = #005A3F
       • Lime Green    = #C2EA46
       • Mint Green    = #E1FAD1
       • Powder White  = #F2FAF4
       • Grey          = #4A4443
    /* ---------------------------------------------- */

    /* Global page styles */
    body {
      background-color: #F2FAF4;   /* Powder White */
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main {
      padding: 2rem;
    }

    /* H1: Primary title in Brand Green */
    h1 {
      font-size: 3.2rem;
      color: #005A3F;   /* Brand Green */
      text-align: center;
      margin-bottom: 1rem;
    }

    /* H2, H3, H4: Dark Grey for subtitles/headings */
    h2, h3, h4 {
      color: #4A4443;   /* Grey */
    }

    /* Paragraph text, labels, etc. in very dark grey */
    p, label, .css-1cpxqw2 {
      color: #4A4443;   /* Grey */
    }

    /* Progress bar style: Lime Green fill */
    .stProgress > div > div > div {
        background-color: #C2EA46;  /* Lime Green */
    }

    /* Buttons (e.g. st.button, st.download_button) 
       use Lime Green background and white text */
    button[class*="stButton"] {
        background-color: #C2EA46 !important; /* Lime Green */
        color: white !important;
        border-radius: 4px;
        border: none;
        padding: 0.4rem 1rem;
        font-weight: 600;
    }
    button[class*="stButton"]:hover {
        background-color: #B3D841 !important; /* slightly darker lime on hover */
        color: white !important;
    }

    /* Text input, text area, and selectbox borders in Brand Green */
    .stTextInput>div>div>input, 
    .stTextArea>div>div>textarea,
    .stSelectbox>div>div>div>input {
      border: 1px solid #005A3F !important; /* Brand Green border */
      border-radius: 4px;
    }
    .stTextInput>div>div>input:focus, 
    .stTextArea>div>div>textarea:focus,
    .stSelectbox>div>div>div>input:focus {
      outline: 2px solid #C2EA46 !important;  /* Lime Green focus ring */
    }

    /* Sidebar – white background with mint‐green accents */
    .css-1d391kg .css-1d391kg {
        background-color: #FFFFFF; 
        border-radius: 5px; 
        padding: 1rem;
    }
    /* Sidebar headings in Brand Green */
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3 {
        color: #005A3F; 
    }
    /* Sidebar links or markdown text in dark grey */
    .css-1d391kg p, .css-1d391kg label {
        color: #4A4443;
    }

    /* Dataframe header background: Mint Green, header text in Grey */
    .stDataFrame thead th {
      background-color: #E1FAD1 !important; /* Mint Green */
      color: #4A4443 !important;            /* Grey */
    }
    /* Dataframe rows: alternate powder‐white backgrounds */
    .stDataFrame tbody tr:nth-child(even) {
      background-color: #FFFFFF !important; /* White (or #F2FAF4) for contrast */
    }
    .stDataFrame tbody tr:nth-child(odd) {
      background-color: #F2FAF4 !important; /* Powder White */
    }

    /* Code blocks (st.code) in a light mint background */
    .stCodeBlock pre {
      background-color: #E1FAD1 !important; /* Mint Green (very pale) */
      color: #4A4443 !important;            /* Grey text */
      border-radius: 4px;
      padding: 1rem;
    }

    /* Plotly charts: gridlines to Powder White and background in Mint Green */
    .js-plotly-plot .plotly .main-svg {
      background-color: transparent !important; 
    }
    .js-plotly-plot .gridlayer line {
      stroke: #F2FAF4 !important;           
      stroke-width: 1px;
    }

    /* Streamlit “success” messages in Lime Green background with white text */
    .stAlert.success {
      background-color: #C2EA46 !important;   /* Lime Green */
      color: #FFFFFF !important;
      border-radius: 4px;
    }

    /* Streamlit “error” messages in coral tone (#EB6C4D) for contrast */
    .stAlert.error {
      background-color: #EB6C4D !important;   /* Coral 1 */
      color: #FFFFFF !important;
      border-radius: 4px;
    }

    /* Streamlit info/warning messages in soft yellow (#FAEBC3) */
    .stAlert.info {
      background-color: #FAEBC3 !important;   /* Yellow 2 */
      color: #4A4443 !important;              /* Grey text */
      border-radius: 4px;
    }
    .stAlert.warning {
      background-color: #F4C300 !important;   /* Yellow 1 */
      color: #4A4443 !important;              /* Grey text */
      border-radius: 4px;
    }

    </style>
    """, unsafe_allow_html=True
)

#############################
#  Sidebar – Branding Info  #
#############################
st.sidebar.markdown("# Flexible AI Checker")
st.sidebar.image(
    "https://cdn.freelogovectors.net/wp-content/uploads/2023/04/holland_and_barrett_logo-freelogovectors.net_.png",
    use_container_width=True
)
st.sidebar.markdown(
    """
    ### 🧠 Flexible AI Product Data Assistant

    This powerful tool uses a range of OpenAI models to extract, check, and structure product data — from label image crops to batch CSV audits.

    - 🖼️ Crop product label images to extract INGREDIENTS, DIRECTIONS, WARNINGS, STORAGE and more
    - 📄 Upload CSVs to run row-by-row GPT checks across custom prompts
    - 🔎 Choose a pre-written audit or write your own
    - 💸 See real-time OpenAI cost estimates before running

    **Supported Models:**

    - **gpt-3.5-turbo** — Fast & low-cost for spelling, logic, and simple checks  
    - **gpt-4.1-nano** — Ultra-lightweight for basic, high-speed validation  
    - **gpt-4.1-mini** — Balanced model for most rule-based or JSON tasks  
    - **gpt-4o-mini** — Cheaper version of GPT-4o for fast multimodal jobs  
    - **gpt-4o** — Multimodal expert for accurate image+text extraction  
    - **gpt-4-turbo** — Premium model for the most complex audit logic

    *Choose the model that fits your need for cost, accuracy, or speed.*
    """, unsafe_allow_html=True
)

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
        "gpt-4o":        (0.005,  0.015),  # Correct cost as of May 2024
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

#############################
# Model Descriptions + UI   #
#############################
MODEL_OPTIONS = {
    "gpt-3.5-turbo": "Cheapest, good for basic tasks with acceptable quality.",
    "gpt-4.1-mini":  "Balanced cost and intelligence, great for language tasks.",
    "gpt-4.1-nano":  "Ultra-cheap and fast, best for very lightweight checks.",
    "gpt-4o-mini":   "Higher quality than 4.1-mini, still affordable.",
    "gpt-4o":        "The latest and fastest multimodal GPT-4 model. Supports image + text input.",
    "gpt-4-turbo":   "Very powerful and expensive — best for complex, high-value use cases."
}

# ---- Main Page Layout ----
st.markdown(
    "<h1>📄 Flexible AI Product Data Checker With Cost Estimate</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; font-size:16px; color:#4A4443;'>"
    "Process your CSV row by row with OpenAI's GPT. Configure your columns, select (or write) a prompt, and choose a model."
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
    # Initialize the new OpenAI client
    client = OpenAI(api_key=api_key_input)
    
    # ------------------------------------------------------------------
# Two-pass image ingredient extractor (add right after client = OpenAI(...))
# ------------------------------------------------------------------
def two_pass_extract(image_bytes: bytes) -> str:
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
    resp1 = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": pass1_sys},
            {"role": "user",   "content": [
                {"type": "text", "text": "Label image incoming."},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]}
        ],
        temperature=0.0, top_p=0
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
        provided by the user exactly as HTML and bold (**<b>…</b>**) every word
        that matches this UK-FIC allergen list:

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
        temperature=0.0, top_p=0
    )
    html_out = resp2.choices[0].message.content.strip()

    # ---- PASS 3 ----
    pass3_sys = (
        "You previously extracted an ingredient list from a UK food label image. "
        "Please double-check for spelling errors or OCR misreads in these items. "
        "If any corrections are needed, return the corrected string, preserving original HTML formatting. "
        "Otherwise return the input unchanged."
    )
    resp3 = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": pass3_sys},
            {"role": "user",   "content": html_out}
        ],
        temperature=0.0, top_p=0
    )
    return resp3.choices[0].message.content.strip()

def fetch_image_as_base64(url: str) -> str:
    """
    Fetch an image from a URL and return it as a base64-encoded string.
    Returns None if the image cannot be fetched.
    """
    try:
        if not url.startswith("https"):
            url = "https://" + url.strip().lstrip("/")
        response = requests.get(url)
        response.raise_for_status()
        return base64.b64encode(response.content).decode("utf-8")
    except Exception:
        return None

# 4. Choose a Pre-Written Prompt
st.subheader("💬 Choose a Prompt")
prompt_choice = st.selectbox(
    "Select a pre-written prompt or 'Custom':",
    list(PROMPT_OPTIONS.keys()),
    index=0
)

# 5. If they picked Competitor SKU Match, ask for a competitor CSV
# — Initialize COMP_DB only when needed —
COMP_DB = None
if prompt_choice == "Competitor SKU Match":
    comp_file = st.file_uploader(
        "🔍 Upload competitor CSV",
        type=["csv"],
        key="comp_csv"
    )
    if comp_file:
        comp_df = pd.read_csv(comp_file, dtype=str).fillna("")
        COMP_DB = [
            parse_sku(row["Retailer Product Name"], uid=row.get("UID", ""))
            for _, row in comp_df.iterrows()
        ]
    else:
        st.warning("Please upload a competitor CSV to enable SKU matching.")



# ─ Extract chosen prompt details ─
selected = PROMPT_OPTIONS[prompt_choice]
selected_prompt_text = selected["prompt"]
recommended_model    = selected["recommended_model"]
prompt_description   = selected["description"]
st.markdown(f"**Prompt Info:** {prompt_description}")

# ─ Session‐state to reset crops when the prompt changes ─
if "last_prompt" not in st.session_state:
    st.session_state["last_prompt"] = prompt_choice
if "cropped_bytes" not in st.session_state:
    st.session_state["cropped_bytes"] = None
if st.session_state["last_prompt"] != prompt_choice:
    st.session_state["last_prompt"] = prompt_choice
    st.session_state["cropped_bytes"] = None

# ─ Ensure fuzzy_threshold always exists ─
fuzzy_threshold = 87
if prompt_choice == "Novel Food Checker (EU)":
    fuzzy_threshold = st.slider(
        "Novel-food fuzzy threshold",
        min_value=70, max_value=100, value=87,
        help="Lower = catch more variants (but watch for false positives)."
    )

# --- Determine if image-based (single-image cropping prompts only) ---
single_image_prompts = {
    "Image: Ingredient Scrape (HTML)",
    "Image: Directions for Use",
    "Image: Storage Instructions",
    "Image: Warnings and Advisory (JSON)",
    # add any other single-image crop prompts here
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

# 5. Model Selector (default to recommended model, but user can override)
all_model_keys = list(MODEL_OPTIONS.keys())
default_index = all_model_keys.index(recommended_model) if recommended_model in all_model_keys else 0
model_choice = st.selectbox("🧠 Choose GPT model", all_model_keys, index=default_index)
st.markdown(f"**Model Info:** {MODEL_OPTIONS[model_choice]}")
st.markdown("---")

# 6. User Prompt Text Area
user_prompt = st.text_area(
    "✍️ Your prompt for GPT",
    value=selected_prompt_text,
    height=200
)

# -------------------------------
# Image uploader and crop logic
# -------------------------------
if is_image_prompt:
    st.markdown("### 🖼️ Upload Product Image & crop just the relevant panel")
    uploaded_image = st.file_uploader(
        "Choose JPG or PNG",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:
        img = Image.open(uploaded_image).convert("RGB")
        st.markdown("### ✂️ Crop the label to the relevant section below:")

        with st.spinner("🖼️ Loading crop tool..."):
            cropped_img = st_cropper(
                img,
                box_color="#C2EA46",  # Lime Green crop box
                realtime_update=True,
                aspect_ratio=None,
                return_type="image"
            )

            # Ensure competitor DB exists before running match
            if prompt_choice == "Competitor SKU Match" and COMP_DB is None:
                st.error("Cannot run SKU match—no competitor CSV uploaded.")
                st.stop()

        if st.button("✅ Use this crop →"):
            buf = io.BytesIO()
            cropped_img.save(buf, format="PNG")
            st.session_state["cropped_bytes"] = buf.getvalue()
            st.session_state["cropped_preview"] = cropped_img

            st.success("✅ Crop captured! Preview below:")
            st.image(
                cropped_img,
                use_container_width=True,
                caption="Cropped Area Sent to GPT"
            )

            st.download_button(
                label="⬇️ Download Cropped Image Sent to GPT",
                data=st.session_state["cropped_bytes"],
                file_name="cropped_label.png",
                mime="image/png"
            )

# ────────────────────────────────────────────────
# Non-image prompts always get the product-CSV uploader
# ────────────────────────────────────────────────
if not is_image_prompt:
    uploaded_file = st.file_uploader(
        "📁 Upload your product CSV",
        type=["csv"],
        key="data_csv"
    )

# ---------------------------------------------------------------
# Image-prompt flow – two-pass high-accuracy extraction (single-image)
# ---------------------------------------------------------------
if is_image_prompt and st.session_state.get("cropped_bytes"):
    st.markdown("### 📤 Processing image…")
    with st.spinner("Running high-accuracy two-pass extraction"):
        # Enforce the correct model
        if model_choice != "gpt-4o":
            st.error(
                "🛑  Image prompts require the **gpt-4o** model. "
                "Please choose it above and try again."
            )
            st.stop()

        try:
            if "Ingredient Scrape" in prompt_choice:
                html_out = two_pass_extract(st.session_state["cropped_bytes"])
            else:
                data_url = (
                    "data:image/jpeg;base64,"
                    + base64.b64encode(
                        st.session_state["cropped_bytes"]
                    ).decode()
                )
                system_msg = user_prompt.replace("SYSTEM MESSAGE:", "").strip()
                response = client.chat.completions.create(
                    model=model_choice,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Cropped label image below."},
                                {"type": "image_url", "image_url": {"url": data_url}}
                            ]
                        }
                    ],
                    temperature=0.0,
                    top_p=0
                )
                html_out = response.choices[0].message.content.strip()

            if html_out == "IMAGE_UNREADABLE":
                st.error(
                    "🛑  The image was unreadable or missing the required section."
                )
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

# ---------- Main Execution Logic ----------
if uploaded_file and (
    user_prompt.strip() or
    prompt_choice in {
        "Novel Food Checker (EU)",
        "Competitor SKU Match",
        "GHS Pictogram Detector"
    }
):
    df = pd.read_csv(uploaded_file, dtype=str)
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

    # Enforce user picks
    if not selected_columns:
        st.error("⚠️ Please select at least one column.")
        st.stop()
    if len(selected_columns) > 10:
        st.error("⚠️ You can select at most 10 columns. Please deselect some.")
        st.stop()

    cols_to_use = selected_columns          # ← existing line
    # ──────────────────────────────────────────────────────────────
    #  NEW: ask **only** for Competitor-match prompt
    # ──────────────────────────────────────────────────────────────
    if prompt_choice == "Competitor SKU Match":
        sku_col = st.selectbox(
            "Which column contains *your* product name / volume?",
            options=cols_to_use,            # user may pick only among those already passed to GPT
            help="e.g. 'Product_Name' or 'SKU Title'"
        )

    # Display estimated cost (dark background card with white text)
    st.markdown(
        f"""
        <div style='
            padding:10px; 
            background-color:#FFFFFF;  /* Grey */
            color:#4A4443; 
            border-radius:5px;
            margin-bottom:1rem;
        '>
            <strong>Estimated Cost:</strong> ${estimate_cost(model_choice, df, user_prompt, cols_to_use):0.4f} (rough estimate based on token usage)
        </div>
        """,
        unsafe_allow_html=True
    )

    # Create gauge placeholder before starting the loop
    gauge_placeholder = st.empty()

    # Button to run GPT
    if st.button("🚀 Run GPT on CSV"):
        # Immediately stop if they chose SKU match but never loaded a competitor DB
        if prompt_choice == "Competitor SKU Match" and not COMP_DB:
            st.error("Cannot run SKU match—no competitor CSV uploaded.")
            st.stop()
    
        with st.spinner("Processing with GPT..."):
            progress_bar = st.progress(0)
            progress_text = st.empty()
            n_rows = len(df)
            results = []
            failed_rows = []
            rolling_log = []
            log_placeholder = st.empty()
            status_placeholder = st.empty()
            
            # ---------- Processing loop ----------
            for idx, row in df.iterrows():
                row_data = {c: row.get(c, "") for c in cols_to_use}
                content = ""

                # -------------------------------------------------------------
                # 🔍 GHS Pictogram Detector logic (for rows with image URLs)
                # -------------------------------------------------------------
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
                                temperature=0.0,
                                top_p=0
                            )
                            result = json.loads(gpt_response.choices[0].message.content.strip())
                            icons = [i.strip() for i in result.get("pictograms", "").split(",") if i.strip()]
                            pictograms_found.update(icons)
                            debug_notes_all.append(result.get("debug_notes", ""))
                
                        except Exception as e:
                            failed_rows.append(idx)
                            results.append({
                                "error": f"Error in GPT call for image: {url}",
                                "debug_notes": str(e)
                            })
                            break  # Skip to next row
                
                    results.append({
                        "pictograms": ", ".join(sorted(pictograms_found)),
                        "debug_notes": " | ".join(debug_notes_all)
                    })
                    continue  # ⛔ Skip rest of the loop for this row

                if prompt_choice == "HFSS Checker":
                    try:
                        # Pass 1 – extract structured nutrients
                        p1 = client.chat.completions.create(
                            model=model_choice,
                            messages=[{"role": "system", "content": build_pass_1_prompt(row_data)}]
                        ).choices[0].message.content.strip()
                        parsed_1 = json.loads(p1)

                        # Pass 2 – compute NPM score
                        p2 = client.chat.completions.create(
                            model=model_choice,
                            messages=[{"role": "system", "content": build_pass_2_prompt(parsed_1)}]
                        ).choices[0].message.content.strip()
                        parsed_2 = json.loads(p2)

                        # Pass 3 – determine HFSS status
                        p3 = client.chat.completions.create(
                            model=model_choice,
                            messages=[{"role": "system", "content": build_pass_3_prompt({
                                **parsed_2, "is_drink": parsed_1.get("is_drink", False)
                            })}]
                        ).choices[0].message.content.strip()
                        parsed_3 = json.loads(p3)

                        # Pass 4 – final validator
                        all_passes = {
                            "parsed_nutrients": parsed_1,
                            "npm_scoring": parsed_2,
                            "hfss_classification": parsed_3
                        }
                        p4 = client.chat.completions.create(
                            model=model_choice,
                            messages=[{"role": "system", "content": build_pass_4_prompt(all_passes)}]
                        ).choices[0].message.content.strip()
                        parsed_4 = json.loads(p4)

                        # Combine all into one dict for export
                        full_result = {
                            **parsed_1,
                            **parsed_2,
                            **parsed_3,
                            **parsed_4
                        }
                        results.append(full_result)

                        # ─── Live log and progress ───────────────────────────────
                        if "rolling_log_dicts" not in st.session_state:
                            st.session_state.rolling_log_dicts = []
                        st.session_state.rolling_log_dicts.append(full_result)
                        st.session_state.rolling_log_dicts = st.session_state.rolling_log_dicts[-20:]

                        log_placeholder.empty()
                        log_placeholder.markdown(
                            "<h4 style='color:#4A4443;'>📝 Recent Outputs (Last 20)</h4>",
                            unsafe_allow_html=True
                        )

                        # Show last 3 entries fully
                        for entry in st.session_state.rolling_log_dicts[-3:]:
                            log_placeholder.json(entry)

                        # Older entries in expanders
                        for i, entry in enumerate(st.session_state.rolling_log_dicts[:-3]):
                            row_num = (idx + 1) - (len(st.session_state.rolling_log_dicts) - i)
                            with log_placeholder.expander(f"Row {row_num} output", expanded=False):
                                st.json(entry)

                        # Optional: expanded view of each GPT pass
                        with log_placeholder.expander(f"Row {idx+1} | Pass-by-pass breakdown", expanded=False):
                            st.json({
                                "Pass 1 – Nutrients": parsed_1,
                                "Pass 2 – NPM Score": parsed_2,
                                "Pass 3 – HFSS Status": parsed_3,
                                "Pass 4 – Validator": parsed_4
                            })

                        # ─── Progress UI update ─────────────────────────────────
                        progress = (idx + 1) / n_rows
                        progress_bar.progress(progress)
                        progress_text.markdown(
                            f"<h4 style='text-align:center; color:#4A4443;'>Processed {idx + 1} of {n_rows} rows ({progress*100:.1f}%)</h4>",
                            unsafe_allow_html=True
                        )

                    except Exception as e:
                        failed_rows.append(idx)
                        results.append({
                            "error": f"Row {idx}: {e}",
                            "raw_output": "Check individual passes for debug info"
                        })
                    continue

                
                # Handle multi-image ingredient extract
                if prompt_choice == "Image: Multi-Image Ingredient Extract & Compare":
                    image_urls = row.get("image URLs", "")
                    image_list = [url.strip().replace('"', '') for url in image_urls.split(",") if url.strip()]
                    extracted = []

                    # 1) Run OCR on each URL
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
                                temperature=0.0
                            )
                            result = response.choices[0].message.content.strip()
                            if result and "IMAGE_UNREADABLE" not in result.upper():
                                extracted.append(result)
                        except Exception as e:
                            extracted.append(f"[ERROR processing image: {e}]")

                    combined_html = "\n".join(extracted).strip()
                    reference = row.get("full_ingredients", "")

                    # 2) Simple containment check to give a quick Pass/Needs Review
                    match_flag = "Pass" if combined_html in reference else "Needs Review"

                    # 3) Ask GPT to compare combined_html vs. reference in detail
                    diff_prompt = [
                        {
                            "role": "system",
                            "content": (
                                "You are a detailed comparison assistant for UK food label ingredients. "
                                "You will be given two strings:\n"
                                "  • OCR_OUTPUT (HTML-bolded ingredients from the image)\n"
                                "  • REFERENCE (the expected 'full_ingredients' text from our CSV).\n\n"
                                "Your task:\n"
                                "1. Identify any differences in wording, order, punctuation, or missing/extra ingredients.\n"
                                "2. Pay special attention to allergen tokens (bolded HTML tags in OCR_OUTPUT) and flag if any allergen is missing or incorrect.\n"
                                "3. For each difference, decide if it is “Minor” (typos, small punctuation/ordering) or “Major” (missing allergen, wrong ingredient, or substantial content changes).\n\n"
                                "Return exactly one JSON object (no extra commentary) with these fields:\n"
                                "{\n"
                                "  \"severity\": \"Minor\" | \"Major\",   # overall severity of all differences\n"
                                "  \"diff_explanation\": \"<a few sentences explaining what changed and why you chose that severity>\"\n"
                                "}"
                            )
                        },
                        {
                            "role": "user",
                            "content": (
                                "OCR_OUTPUT:\n"
                                + combined_html
                                + "\n\nREFERENCE:\n"
                                + reference
                            )
                        }
                    ]

                    try:
                        diff_resp = client.chat.completions.create(
                            model="gpt-4.1-mini",
                            messages=diff_prompt,
                            temperature=0.0
                        )
                        diff_content = diff_resp.choices[0].message.content.strip()
                        # Parse GPT output (JSON)
                        diff_json = json.loads(diff_content)
                    except Exception as e:
                        diff_json = {
                            "severity": "Major",
                            "diff_explanation": f"[Error comparing: {e}]"
                        }

                    results.append({
                        "extracted_ingredients": combined_html,
                        "comparison_result": match_flag,
                        "severity": diff_json.get("severity", ""),
                        "diff_explanation": diff_json.get("diff_explanation", "")
                    })

                else:
                    try:
                        # ------------------------------------------------------------------
                        # 🚀  Competitor SKU Match
                        # ------------------------------------------------------------------
                        if prompt_choice == "Competitor SKU Match":
                            try:
                                my_sku      = parse_sku(row[sku_col])
                                cands_raw   = top_candidates(my_sku, db=COMP_DB, k=8)   # [(ParsedSKU, score)]
                                cand_list   = [c for c, _ in cands_raw]                 # strip scores
                                status_placeholder.info(f"Row {idx+1}/{n_rows}: running fuzzy match…")
                                cands_raw   = top_candidates(my_sku, db=COMP_DB, k=8)   # [(ParsedSKU, score)]
                                status_placeholder.success(f"Row {idx+1}/{n_rows}: found {len(cands_raw)} candidate(s)")
                        
                                # guard clause: nothing plausible
                                if not cand_list:
                                    results.append({
                                        "match_found": "No",
                                        "best_match_uid": "",
                                        "best_match_name": "",
                                        "confidence_pct": 0,
                                        "reason": "No candidate met minimum fuzzy+size rules"
                                    })
                                    continue
                        
                                system_prompt = build_match_prompt(my_sku, cand_list)
                                status_placeholder.info(f"Row {idx+1}/{n_rows}: calling GPT to pick best match…")
                                resp = client.chat.completions.create(
                                    model=model_choice,
                                    messages=[{"role": "system", "content": system_prompt}],
                                    temperature=0.0,
                                    top_p=0
                                )
                                gpt_json = json.loads(resp.choices[0].message.content)
                                status_placeholder.success(f"Row {idx+1}/{n_rows}: GPT done (match_found={gpt_json.get('match_found')})")
                        
                                resp = client.chat.completions.create(
                                    model=model_choice,
                                    messages=[{"role": "system", "content": system_prompt}],
                                    temperature=0.0,
                                    top_p=0
                                )
                                gpt_json = json.loads(resp.choices[0].message.content)
                        
                                # enrich with UID so you can JOIN later or show hyperlink
                                best = next((c for c in cand_list
                                             if c.uid == gpt_json.get("best_match_uid")), None)
                        
                                results.append({
                                    **gpt_json,
                                    "best_match_uid": getattr(best, "uid", ""),
                                    "best_match_name": getattr(best, "raw_name", ""),
                                    "candidate_debug": [(c.raw_name, s) for c, s in cands_raw]
                                })
                        
                            except Exception as e:
                                failed_rows.append(idx)
                                results.append({"error": f"Row {idx}: {e}"})
                            finally:
                                continue  # skip the rest of the loop body
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
                                continue

                            # run substring + fuzzy with dynamic threshold, capture scores
                            matches_with_scores = find_novel_matches(
                                ing_text,
                                threshold=fuzzy_threshold,
                                return_scores=True
                            )
                            # unpack into candidates + debug list
                            candidate_matches = [term for term, _ in matches_with_scores]
                            debug_scores      = matches_with_scores

                            if candidate_matches:
                                system_txt = build_novel_food_prompt(candidate_matches, ing_text)
                                user_txt   = ""
                            else:
                                results.append({
                                    "novel_food_flag": "No",
                                    "confirmed_matches": [],
                                    "explanation": "No potential matches found via fuzzy/substring match.",
                                    "fuzzy_debug_matches": debug_scores
                                })
                                continue

                        else:
                            # your existing USER MESSAGE split logic
                            if "USER MESSAGE:" in user_prompt:
                                system_txt, user_txt = user_prompt.split("USER MESSAGE:", 1)
                            else:
                                system_txt, user_txt = user_prompt, ""

                            system_txt = system_txt.replace("SYSTEM MESSAGE:", "").strip()
                            user_txt = user_txt.strip().format(**row_data)
                            user_txt += f"\n\nSelected fields:\n{json.dumps(row_data, ensure_ascii=False)}"

                        # GPT call remains unchanged
                        response = client.chat.completions.create(
                            model=model_choice,
                            messages=[
                                {"role": "system", "content": system_txt},
                                {"role": "user",   "content": user_txt}
                            ],
                            temperature=0.0,
                            top_p=0
                        )
                        content = response.choices[0].message.content.strip()

                        if content.startswith("```"):
                            parts = content.split("```", maxsplit=2)
                            content = parts[1].lstrip("json").strip().split("```")[0].strip()

                        parsed = json.loads(content)
                        # attach debug scores to your parsed result if you like
                        if prompt_choice == "Novel Food Checker (EU)":
                            parsed["fuzzy_debug_matches"] = debug_scores

                        results.append(parsed)

                    except Exception as e:
                        failed_rows.append(idx)
                        error_result = {
                            "error": f"Failed to process row {idx}: {e}",
                            "raw_output": content if content else "No content returned"
                        }
                        results.append(error_result)



                # 6 – update progress UI
                progress = (idx + 1) / n_rows
                progress_bar.progress(progress)
                progress_text.markdown(
                    f"<h4 style='text-align:center; color:#4A4443;'>Processed {idx + 1} of {n_rows} rows ({progress*100:.1f}%)</h4>",
                    unsafe_allow_html=True
                )

                # collect up to the last 20 raw result dicts
                if "rolling_log_dicts" not in st.session_state:
                    st.session_state.rolling_log_dicts = []
                st.session_state.rolling_log_dicts.append(results[-1])
                st.session_state.rolling_log_dicts = st.session_state.rolling_log_dicts[-20:]

                # clear out the previous widget
                log_placeholder.empty()
                # render a header
                log_placeholder.markdown(
                    "<h4 style='color:#4A4443;'>📝 Recent Outputs (Last 20)</h4>",
                    unsafe_allow_html=True
                )

                # first, always show the last few outright
                num_always_show = 3
                always_show = st.session_state.rolling_log_dicts[-num_always_show:]
                for entry in always_show:
                    log_placeholder.json(entry)

                # then render each in an expander, expanded by default
                for i, entry in enumerate(st.session_state.rolling_log_dicts):
                    # compute the original row number
                    row_num = (idx + 1) - (len(st.session_state.rolling_log_dicts) - i)
                    with log_placeholder.expander(f"Row {row_num} output", expanded=True):
                        st.json(entry)

                # … inside your row‐processing loop, after updating progress_text and rolling_log …

                # Build a Plotly gauge that uses our brand colours:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=progress * 100,
                    number={
                        'font': {'color': '#4A4443'}
                    },
                    title={
                        'text': "Progress",
                        'font': {'color': '#4A4443'}
                    },
                    gauge={
                        'axis': {
                            'range': [0, 100],
                            'tickcolor': '#4A4443',
                            'tickfont': {'color': '#4A4443'},
                            'tickwidth': 2,
                            'ticklen': 8
                        },
                        'bar': {
                            'color': "#C2EA46"
                        },
                        'bgcolor': "#E1FAD1",
                        'borderwidth': 1,
                        'steps': [
                            {'range': [0, 50], 'color': "#E1FAD1"},
                            {'range': [50, 100], 'color': "#F2FAF4"}
                        ]
                    },
                    domain={'x': [0, 1], 'y': [0, 1]}
                ))
                gauge_placeholder.plotly_chart(fig, use_container_width=True)


            # ---------- end for loop ----------

            # Combine original CSV with GPT results
            results_df = pd.DataFrame(results)
            final_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)
            st.success("✅ GPT processing complete!")

            st.markdown(
                "<h3 style='color:#005A3F;'>🔍 Final Result</h3>",
                unsafe_allow_html=True
            )
            # Stringify any complex columns so Streamlit can render them
            if "fuzzy_debug_matches" in final_df.columns:
                final_df["fuzzy_debug_matches"] = final_df["fuzzy_debug_matches"].apply(lambda x: json.dumps(x, ensure_ascii=False))
            if "candidate_debug" in final_df.columns:
                final_df["candidate_debug"] = final_df["candidate_debug"].apply(
                    lambda x: json.dumps(x, ensure_ascii=False)
                )
            st.dataframe(final_df)

            # Download buttons
            st.download_button(
                "⬇️ Download Full Results CSV",
                final_df.to_csv(index=False).encode("utf-8"),
                "gpt_output.csv",
                "text/csv"
            )

            if failed_rows:
                failed_df = df.iloc[failed_rows].copy()
                st.warning(f"{len(failed_rows)} rows failed to process. You can download them and retry.")
                st.download_button(
                    "⬇️ Download Failed Rows CSV",
                    failed_df.to_csv(index=False).encode("utf-8"),
                    "gpt_failed_rows.csv",
                    "text/csv"
                )
