import streamlit as st
import pandas as pd
import json
import os
import requests
import base64
import plotly.graph_objects as go
from streamlit_cropper import st_cropper    # NEW
from PIL import Image                       # NEW
import io                                   # NEW
from openai import OpenAI

# Import prompts from your new module:
from prompts.prompts import PROMPT_OPTIONS

# Set page configuration immediately after imports!
st.set_page_config(page_title="Flexible AI Product Data Checker", layout="wide")

#############################
#  Custom CSS Styling Block! #
#############################
st.markdown(
    """
    <style>
    /* Global page styles */
    body {
      background-color: #f4f7f6;
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main {
      padding: 2rem;
    }
    h1 {
      font-size: 3.2rem;
      color: #4a90e2;
      text-align: center;
      margin-bottom: 1rem;
    }
    h2, h3, h4 {
      color: #333333;
    }
    /* Progress bar style */
    .stProgress > div > div > div {
        background-color: #4a90e2;
    }
    /* Custom sidebar style */
    .css-1d391kg .css-1d391kg { 
        background-color: #ffffff; 
        border-radius: 5px; 
        padding: 1rem; 
    }
    </style>
    """, unsafe_allow_html=True
)

#############################
#  Sidebar – Branding Info  #
#############################
st.sidebar.markdown("# Flexible AI Checker")
st.sidebar.image("https://cdn.freelogovectors.net/wp-content/uploads/2023/04/holland_and_barrett_logo-freelogovectors.net_.png", use_container_width=True)
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
    """
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
        "gpt-4o-mini": (0.00015, 0.0006),
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
    "gpt-4.1-mini": "Balanced cost and intelligence, great for language tasks.",
    "gpt-4.1-nano": "Ultra-cheap and fast, best for very lightweight checks.",
    "gpt-4o-mini":  "Higher quality than 4.1-mini, still affordable.",
    "gpt-4o": "The latest and fastest multimodal GPT-4 model. Supports image + text input.",
    "gpt-4-turbo":  "Very powerful and expensive — best for complex, high-value use cases."
}

# ---- Main Page Layout ----
st.markdown("<h1>📄 Flexible AI Product Data Checker With Cost Estimate</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:16px;'>Process your CSV row by row with OpenAI's GPT. Configure your columns, select (or write) a prompt, and choose a model.</p>", unsafe_allow_html=True)

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

# Track last selected prompt and reset crop if changed
if "last_prompt" not in st.session_state:
    st.session_state["last_prompt"] = prompt_choice
if "cropped_bytes" not in st.session_state:
    st.session_state["cropped_bytes"] = None
if st.session_state["last_prompt"] != prompt_choice:
    st.session_state["last_prompt"] = prompt_choice
    st.session_state["cropped_bytes"] = None

# Extract chosen prompt details
selected_prompt_data = PROMPT_OPTIONS[prompt_choice]
selected_prompt_text = selected_prompt_data["prompt"]
recommended_model = selected_prompt_data["recommended_model"]
prompt_description = selected_prompt_data["description"]
st.markdown(f"**Prompt Info:** {prompt_description}")

# --- Determine if image-based (single-image cropping prompts only) ---
single_image_prompts = {
    "Image: Ingredient Scrape (HTML)",
    "Image: Directions for Use",
    "Image: Storage Instructions",
    "Image: Warnings and Advisory (JSON)",
    # add any other single-image crop prompts here
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
    uploaded_image = st.file_uploader("Choose JPG or PNG", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        img = Image.open(uploaded_image).convert("RGB")
        st.markdown("### ✂️ Crop the label to the relevant section below:")

        with st.spinner("🖼️ Loading crop tool..."):
            cropped_img = st_cropper(
                img,
                box_color='#ff1744',
                realtime_update=True,
                aspect_ratio=None,
                return_type="image"
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

else:
    # For all other prompts (including multi-image URL), show CSV uploader
    uploaded_file = st.file_uploader("📁 Upload your CSV", type=["csv"])


# ---------------------------------------------------------------
# Image-prompt flow – two-pass high-accuracy extraction (single-image)
# ---------------------------------------------------------------
if is_image_prompt and st.session_state.get("cropped_bytes"):
    st.markdown("### 📤 Processing image…")
    with st.spinner("Running high-accuracy two-pass extraction"):
        # Enforce the correct model
        if model_choice != "gpt-4o":
            st.error("🛑  Image prompts require the **gpt-4o** model. Please choose it above and try again.")
            st.stop()

        try:
            # Use the general crop+prompt pipeline for non-ingredient prompts
            if "Ingredient Scrape" in prompt_choice:
                html_out = two_pass_extract(st.session_state["cropped_bytes"])
            else:
                data_url = f"data:image/jpeg;base64,{base64.b64encode(st.session_state['cropped_bytes']).decode()}"

                # System prompt from user config
                system_msg = user_prompt.replace("SYSTEM MESSAGE:", "").strip()

                # Call OpenAI with cropped image and prompt
                response = client.chat.completions.create(
                    model=model_choice,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": [
                            {"type": "text", "text": "Cropped label image below."},
                            {"type": "image_url", "image_url": {"url": data_url}}
                        ]}
                    ],
                    temperature=0.0,
                    top_p=0
                )
                html_out = response.choices[0].message.content.strip()

            # Show result
            if html_out == "IMAGE_UNREADABLE":
                st.error("🛑  The image was unreadable or missing the required section.")
            else:
                st.success("✅ GPT image processing complete!")

                # Auto-detect output format
                output_type = "html"
                if "Directions" in prompt_choice or "Storage" in prompt_choice:
                    output_type = "text"
                elif "Warnings and Advisory" in prompt_choice:
                    output_type = "json"

                st.code(html_out, language=output_type)

        except Exception as e:
            st.error(f"Image processing failed: {e}")

# ---------- Main Execution Logic ----------
if uploaded_file and user_prompt.strip():
    df = pd.read_csv(uploaded_file, dtype=str)
    st.markdown("### 📄 CSV Preview")
    st.dataframe(df.head())
    
    # 3. Dynamic Column Selector (up to 10 columns)
    st.subheader("📊 Select up to 10 CSV columns to pass to GPT")
    selected_columns = st.multiselect(
        "Use in Processing",
        options=df.columns.tolist(),
        default=df.columns.tolist()[:3],      # pre-select the first 3 by default
        help="Pick between 1 and 10 columns."
    )

    # Enforce user picks
    if not selected_columns:
        st.error("⚠️ Please select at least one column.")
        st.stop()
    if len(selected_columns) > 10:
        st.error("⚠️ You can select at most 10 columns. Please deselect some.")
        st.stop()

    cols_to_use = selected_columns

    # Display estimated cost
    cost_est = estimate_cost(model_choice, df, user_prompt, cols_to_use)
    st.markdown(
        f"<div style='padding:10px; background-color:#2a2a2a; color:#ffffff; border-radius:5px;'>"
        f"<strong>Estimated Cost:</strong> ${cost_est:0.4f} (rough estimate based on token usage)"
        "</div>",
        unsafe_allow_html=True
    )

    # Create gauge placeholder before starting the loop
    gauge_placeholder = st.empty()

    # Button to run GPT
    if st.button("🚀 Run GPT on CSV"):
        with st.spinner("Processing with GPT..."):
            progress_bar = st.progress(0)
            progress_text = st.empty()
            n_rows = len(df)
            results = []
            failed_rows = []
            rolling_log = []
            log_placeholder = st.empty()

            # ---------- Processing loop ----------
            for idx, row in df.iterrows():
                row_data = {c: row.get(c, "") for c in cols_to_use}
                content = ""

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
                        # 1 – split the stored prompt into true roles
                        if "USER MESSAGE:" in user_prompt:
                            system_txt, user_txt = user_prompt.split("USER MESSAGE:", 1)
                        else:
                            system_txt, user_txt = user_prompt, ""

                        system_txt = system_txt.replace("SYSTEM MESSAGE:", "").strip()
                        user_txt = user_txt.strip().format(**row_data)
                        user_txt += f"\n\nSelected fields:\n{json.dumps(row_data, ensure_ascii=False)}"

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
                        results.append(parsed)

                    except Exception as e:
                        failed_rows.append(idx)
                        error_result = {
                            "error": f"Failed to process row {idx}: {e}",
                            "raw_output": content if content else "No content returned"
                        }
                        results.append(error_result)

                # 6 – update progress UI
                progress = (idx + 1) / n_rows
                progress_bar.progress(progress)
                progress_text.markdown(
                    f"<h4 style='text-align:center;'>Processed {idx + 1} of {n_rows} rows ({progress*100:.1f}%)</h4>",
                    unsafe_allow_html=True
                )

                rolling_log.append(f"Row {idx + 1}: {json.dumps(results[-1])[:500]}")
                rolling_log = rolling_log[-20:]
                log_placeholder.markdown(
                    "<h4>📝 Recent Outputs (Last 20)</h4>"
                    "<pre style='background:#f0f0f0; padding:10px; border-radius:5px; max-height:400px; overflow:auto;'>"
                    + "\n".join(rolling_log) +
                    "</pre>",
                    unsafe_allow_html=True
                )

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=progress * 100,
                    title={'text': "Progress"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#4a90e2"},
                        'steps': [
                            {'range': [0, 50], 'color': "#e0e0e0"},
                            {'range': [50, 100], 'color': "#c8c8c8"}
                        ]
                    }
                ))
                gauge_placeholder.plotly_chart(fig, use_container_width=True)
            # ---------- end for loop ----------

            # Combine original CSV with GPT results
            results_df = pd.DataFrame(results)
            final_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)
            st.success("✅ GPT processing complete!")
            st.markdown("### 🔍 Final Result")
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
