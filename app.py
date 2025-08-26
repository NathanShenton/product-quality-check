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
from sidebar import render_sidebar
from style import inject_css
from prompts.bannedingredients import find_banned_matches, build_banned_prompt

from prompts.prompts import PROMPT_OPTIONS
from prompts.competitor_match import (
    parse_sku,
    top_candidates,
    build_match_prompt,
    load_competitor_db,
)

from prompts.hfss import (
    build_pass_1_prompt,
    build_pass_2_prompt,
    build_pass_3_prompt,
    build_pass_4_prompt
)

# ─── Streamlit page config ─────────────────────
st.set_page_config(page_title="Flexible AI Product Data Checker", layout="wide")

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
    model_costs_per_1k = {
        "gpt-3.5-turbo": (0.0005, 0.002),
        "gpt-4.1-mini":  (0.0004, 0.0016),
        "gpt-4.1-nano":  (0.0001, 0.0004),
        "gpt-4o-mini":   (0.00015, 0.0006),
        "gpt-4o":        (0.005,  0.015),  # Correct cost as of May 2024
        "gpt-4-turbo":   (0.01,   0.03),
        "gpt-5":         (0.00125, 0.01),
        "gpt-5-mini":    (0.00025, 0.002),
        "gpt-5-nano":    (0.00005, 0.0004)
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
        total_output_tokens += 100  # assume ~100 tokens output

    input_ktokens = total_input_tokens / 1000
    output_ktokens = total_output_tokens / 1000
    return (input_ktokens * cost_in) + (output_ktokens * cost_out)

#############################
#   JSON Cleaner            #
#############################
def clean_gpt_json_block(text: str) -> str:
    import re
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"```$", "", text.strip(), flags=re.IGNORECASE)
    json_start = text.find("{")
    if json_start != -1:
        text = text[json_start:]
    return text.strip()

def _flatten(x):
    if isinstance(x, (list, dict, tuple)):
        try:
            return json.dumps(x, ensure_ascii=False)
        except Exception:
            return str(x)
    return x

#############################
#   Build Chat Args Helper  #
#############################
def build_chat_args(model_choice, system_txt, user_txt, temperature_val=1.0):
    messages = [
        {"role": "system", "content": system_txt},
        {"role": "user",   "content": user_txt}
    ]
    args = {"model": model_choice, "messages": messages}

    if model_choice.startswith("gpt-5"):
        if temperature_val < 0.5:
            messages[0]["content"] += "\n\nIMPORTANT: Output deterministically, avoid creativity."
        elif temperature_val > 0.5:
            messages[0]["content"] += "\n\nIMPORTANT: Allow creativity and variation in responses."
    else:
        args["temperature"] = temperature_val
        args["top_p"] = 0
    return args

#############################
#   Model Descriptions      #
#############################
MODEL_OPTIONS = {
    "gpt-3.5-turbo": "Cheapest option, good for lightweight tasks and simple text checks.",
    "gpt-4.1-mini":  "Balanced cost and capability — reliable for most language and data-processing tasks.",
    "gpt-4.1-nano":  "Ultra-cheap and very fast — best for trivial checks or bulk lightweight processing.",
    "gpt-4o-mini":   "Affordable, higher quality than 4.1-mini, suitable for structured audits at scale.",
    "gpt-4o":        "Multimodal GPT-4 (text + images). Fast and accurate for complex text/image tasks.",
    "gpt-4-turbo":   "Advanced GPT-4 variant — powerful and more expensive. Best for high-value, complex use cases.",
    "gpt-5":         "Most advanced, highest reasoning power. Use for nuanced compliance or mission-critical tasks.",
    "gpt-5-mini":    "Great balance of cost and intelligence — strong reasoning at competitive pricing. Ideal for structured audits like allergen checks.",
    "gpt-5-nano":    "Cheapest GPT-5 tier — ultra-fast and low cost. Best for simple classification or first-pass filtering."
}

# ---- Main Page Layout ----
st.markdown("<h1>📄 Flexible AI Product Data Checker With Cost Estimate</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; font-size:16px; color:#4A4443;'>"
    "Process your CSV row by row with OpenAI's GPT. Configure your columns, select (or write) a prompt, and choose a model."
    "</p>",
    unsafe_allow_html=True
)

# --- API Key Entry ---
col1, col2 = st.columns(2)
with col1:
    api_key_input = st.text_input("🔑 Enter your OpenAI API Key", type="password")
    if not api_key_input:
        st.warning("Please enter your OpenAI API key to proceed.")
        st.stop()
    client = OpenAI(api_key=api_key_input)

# -------------------------------
# Prompt selection
# -------------------------------
st.subheader("💬 Choose a Prompt")
prompt_choice = st.selectbox(
    "Select a pre-written prompt or 'Custom':",
    list(PROMPT_OPTIONS.keys()),
    index=0
)

selected = PROMPT_OPTIONS[prompt_choice]
selected_prompt_text = selected["prompt"]
recommended_model    = selected["recommended_model"]
prompt_description   = selected["description"]

st.markdown(f"**Prompt Info:** {prompt_description}")

# --- Session state resets ---
if "last_prompt" not in st.session_state:
    st.session_state["last_prompt"] = prompt_choice
if "cropped_bytes" not in st.session_state:
    st.session_state["cropped_bytes"] = None
if st.session_state["last_prompt"] != prompt_choice:
    st.session_state["last_prompt"] = prompt_choice
    st.session_state["cropped_bytes"] = None

# -------------------------------
# Threshold sliders (when needed)
# -------------------------------
fuzzy_threshold = 87
if prompt_choice == "Novel Food Checker (EU)":
    fuzzy_threshold = st.slider(
        "Novel-food fuzzy threshold",
        min_value=70, max_value=100, value=87,
        help="Lower = catch more variants (but watch for false positives)."
    )

if prompt_choice == "Banned/Restricted Checker":
    banned_fuzzy_threshold = st.slider(
        "Banned/Restricted fuzzy threshold",
        min_value=80, max_value=100, value=90,
        help="Lower = catch more variants (but watch false positives)."
    )

# -------------------------------
# Model selector + creativity slider
# -------------------------------
all_model_keys  = list(MODEL_OPTIONS.keys())
default_index   = all_model_keys.index(recommended_model) if recommended_model in all_model_keys else 0

model_choice = st.selectbox(
    "🧠 Choose GPT model",
    all_model_keys,
    index=default_index
)

st.markdown(f"**Model Info:** {MODEL_OPTIONS[model_choice]}")

temperature_val = st.slider(
    "🎛️ Model temperature (0 = deterministic, 1 = very creative)",
    min_value=0.0, max_value=1.0, value=0.0, step=0.05
)

st.markdown("---")

# -------------------------------
# User Prompt Text Area
# -------------------------------
user_prompt = st.text_area(
    "✍️ Your prompt for GPT",
    value=selected_prompt_text,
    height=200
)

# -------------------------------
# Image handling (for OCR + crops)
# -------------------------------
def two_pass_extract(image_bytes: bytes) -> str:
    """
    Run GPT-4o in three passes:
      • Pass-1: OCR of ingredients panel
      • Pass-2: Format and bold allergens
      • Pass-3: Correct OCR misreads
    """
    import textwrap
    data_url = f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode()}"

    # Pass 1 – OCR
    resp1 = client.chat.completions.create(
        **build_chat_args(
            "gpt-4o",
            "You are a specialist OCR engine. Extract the EXACT text of the INGREDIENTS panel on a UK food label image. Preserve punctuation, %, brackets. If unreadable, output IMAGE_UNREADABLE.",
            json.dumps([{"type": "image_url", "image_url": {"url": data_url}}]),
            temperature_val
        )
    )
    raw = resp1.choices[0].message.content.strip()
    if "IMAGE_UNREADABLE" in raw.upper():
        return "IMAGE_UNREADABLE"

    # Pass 2 – Bold allergens
    allergens = (
        "celery,wheat,rye,barley,oats,spelt,kamut,crustaceans,eggs,fish,lupin,"
        "milk,molluscs,mustard,almond,hazelnut,walnut,cashew,pecan,pistachio,"
        "macadamia,brazil nut,peanut,sesame,soy,soya,sulphur dioxide,sulphites"
    )
    pass2_sys = f"You are a food-label compliance agent. Format the INGREDIENT string provided by the user exactly as HTML and bold (<b>…</b>) every word that matches this allergen list: {allergens}. Return HTML only."
    resp2 = client.chat.completions.create(
        **build_chat_args("gpt-4o", pass2_sys, raw, temperature_val)
    )
    html_out = resp2.choices[0].message.content.strip()

    # Pass 3 – Correct OCR
    resp3 = client.chat.completions.create(
        **build_chat_args(
            "gpt-4o",
            "You previously extracted an ingredient list. Double-check spelling/OCR errors. If corrections are needed, return corrected string, preserving HTML formatting. Otherwise return unchanged.",
            html_out,
            temperature_val
        )
    )
    return resp3.choices[0].message.content.strip()

# Image or CSV prompt types
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

if is_image_prompt:
    recommended_model = "gpt-4o"
    st.markdown("### 🖼️ Upload Product Image & crop just the relevant panel")
    uploaded_image = st.file_uploader("Choose JPG or PNG", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        img = Image.open(uploaded_image).convert("RGB")
        st.markdown("### ✂️ Crop the label to the relevant section below:")
        cropped_img = st_cropper(img, box_color="#C2EA46", realtime_update=True, aspect_ratio=None, return_type="image")
        if st.button("✅ Use this crop →"):
            buf = io.BytesIO()
            cropped_img.save(buf, format="PNG")
            st.session_state["cropped_bytes"] = buf.getvalue()
            st.image(cropped_img, use_container_width=True, caption="Cropped Area Sent to GPT")
            st.success("✅ Crop captured!")

else:
    uploaded_file = st.file_uploader("📁 Upload your product CSV", type=["csv"], key="data_csv")

# ---------- Main Execution Logic ----------
if uploaded_file and (
    user_prompt.strip() or
    prompt_choice in {
        "Novel Food Checker (EU)",
        "Competitor SKU Match",
        "GHS Pictogram Detector",
        "Banned/Restricted Checker"
    }
):
    df = pd.read_csv(uploaded_file, dtype=str)
    st.markdown("### 📄 CSV Preview")
    st.dataframe(df.head())

    # --- Column Selector ---
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
        st.error("⚠️ You can select at most 10 columns.")
        st.stop()

    cols_to_use = selected_columns

    # Extra selectors for some prompts
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

    # --- Estimated cost card ---
    st.markdown(
        f"""
        <div style='
            padding:10px; 
            background-color:#FFFFFF;
            color:#4A4443; 
            border-radius:5px;
            margin-bottom:1rem;
        '>
            <strong>Estimated Cost:</strong> ${estimate_cost(model_choice, df, user_prompt, cols_to_use):0.4f}
            (rough estimate based on token usage)
        </div>
        """,
        unsafe_allow_html=True
    )

    # Gauge placeholder
    gauge_placeholder = st.empty()

    # --- Run Button ---
    if st.button("🚀 Run GPT on CSV"):
        if prompt_choice == "Competitor SKU Match" and 'COMP_DB' in locals() and COMP_DB is None:
            st.error("Cannot run SKU match — no competitor CSV uploaded.")
            st.stop()

        with st.spinner("Processing with GPT..."):
            progress_bar = st.progress(0)
            progress_text = st.empty()
            n_rows = len(df)
            results = []
            failed_rows = []
            log_placeholder = st.empty()

            # ---------- Processing loop ----------
            for idx, row in df.iterrows():
                row_data = {c: row.get(c, "") for c in cols_to_use}
                content = ""

                # -------------------------------------------------------------
                # 🔍 GHS Pictogram Detector
                # -------------------------------------------------------------
                if prompt_choice == "GHS Pictogram Detector":
                    image_urls = row_data.get("image_link", "")
                    image_list = [image_urls.strip()] if image_urls.strip() else []
                    pictograms_found = set()
                    debug_notes_all = []

                    for url in image_list:
                        try:
                            resp = client.chat.completions.create(
                                **build_chat_args(
                                    "gpt-4o",
                                    selected_prompt_text,
                                    json.dumps([{"type": "image_url", "image_url": {"url": url}}]),
                                    temperature_val
                                )
                            )
                            result = json.loads(resp.choices[0].message.content.strip())
                            icons = [i.strip() for i in result.get("pictograms", "").split(",") if i.strip()]
                            pictograms_found.update(icons)
                            debug_notes_all.append(result.get("debug_notes", ""))
                        except Exception as e:
                            failed_rows.append(idx)
                            results.append({"error": f"Error for image: {url}", "debug_notes": str(e)})
                            break

                    results.append({
                        "pictograms": ", ".join(sorted(pictograms_found)),
                        "debug_notes": " | ".join(debug_notes_all)
                    })
                    continue  # skip to next row

                # -------------------------------------------------------------
                # ⚖️ HFSS Checker
                # -------------------------------------------------------------
                if prompt_choice == "HFSS Checker":
                    try:
                        # Pass 1 – extract structured nutrients
                        resp1 = client.chat.completions.create(
                            **build_chat_args(model_choice, build_pass_1_prompt(row_data), "", temperature_val)
                        )
                        parsed_1 = json.loads(clean_gpt_json_block(resp1.choices[0].message.content))

                        # Pass 2 – compute NPM score
                        resp2 = client.chat.completions.create(
                            **build_chat_args(model_choice, build_pass_2_prompt(parsed_1), "", temperature_val)
                        )
                        parsed_2 = json.loads(clean_gpt_json_block(resp2.choices[0].message.content))

                        # Pass 3 – determine HFSS status
                        resp3 = client.chat.completions.create(
                            **build_chat_args(model_choice, build_pass_3_prompt({**parsed_2, "is_drink": parsed_1.get("is_drink", False)}), "", temperature_val)
                        )
                        parsed_3 = json.loads(clean_gpt_json_block(resp3.choices[0].message.content))

                        # Pass 4 – final validator
                        all_passes = {"parsed_nutrients": parsed_1, "npm_scoring": parsed_2, "hfss_classification": parsed_3}
                        resp4 = client.chat.completions.create(
                            **build_chat_args(model_choice, build_pass_4_prompt(all_passes), "", temperature_val)
                        )
                        parsed_4 = json.loads(clean_gpt_json_block(resp4.choices[0].message.content))

                        results.append({**parsed_1, **parsed_2, **parsed_3, **parsed_4})

                    except Exception as e:
                        failed_rows.append(idx)
                        results.append({"error": f"Row {idx}: {e}"})
                    continue

                # -------------------------------------------------------------
                # 🖼️ Multi-Image Ingredient Extract
                # -------------------------------------------------------------
                if prompt_choice == "Image: Multi-Image Ingredient Extract & Compare":
                    image_urls = row.get("image URLs", "")
                    image_list = [u.strip().replace('"', '') for u in image_urls.split(",") if u.strip()]
                    extracted = []

                    for url in image_list:
                        try:
                            resp = client.chat.completions.create(
                                **build_chat_args("gpt-4o", selected_prompt_text,
                                                  json.dumps([{"type": "image_url", "image_url": {"url": url}}]),
                                                  temperature_val)
                            )
                            content = resp.choices[0].message.content.strip()
                            if content and "IMAGE_UNREADABLE" not in content.upper():
                                extracted.append(content)
                        except Exception as e:
                            extracted.append(f"[ERROR: {e}]")

                    combined_html = "\n".join(extracted).strip()
                    reference = row.get("full_ingredients", "")

                    results.append({
                        "extracted_ingredients": combined_html,
                        "comparison_result": "Pass" if combined_html in reference else "Needs Review"
                    })
                    continue

                # -------------------------------------------------------------
                # 🏷️ Competitor SKU Match
                # -------------------------------------------------------------
                if prompt_choice == "Competitor SKU Match":
                    try:
                        my_sku = parse_sku(row[sku_col])
                        candidates = top_candidates(my_sku, db=COMP_DB, k=8)
                        cand_list = [c for c, _ in candidates]

                        if not cand_list:
                            results.append({
                                "match_found": "No",
                                "best_match_uid": "",
                                "best_match_name": "",
                                "confidence_pct": 0,
                                "reason": "No candidate met minimum rules"
                            })
                            continue

                        system_prompt = build_match_prompt(my_sku, cand_list)
                        resp = client.chat.completions.create(
                            **build_chat_args(model_choice, system_prompt, "", temperature_val)
                        )
                        gpt_json = json.loads(resp.choices[0].message.content)

                        best = next((c for c in cand_list if c.uid == gpt_json.get("best_match_uid")), None)
                        results.append({
                            **gpt_json,
                            "best_match_uid": getattr(best, "uid", ""),
                            "best_match_name": getattr(best, "raw_name", ""),
                            "candidate_debug": [(c.raw_name, s) for c, s in candidates]
                        })
                    except Exception as e:
                        failed_rows.append(idx)
                        results.append({"error": f"Row {idx}: {e}"})
                    continue

                # -------------------------------------------------------------
                # 🚫 Banned/Restricted Checker
                # -------------------------------------------------------------
                if prompt_choice == "Banned/Restricted Checker":
                    try:
                        ing_text = row_data.get(banned_ing_col, "")
                        if not ing_text.strip():
                            results.append({"overall": {"banned_present": False, "restricted_present": False}, "items": []})
                        else:
                            cands = find_banned_matches(ing_text, threshold=banned_fuzzy_threshold, return_details=True)
                            if not cands:
                                results.append({"overall": {"banned_present": False, "restricted_present": False}, "items": []})
                            else:
                                system_txt = build_banned_prompt(cands, ing_text)
                                resp = client.chat.completions.create(
                                    **build_chat_args(model_choice, system_txt, "", temperature_val)
                                )
                                parsed = json.loads(clean_gpt_json_block(resp.choices[0].message.content))
                                parsed["candidates_debug"] = cands
                                results.append(parsed)
                    except Exception as e:
                        failed_rows.append(idx)
                        results.append({"error": f"Row {idx}: {e}"})
                    continue

                # -------------------------------------------------------------
                # 🌍 Novel Food Checker
                # -------------------------------------------------------------
                if prompt_choice == "Novel Food Checker (EU)":
                    from prompts.novel_check_utils import find_novel_matches, build_novel_food_prompt
                    ing_text = row_data.get("full_ingredients", "")
                    if not ing_text:
                        results.append({"novel_food_flag": "No", "confirmed_matches": []})
                        continue

                    matches = find_novel_matches(ing_text, threshold=fuzzy_threshold, return_scores=True)
                    candidate_matches = [term for term, _ in matches]
                    debug_scores = matches

                    if candidate_matches:
                        system_txt = build_novel_food_prompt(candidate_matches, ing_text)
                        try:
                            resp = client.chat.completions.create(
                                **build_chat_args(model_choice, system_txt, "", temperature_val)
                            )
                            parsed = json.loads(clean_gpt_json_block(resp.choices[0].message.content))
                            parsed["fuzzy_debug_matches"] = debug_scores
                            results.append(parsed)
                        except Exception as e:
                            failed_rows.append(idx)
                            results.append({"error": f"Row {idx}: {e}"})
                    else:
                        results.append({"novel_food_flag": "No", "confirmed_matches": [], "fuzzy_debug_matches": debug_scores})
                    continue

                # -------------------------------------------------------------
                # 📝 Default GPT Call (for general/custom prompts)
                # -------------------------------------------------------------
                try:
                    if "USER MESSAGE:" in user_prompt:
                        system_txt, user_txt = user_prompt.split("USER MESSAGE:", 1)
                    else:
                        system_txt, user_txt = user_prompt, ""

                    system_txt = system_txt.replace("SYSTEM MESSAGE:", "").strip()
                    user_txt = user_txt.strip().format(**row_data)
                    user_txt += f"\n\nSelected fields:\n{json.dumps(row_data, ensure_ascii=False)}"

                    resp = client.chat.completions.create(
                        **build_chat_args(model_choice, system_txt, user_txt, temperature_val)
                    )
                    content = resp.choices[0].message.content.strip()

                    if content.startswith("```"):
                        parts = content.split("```", maxsplit=2)
                        content = parts[1].lstrip("json").strip().split("```")[0].strip()

                    parsed = json.loads(content)
                    results.append(parsed)
                except Exception as e:
                    failed_rows.append(idx)
                    results.append({"error": f"Row {idx}: {e}", "raw_output": content if content else "No content"})

                # -------------------------------------------------------------
                # ✅ Progress + Gauge + Debug Log (runs after every row)
                # -------------------------------------------------------------
                progress = (idx + 1) / n_rows
                progress_bar.progress(progress)
                progress_text.markdown(
                    f"<h4 style='text-align:center; color:#4A4443;'>Processed {idx + 1} of {n_rows} rows ({progress*100:.1f}%)</h4>",
                    unsafe_allow_html=True
                )

                # Rolling log (last 20 rows)
                if "rolling_log_dicts" not in st.session_state:
                    st.session_state.rolling_log_dicts = []
                st.session_state.rolling_log_dicts.append(results[-1])
                st.session_state.rolling_log_dicts = st.session_state.rolling_log_dicts[-20:]

                log_placeholder.empty()
                log_placeholder.markdown(
                    "<h4 style='color:#4A4443;'>📝 Recent Outputs (Last 20)</h4>",
                    unsafe_allow_html=True
                )

                # Show last 3 always
                for entry in st.session_state.rolling_log_dicts[-3:]:
                    log_placeholder.json(entry)

                # Expanders for earlier rows
                for i, entry in enumerate(st.session_state.rolling_log_dicts[:-3]):
                    row_num = (idx + 1) - (len(st.session_state.rolling_log_dicts) - i)
                    with log_placeholder.expander(f"Row {row_num} output", expanded=False):
                        st.json(entry)

                # Gauge indicator
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=progress * 100,
                    number={'font': {'color': '#4A4443'}},
                    title={'text': "Progress", 'font': {'color': '#4A4443'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': '#4A4443'},
                        'bar': {'color': "#C2EA46"},
                        'bgcolor': "#E1FAD1",
                        'borderwidth': 1,
                        'steps': [
                            {'range': [0, 50], 'color': "#E1FAD1"},
                            {'range': [50, 100], 'color': "#F2FAF4"}
                        ]
                    }
                ))
                gauge_placeholder.plotly_chart(fig, use_container_width=True)

            # ---------- end for loop ----------

            # Combine results with input CSV
            results_df = pd.DataFrame(results)
            final_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)

            st.success("✅ GPT processing complete!")

            st.markdown("<h3 style='color:#005A3F;'>🔍 Final Result</h3>", unsafe_allow_html=True)

            # Flatten JSON cells to strings
            final_df = final_df.applymap(_flatten)
            final_df = final_df.astype(str)

            # Preview rows
            max_preview = st.number_input(
                "How many rows would you like to preview?",
                min_value=1,
                max_value=min(1000, len(final_df)),
                value=min(20, len(final_df)),
                step=1
            )
            st.dataframe(final_df.head(int(max_preview)))

            # --- Download buttons ---
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

