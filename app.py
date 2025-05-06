import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
from openai import OpenAI

"""
Flexible AI Product Data Checker
-------------------------------
Unlimited column selection • Pre‑written prompt library • Token‑based cost estimate • Progress gauge & rolling log
"""

# ──────────────────────────────── 0 · PAGE CONFIG & STYLES
st.set_page_config(page_title="Flexible AI Product Data Checker", layout="wide")

st.markdown(
    """<style>
      body{background:#f4f7f6;font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif}
      .main{padding:2rem}
      h1{font-size:3.2rem;color:#4a90e2;text-align:center;margin-bottom:1rem}
      h2,h3,h4{color:#333}
      .stProgress>div>div>div{background:#4a90e2}
      .css-1d391kg .css-1d391kg{background:#fff;border-radius:5px;padding:1rem}
    </style>""",
    unsafe_allow_html=True,
)

# ──────────────────────────────── 1 · SIDEBAR
st.sidebar.markdown("# Flexible AI Checker")
st.sidebar.image(
    "https://cdn.freelogovectors.net/wp-content/uploads/2023/04/holland_and_barrett_logo-freelogovectors.net_.png",
    use_container_width=True,
)
st.sidebar.markdown(
    """**Welcome!**  
    • Process your CSV row‑by‑row with GPT  
    • Pick columns, choose / write a prompt  
    • See a live cost estimate and results""",
)

# ──────────────────────────────── 2 · HELPERS

def approximate_tokens(text: str) -> int:
    """4 chars ≈ 1 token."""
    return max(1, len(text) // 4)

def estimate_cost(model: str, df: pd.DataFrame, prompt: str, cols: list[str]) -> float:
    prices = {
        "gpt-3.5-turbo": (0.0005, 0.0015),
        "gpt-4.1-mini":  (0.0004, 0.0016),
        "gpt-4.1-nano":  (0.0001, 0.0004),
        "gpt-4o-mini":   (0.0006, 0.0024),
        "gpt-4-turbo":   (0.01,   0.03),
    }
    cin, cout = prices.get(model, (0.001, 0.003))
    total_in = total_out = 0
    base_tok = approximate_tokens(prompt)
    for _, row in df.iterrows():
        row_json = json.dumps({c: row.get(c, "") for c in cols}, ensure_ascii=False)
        total_in += 30 + base_tok + approximate_tokens(row_json)  # 30 ≈ system msg
        total_out += 100  # assume 100 output tokens
    return (total_in / 1_000)*cin + (total_out / 1_000)*cout

# ──────────────────────────────── 3 · MODEL INFO
MODEL_OPTIONS = {
    "gpt-3.5-turbo": "Cheapest, decent for basic tasks",
    "gpt-4.1-mini":  "Good balance of cost & reasoning",
    "gpt-4.1-nano":  "Ultra‑cheap & fast for lightweight checks",
    "gpt-4o-mini":   "Better quality than 4.1‑mini, still affordable",
    "gpt-4-turbo":   "Powerful but pricey – for complex jobs",
}

# ──────────────────────────────── 4 · TOP‑LEVEL UI (title, key, file)
st.markdown("<h1>📄 Flexible AI Product Data Checker with Cost Estimate</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;font-size:16px'>Process a CSV row‑by‑row with OpenAI. Pick columns, choose a prompt, pick a model.</p>", unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    api_key = st.text_input("🔑 Enter your OpenAI API key", type="password")
    if not api_key:
        st.warning("Please enter your API key to continue.")
        st.stop()
    client = OpenAI(api_key=api_key)
with c2:
    uploaded = st.file_uploader("📁 Upload your CSV", type=["csv"])

#############################
# Pre-Written Prompts       #
#############################
PROMPT_OPTIONS = {
    "--Select--": {
        "prompt": "",
        "recommended_model": "gpt-3.5-turbo",
        "description": "No pre-written prompt selected."
    },
    "Spelling Checker": {
        "prompt": (
            "SYSTEM MESSAGE:\n"
            "\"You are a JSON-producing assistant. You never invent placeholder text. "
            "Only output real spelling mistakes from 'variants_description' in the format (wrong->correct). "
            "If you find none, respond with 'errors': '' and no placeholders.\"\n\n"
            "USER MESSAGE:\n"
            "Analyze the text in 'variants_description' for real English misspellings. "
            "Output valid JSON only, like this:\n\n"
            "{\n  \"errors\": \"(wrong->right),(wrong2->right2)\"\n}\n\n"
            "If no mistakes, use:\n\n"
            "{\n  \"errors\": \"\"\n}\n\n"
            "**Important**:\n"
            "1. Do not list brand or domain words.\n"
            "2. Never produce placeholders like 'mispell1->correct1'.\n"
            "3. If uncertain, skip the word.\n"
            "4. No disclaimers or instructions—just do it."
        ),
        "recommended_model": "gpt-3.5-turbo",
        "description": "Use gpt-3.5-turbo for a balance of cost and complexity."
    },
        "Price Marking Order Category": {
        "prompt": (
            "SYSTEM MESSAGE:\\n"
            "\"You are a JSON-producing assistant. Never invent placeholder text. "
            "Return **valid JSON only** in exactly this shape:\\n\\n"
            "{\\n"
            "  \\\"pmo_category\\\": \\\"<category name or unsure>\\\",\\n"
            "  \\\"unit_price_basis\\\": \\\"per 10 g | per 10 ml | per 100 g | "
            "per 100 ml | per 750 ml | per 50 kg | per 1,000 kg | per 1 item\\\"\\n"
            "}\\n\\n"
            "No other keys are allowed.\"\\n\\n"
            "USER MESSAGE:\\n"
            "Using the product data supplied in **Selected fields** (typically "
            "name, description, ingredients, SKU), decide which Schedule 1 "
            "category of the UK Price Marking Order 2004 the product belongs to "
            "and therefore which unit quantity the shelf-edge price must show. "
            "Pick **one** category from the list below and output the matching "
            "unit as shown. If nothing fits, use \\\"unsure\\\" and "
            "\\\"per 1 item\\\".\\n\\n"
            "— Herbs → per 10 g\\n"
            "— Spices → per 10 g\\n"
            "— Flavouring essences → per 10 ml\\n"
            "— Food colourings → per 10 ml\\n"
            "— Seeds (non-pea/bean) → per 10 g\\n"
            "— Cosmetic make-up → per 10 g/ml (choose g for solids, ml for liquids)\\n"
            "— Rice → per 100 g\\n"
            "— Pickles → per 100 g\\n"
            "— Sauces & edible oils → per 100 ml\\n"
            "— Fresh processed salad → per 100 g\\n"
            "— Chilled desserts → per 100 ml\\n"
            "— Cream → per 100 ml\\n"
            "— Bread → per 100 g\\n"
            "— Biscuits → per 100 g\\n"
            "— Pies/pasties/flans → per 100 g\\n"
            "— Ice cream & frozen desserts → per 100 g/ml\\n"
            "— Preserves → per 100 g\\n"
            "— Soups → per 100 g\\n"
            "— Fruit juices & soft drinks → per 100 ml\\n"
            "— Coffee → per 100 g/ml\\n"
            "— Tea & infusions → per 100 g\\n"
            "— Confectionery → per 100 g\\n"
            "— Snack foods (crisps, nuts, etc.) → per 100 g\\n"
            "— Breakfast cereals → per 100 g\\n"
            "— Dry sauce mixes → per 100 g\\n"
            "— Lubricating oils (non-engine) → per 100 ml\\n"
            "— Shaving creams → per 100 g/ml\\n"
            "— Hand creams → per 100 ml\\n"
            "— Lotions & creams → per 100 ml\\n"
            "— Sun products → per 100 ml\\n"
            "— Oral products (toothpaste, mouthwash) → per 100 g/ml\\n"
            "— Hair lacquer → per 100 ml\\n"
            "— Hair shampoos & conditioners → per 100 g/ml\\n"
            "— Hair strengtheners & brilliantines → per 100 ml\\n"
            "— Bubble-bath & shower foams → per 100 ml\\n"
            "— Deodorants → per 100 g/ml\\n"
            "— Talcum powders → per 100 g\\n"
            "— Toilet soaps → per 100 g\\n"
            "— Alcohol-based toiletries (<3 % perfume) → per 100 ml\\n"
            "— Hand-rolling & pipe tobacco → per 100 g\\n"
            "— Wine & fortified wine (750 ml pack) → per 750 ml\\n"
            "— Coal → per 50 kg\\n"
            "— Ballast → per 1,000 kg\\n\\n"
            "Rules:\\n"
            "1. Choose the single most appropriate category.\\n"
            "2. Base the decision on typical form (solid vs liquid).\\n"
            "3. If ambiguous, return \\\"unsure\\\".\\n"
            "4. Output the JSON only – no markdown, no extra text.\""
        ),
        "recommended_model": "gpt-4.1-mini",
        "description": "Maps each product to its Schedule 1 PMO category and the correct unit price basis (e.g. per 100 g, per 10 ml)."
    },
    "French Sell Copy Translator": {
        "prompt": (
            "SYSTEM MESSAGE:\n"
            "\"You are a JSON-producing assistant. You never invent placeholder text. "
            "Only output real translations from 'variants_description' in the format (english->french) - no placeholders.\"\n\n"
            "USER MESSAGE:\n"
            "Analyze the text in 'variants_description' and convert it correctly to French. "
            "Output valid JSON only, like this:\n\n"
            "{\n  \"translation\": \"(english||french)\"\n}\n\n"
            "**Important**:\n"
            "1. Never produce placeholders.\n"
            "2. Translate intelligently and ensure it reads correctly in French.\n"
            "3. Maintain HTML formatting, paragraph breaks, punctuation, etc.\n"
            "4. No disclaimers or instructions—just do it."
        ),
        "recommended_model": "gpt-4.1-mini",
        "description": "Use gpt-4.1-mini for more nuanced translation tasks."
    },
    "Product Usage Type Checker": {
            "prompt": (
                "SYSTEM MESSAGE:\n"
                "\"You are a JSON-producing assistant. You never invent placeholder text. "
                "You must respond with valid JSON in this format:\n\n"
                "{\n  \"product_usage_type\": \"consumable\" or \"topical\" or \"other\" or \"unsure\"\n}\n\n"
                "No other fields are allowed.\"\n\n"
                "USER MESSAGE:\n"
                "Review the product data (name, description, ingredients, etc.) provided in 'Selected fields'. "
                "Determine the most appropriate product usage type:\n\n"
                "- If the product is typically *ingested* by mouth (food, drink, supplement, sports powder, etc.), output:\n"
                "{\n  \"product_usage_type\": \"consumable\"\n}\n\n"
                "- If the product is typically *applied externally to the body* (skin, hair, teeth, mouth, etc.), output:\n"
                "{\n  \"product_usage_type\": \"topical\"\n}\n\n"
                "- If the product is *not used on or in the body* (e.g., accessories, containers, devices), output:\n"
                "{\n  \"product_usage_type\": \"other\"\n}\n\n"
                "- If you are *uncertain or cannot tell* based on the provided information, output:\n"
                "{\n  \"product_usage_type\": \"unsure\"\n}\n\n"
                "No disclaimers or instructions—only valid JSON."
            ),
            "recommended_model": "gpt-4.1-mini",
            "description": "Classifies product usage type as consumable, topical, other, or unsure based on product data."
    },
    "Vegan Flag Check": {
        "prompt": (
            "SYSTEM MESSAGE:\n"
            "You are a JSON-producing assistant. You only output real validation results based on product origin "
            "and ingredients. Never invent placeholder text or guesses.\n\n"
            "Only return JSON in this format:\n"
            "{\n"
            "  \"is_animal_origin_flag\": \"Valid\" or \"Invalid\",\n"
            "  \"ingredient_conflict\": \"None\" or ingredient name,\n"
            "  \"overall\": \"Pass\" or \"Fail\"\n"
            "}\n\n"
            "USER MESSAGE:\n"
            "This product has been pre-flagged as suitable for vegans based on upstream filters.\n\n"
            "Validate its vegan status by:\n"
            "1. Confirming 'is_animal_origin' is 0. If it is not, mark as Invalid.\n"
            "2. Checking 'full_ingredients' for animal-derived terms like:\n"
            "   gelatin, beeswax, whey, honey, lanolin, carmine, shellac, collagen, casein, egg, milk, lactose, "
            "albumin, or similar.\n\n"
            "Return JSON only, in this format:\n"
            "{\n"
            "  \"is_animal_origin_flag\": \"Valid\" or \"Invalid\",\n"
            "  \"ingredient_conflict\": \"None\" or ingredient name,\n"
            "  \"overall\": \"Pass\" or \"Fail\"\n"
            "}\n\n"
            "**Important**:\n"
            "1. Do not invent placeholder ingredient names.\n"
            "2. Only flag real ingredient matches from the input.\n"
            "3. If any conflict is found, overall must be \"Fail\".\n"
            "4. No disclaimers or instructions—just return valid JSON."
        ),
        "recommended_model": "gpt-4.1-mini",
        "description": "Checks if product is truly vegan or not, marking conflicts."
    },
    "Methylated Vitamin Check": {
        "prompt": (
            "SYSTEM MESSAGE:\n"
            "\"You are a JSON-producing assistant. You never invent placeholder text and you must "
            "return **valid JSON only** in exactly this shape:\n\n"
            "{\n"
            "  \"contains_methylated_vitamin\": \"Yes\" | \"No\" | \"Unsure\",\n"
            "  \"match\": \"<exact text you matched, if any>\"\n"
            "}\n\n"
            "No other keys are allowed.\"\n\n"
            "USER MESSAGE:\n"
            "Look at the product data provided in 'Selected fields'. Decide whether the ingredients "
            "contain **any methylated form of folate (B9) or B12**. These are the accepted synonyms:\n\n"
            "— *Methylfolate* group: 5-MTHF, L-5-MTHF, L-5-Methyltetrahydrofolate, "
            "5-Methyltetrahydrofolic acid, Levomefolate, Metafolin®, Quatrefolic®, Magnafolate®.\n"
            "— *Methylcobalamin* group: Methylcobalamin, Methyl-B12, MeB12, MeCbl, Mecobalamin.\n\n"
            "Rules:\n"
            "1. Ignore case, punctuation, HTML tags, and trademark symbols (®/™).\n"
            "2. If you find any synonym, output:\n"
            "{ \"contains_methylated_vitamin\": \"Yes\", \"match\": \"<exact synonym you saw>\" }\n"
            "3. If you are certain none are present, output:\n"
            "{ \"contains_methylated_vitamin\": \"No\", \"match\": \"\" }\n"
            "4. If the text is blank, corrupted, or you really cannot tell, output:\n"
            "{ \"contains_methylated_vitamin\": \"Unsure\", \"match\": \"\" }\n\n"
            "No explanations, no markdown fences, only the JSON."
        ),
        "recommended_model": "gpt-4.1-mini",
        "description": "Flags products that contain methylfolate or methylcobalamin, returning Yes/No/Unsure plus the matched string."
    },
    "Allergen Bold Check": {
        "prompt": (
            "SYSTEM MESSAGE:\n"
            "\"You are a JSON-producing assistant. You never invent placeholder text. Only output real findings about unbolded allergens in HTML-coded ingredients. No disclaimers or extra explanations — just valid JSON.\n\n"
            "Follow these rules carefully:\n\n"
            "1) We are searching the text of an HTML-coded ingredient list for mentions of the 14 main allergens:\n\n"
            "   - celery\n"
            "   - cereals containing gluten (including wheat, rye, barley, oats)\n"
            "   - crustaceans\n"
            "   - eggs\n"
            "   - fish\n"
            "   - lupin\n"
            "   - milk\n"
            "   - molluscs\n"
            "   - mustard\n"
            "   - nuts (including almonds, hazelnuts, walnuts, cashews, pecans, Brazil nuts, pistachios, macadamias)\n"
            "   - peanuts\n"
            "   - sesame\n"
            "   - soy (soya)\n"
            "   - sulphites (SO2)\n\n"
            "2) Only flag an allergen if:\n"
            "   - Its name (or one of its recognized forms) appears outside of any <b>, <B>, <strong>, or <STRONG> tags (case-insensitive for the tags).\n"
            "   - It is not exclusively in a “may contain” or “traces of” statement. (Ignore mentions if they appear only in a 'may contain' or similar disclaimer.)\n\n"
            "3) Consider synonyms or variations:\n"
            "   - “Cereals containing gluten” = any unbolded instance of “wheat,” “rye,” “barley,” “oat,” or “oats.” If found, return it as “cereals containing gluten.”\n"
            "   - “Milk” includes unbolded mentions of “milk powder,” “skimmed milk,” “whey (milk),” “casein,” etc. Return simply “milk.”\n"
            "   - “Soy” includes unbolded “soy,” “soya,” “soja.”\n"
            "   - “Nuts” includes unbolded “almonds,” “hazelnuts,” “walnuts,” “cashews,” “pecans,” “Brazil nuts,” “pistachios,” “macadamias.” Return simply “nuts.”\n"
            "   - “Sulphites” includes unbolded “sulphites,” “SO2,” “sulfur dioxide,” etc.\n\n"
            "4) If part of the allergen word is bolded and part is not, treat it as unbolded. (For instance, `<b>m</b>ilk` means “milk” is not fully bolded, so it should be flagged.)\n\n"
            "5) In the JSON you return, list each allergen only once, in lowercase, separated by commas. If none are found unbolded, return an empty string.\n\n"
            "6) The output must be valid JSON with exactly this shape:\n\n"
            "{\n"
            "  \"unbolded_allergens\": \"milk, fish, celery\"\n"
            "}\n\n"
            "or, if none are found:\n\n"
            "{\n"
            "  \"unbolded_allergens\": \"\"\n"
            "}\n"
        ),
        "recommended_model": "gpt-4.1-mini",
        "description": "Checks HTML-coded ingredient data for any unbolded allergens with detailed rules."
    },
    "Custom": {
        "prompt": "",
        "recommended_model": "gpt-3.5-turbo",
        "description": "Write your own prompt below."
    }
}

# ──────────────────────────────── 6 · PROMPT & MODEL SELECTORS
st.subheader("💬 Choose a Prompt")
choice = st.selectbox("Prompt:", list(PROMPT_OPTIONS.keys()), index=0)
sel = PROMPT_OPTIONS[choice]
user_prompt = st.text_area("✍️ Prompt text", value=sel["prompt"], height=200)
st.markdown(f"**Prompt Info:** {sel['description']}")

models = list(MODEL_OPTIONS.keys())
model_choice = st.selectbox("🧠 GPT model", models, index=models.index(sel["recommended_model"]))
st.markdown(f"**Model Info:** {MODEL_OPTIONS[model_choice]}")
st.markdown("---")

# ──────────────────────────────── 7 · MAIN EXECUTION
if uploaded and user_prompt.strip():
    df = pd.read_csv(uploaded, dtype=str)
    st.markdown("### 📄 CSV Preview")
    st.dataframe(df.head())

    # 7.1 Column picker
    st.subheader("📊 Select columns to pass to GPT")
    default_cols = [c for c in ("sku_id","name","size","ingredients") if c in df.columns]
    cols = st.multiselect("Columns:", df.columns.tolist(), default=default_cols or df.columns[:3])
    if not cols:
        st.error("Please select at least one column.")
        st.stop()

    # 7.2 Cost estimate
    cost = estimate_cost(model_choice, df, user_prompt, cols)
    st.markdown(f"<div style='padding:10px;background:#2a2a2a;color:#fff;border-radius:5px'>Estimated Cost: ${cost:0.4f}</div>", unsafe_allow_html=True)

    # 7.3 Run button
    gauge_placeholder = st.empty()
    if st.button("🚀 Run GPT on CSV"):
        with st.spinner("Processing with GPT..."):
            n_rows = len(df)
            progress_bar = st.progress(0)
            progress_txt = st.empty()
            log_placeholder = st.empty()
            rolling_log = []
            results = []
            failed = []

            for idx, row in df.iterrows():
                row_data = {c: row.get(c, "") for c in cols}
                content = ""
                try:
                    resp = client.chat.completions.create(
                        model=model_choice,
                        temperature=0.0,
                        messages=[
                            {"role":"system","content":"You are a JSON‑producing assistant. Return valid JSON only."},
                            {"role":"user","content":f"{user_prompt}\n\nSelected fields:\n{json.dumps(row_data,ensure_ascii=False)}"},
                        ],
                    )
                    content = resp.choices[0].message.content.strip()
                    # strip markdown fences if needed
                    if content.startswith("```"):
                        chunks = content.split("```")
                        content = chunks[1].lstrip("json").strip() if len(chunks)>1 else content
                    parsed = json.loads(content)
                    results.append(parsed)
                    rolling_log.append(f"Row {idx+1}: {json.dumps(parsed)}")
                except Exception as e:
                    failed.append(idx)
                    results.append({"error":str(e),"raw_output":content})
                    rolling_log.append(f"Row {idx+1}: ERROR - {e}")
                # keep last 20 log lines
                if len(rolling_log) > 20:
                    rolling_log = rolling_log[-20:]
                log_placeholder.markdown("<h4>📝 Recent Outputs (last 20)</h4><pre style='background:#f0f0f0;padding:10px;border-radius:5px;max-height:400px;overflow:auto'>"+"\n".join(rolling_log)+"</pre>", unsafe_allow_html=True)

                progress = (idx+1)/n_rows
                progress_bar.progress(progress)
                progress_txt.markdown(f"<h4 style='text-align:center'>Processed {idx+1}/{n_rows} ({progress*100:.1f}%)</h4>", unsafe_allow_html=True)

                fig = go.Figure(go.Indicator(mode="gauge+number", value=progress*100, title={"text":"Progress"}, gauge={"axis":{"range":[0,100]},"bar":{"color":"#4a90e2"}}))
                gauge_placeholder.plotly_chart(fig, use_container_width=True)

            # 7.4 Merge & downloads
            results_df = pd.DataFrame(results)
            final_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)
            st.success("✅ GPT processing complete!")
            st.markdown("### 🔍 Final Result")
            st.dataframe(final_df)

            st.download_button(
                "⬇️ Download Full Results CSV",
                final_df.to_csv(index=False).encode("utf-8"),
                "gpt_output.csv",
                "text/csv",
            )
            if failed:
                failed_df = df.iloc[failed].copy()
                st.warning(f"{len(failed)} rows failed. Download and retry.")
                st.download_button(
                    "⬇️ Download Failed Rows CSV",
                    failed_df.to_csv(index=False).encode("utf-8"),
                    "gpt_failed_rows.csv",
                    "text/csv",
                )
