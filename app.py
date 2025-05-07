import streamlit as st
import pandas as pd
import json
import os
import plotly.graph_objects as go
from openai import OpenAI

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
    **Welcome to our Flexible AI Product Data Checker!**

    - Process your CSV row by row with GPT.
    - Configure columns, select or write a prompt.
    - Get real-time cost estimates and results.

    *Demo powered by OpenAI's GPT models.*
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
        "gpt-3.5-turbo": (0.0005, 0.0015),
        "gpt-4.1-mini": (0.0004, 0.0016),
        "gpt-4.1-nano": (0.0001, 0.0004),
        "gpt-4o-mini":  (0.0006, 0.0024),
        "gpt-4-turbo":  (0.01,   0.03)
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

with col2:
    # 2. File Upload
    uploaded_file = st.file_uploader("📁 Upload your CSV", type=["csv"])

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
            "SYSTEM MESSAGE:\n"
            "\"You are a JSON-producing assistant. Never invent placeholder text. "
            "Return **valid JSON only** in exactly this shape:\\n\\n"
            "{\\n"
            "  \\\"pmo_category\\\": \\\"<category name or unsure>\\\",\\n"
            "  \\\"unit_price_basis\\\": \\\"per 10 g | per 10 ml | per 100 g | "
            "per 100 ml | per 750 ml | per 50 kg | per 1,000 kg | per 1 item\\\"\\n"
            "}\\n\\n"
            "No other keys are allowed.\"\\n\\n"
            "USER MESSAGE:\n"
            "Using the product data supplied in **Selected fields** (typically "
            "name, description, ingredients, SKU), decide which Schedule 1 "
            "category of the UK Price Marking Order 2004 the product belongs to "
            "and therefore which unit quantity the shelf-edge price must show. "
            "Pick **one** category from the list below and output the matching "
            "unit as shown. If nothing fits, use \"unsure\" and "
            "\"per 1 item\".\n\n"
            "— Herbs → per 10 g\n"
            "— Spices → per 10 g\n"
            "— Flavouring essences → per 10 ml\n"
            "— Food colourings → per 10 ml\n"
            "— Seeds (non-pea/bean) → per 10 g\n"
            "— Cosmetic make-up → per 10 g/ml (choose g for solids, ml for liquids)\n"
            "— Rice → per 100 g\n"
            "— Pickles → per 100 g\n"
            "— Sauces & edible oils → per 100 ml\n"
            "— Fresh processed salad → per 100 g\n"
            "— Chilled desserts → per 100 ml\n"
            "— Cream → per 100 ml\n"
            "— Bread → per 100 g\n"
            "— Biscuits → per 100 g\n"
            "— Pies/pasties/flans → per 100 g\n"
            "— Ice cream & frozen desserts → per 100 g/ml\n"
            "— Preserves → per 100 g\n"
            "— Soups → per 100 g\n"
            "— Fruit juices & soft drinks → per 100 ml\n"
            "— Coffee → per 100 g/ml\n"
            "— Tea & infusions → per 100 g\n"
            "— Confectionery → per 100 g\n"
            "— Snack foods (crisps, nuts, etc.) → per 100 g\n"
            "— Breakfast cereals → per 100 g\n"
            "— Dry sauce mixes → per 100 g\n"
            "— Lubricating oils (non-engine) → per 100 ml\n"
            "— Shaving creams → per 100 g/ml\n"
            "— Hand creams → per 100 ml\n"
            "— Lotions & creams → per 100 ml\n"
            "— Sun products → per 100 ml\n"
            "— Oral products (toothpaste, mouthwash) → per 100 g/ml\n"
            "— Hair lacquer → per 100 ml\n"
            "— Hair shampoos & conditioners → per 100 g/ml\n"
            "— Hair strengtheners & brilliantines → per 100 ml\n"
            "— Bubble-bath & shower foams → per 100 ml\n"
            "— Deodorants → per 100 g/ml\n"
            "— Talcum powders → per 100 g\n"
            "— Toilet soaps → per 100 g\n"
            "— Alcohol-based toiletries (<3 % perfume) → per 100 ml\n"
            "— Hand-rolling & pipe tobacco → per 100 g\n"
            "— Wine & fortified wine (750 ml pack) → per 750 ml\n"
            "— Coal → per 50 kg\n"
            "— Ballast → per 1,000 kg\n\n"
            "Rules:\n"
            "1. Choose the single most appropriate category.\n"
            "2. Base the decision on typical form (solid vs liquid).\n"
            "3. If ambiguous, return \"unsure\".\n"
            "4. Output the JSON only – no markdown, no extra text.\""
        ),
        "recommended_model": "gpt-4.1-mini",
        "description": "Maps each product to its Schedule 1 PMO category and the correct unit price basis (e.g. per 100 g, per 10 ml)."
    },
    "Free From Quick-Check": {
    "prompt": (
        "SYSTEM MESSAGE:\n"
        "You are a rule-based compliance screener.  Only two outputs are allowed:\n"
        " • {\"status\":\"ok\",\"reason\":\"\"}\n"
        " • {\"status\":\"needs_review\",\"reason\":\"<short text>\"}\n"
        "You MUST follow the vocabulary table below: a claim can only be challenged\n"
        "if an ingredient token appears in its dedicated list.  Anything else is ignored.\n\n"

        "VOCAB = {\n"
        "  \"gluten free\": [\"wheat\",\"barley\",\"rye\",\"oats\",\"spelt\",\"kamut\",\"triticale\",\"gluten\",\"malt\",\"semolina\"],\n"
        "  \"dairy free\":  [\"milk\",\"lactose\",\"whey\",\"casein\",\"cheese\",\"butter\",\"cream\"],\n"
        "  \"egg free\":    [\"egg\",\"albumin\",\"ovalbumin\"],\n"
        "  \"soya free\":   [\"soy\",\"soya\",\"soja\",\"lecithin (soya)\",\"soy lecithin\"],\n"
        "  \"peanut free\": [\"peanut\",\"peanuts\",\"groundnut\",\"arachis\"],\n"
        "  \"nut free\":    [\"almond\",\"hazelnut\",\"walnut\",\"cashew\",\"pecan\",\"pistachio\",\"macadamia\",\"brazil nut\"],\n"
        "  \"sesame seed free\": [\"sesame\",\"tahini\"],\n"
        "  \"fish free\":   [\"fish\",\"cod\",\"haddock\",\"tuna\",\"salmon\",\"sardine\",\"anchovy\"],\n"
        "  \"crustaceans free\": [\"prawn\",\"shrimp\",\"crab\",\"lobster\"],\n"
        "  \"mollusc free\": [\"mussel\",\"oyster\",\"clam\",\"squid\",\"octopus\"],\n"
        "  \"celery free\": [\"celery\",\"celeriac\"],\n"
        "  \"mustard free\": [\"mustard\"],\n"
        "  \"sulphites free\": [\"sulphite\",\"sulfur dioxide\",\"e220\",\"e221\",\"e222\",\"e223\",\"e224\",\"e225\",\"e226\",\"e227\",\"e228\"],\n"
        "  \"lupin free\": [\"lupin\",\"lupine\"]\n"
        "}\n\n"

        "USER MESSAGE:\n"
        "- ingredients: {Other Ingredients}\n"
        "- claims:      {Free From}\n\n"

        "STEPS (the only logic you may perform):\n"
        "1. Lower-case and strip HTML/punctuation from ingredients.\n"
        "2. Split into word tokens.\n"
        "3. For each claim, look up VOCAB[claim].\n"
        "4. If any token from that list is present → needs_review (reason: \"<Claim> conflict: <token>\").\n"
        "5. If zero conflicts → ok.\n"
        "6. Never flag tokens not in the list, even if they look related (e.g. lactose is *not* gluten).\n"
        "7. Output exactly one JSON object, minified.\n"
    ),
    "recommended_model": "gpt-4-turbo",
    "description": "Binary screener: flags SKUs only when a claim-specific keyword appears; no cross-claim leakage."
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
        "recommended_model": "gpt-4.1-mini",
        "description": "Write your own prompt below."
    }
}

# 4. Choose a Pre-Written Prompt
st.subheader("💬 Choose a Prompt")
prompt_choice = st.selectbox(
    "Select a pre-written prompt or 'Custom':",
    list(PROMPT_OPTIONS.keys()),
    index=0
)

# Extract chosen prompt details
selected_prompt_data = PROMPT_OPTIONS[prompt_choice]
selected_prompt_text = selected_prompt_data["prompt"]
recommended_model = selected_prompt_data["recommended_model"]
prompt_description = selected_prompt_data["description"]

st.markdown(f"**Prompt Info:** {prompt_description}")

# 5. Model Selector (default to recommended model, but user can override)
all_model_keys = list(MODEL_OPTIONS.keys())
default_index = all_model_keys.index(recommended_model) if recommended_model in all_model_keys else 0
model_choice = st.selectbox("🧠 Choose GPT model", all_model_keys, index=default_index)
st.markdown(f"**Model Info:** {MODEL_OPTIONS[model_choice]}")
st.markdown("---")

# 6. User Prompt Text Area (auto-filled if a pre-written prompt is selected)
user_prompt = st.text_area(
    "✍️ Your prompt for GPT",
    value=selected_prompt_text,
    height=200
)

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

                try:
                    # 1 – split the stored prompt into true roles
                    if "USER MESSAGE:" in user_prompt:
                        system_txt, user_txt = user_prompt.split("USER MESSAGE:", 1)
                    else:                       # fallback: whole prompt is system
                        system_txt, user_txt = user_prompt, ""

                    system_txt = system_txt.replace("SYSTEM MESSAGE:", "").strip()
                    user_txt = user_txt.strip().format(**row_data)
                    user_txt += f"\n\nSelected fields:\n{json.dumps(row_data, ensure_ascii=False)}"

                    # 2 – call OpenAI chat
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

                    # 3 – strip any ``` fences
                    if content.startswith("```"):
                        parts = content.split("```", maxsplit=2)
                        content = parts[1].lstrip("json").strip().split("```")[0].strip()

                    # 4 – parse JSON
                    parsed = json.loads(content)
                    results.append(parsed)

                    # 5 – rolling log (last 20 rows)
                    rolling_log.append(f"Row {idx + 1}: {json.dumps(parsed)}")
                    rolling_log = rolling_log[-20:]
                    log_placeholder.markdown(
                        "<h4>📝 Recent Outputs (Last 20)</h4>"
                        "<pre style='background:#f0f0f0; padding:10px; border-radius:5px; max-height:400px; overflow:auto;'>"
                        + "\n".join(rolling_log) +
                        "</pre>",
                        unsafe_allow_html=True
                    )

                except Exception as e:
                    failed_rows.append(idx)
                    error_result = {
                        "error": f"Failed to process row {idx}: {e}",
                        "raw_output": content if content else "No content returned"
                    }
                    results.append(error_result)

                    rolling_log.append(f"Row {idx + 1}: ERROR - {e}")
                    rolling_log = rolling_log[-20:]
                    log_placeholder.markdown(
                        "<h4>📝 Recent Outputs (Last 20)</h4>"
                        "<pre style='background:#f0f0f0; padding:10px; border-radius:5px; max-height:400px; overflow:auto;'>"
                        + "\n".join(rolling_log) +
                        "</pre>",
                        unsafe_allow_html=True
                    )

                # 6 – update progress UI
                progress = (idx + 1) / n_rows
                progress_bar.progress(progress)
                progress_text.markdown(
                    f"<h4 style='text-align:center;'>Processed {idx + 1} of {n_rows} rows ({progress*100:.1f}%)</h4>",
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
