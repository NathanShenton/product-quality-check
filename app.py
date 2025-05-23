import streamlit as st
import pandas as pd
import json
import os
import plotly.graph_objects as go
from streamlit_cropper import st_cropper    # NEW
from PIL import Image                       # NEW
import io                                   # NEW
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
        "gpt-3.5-turbo": (0.0005, 0.0015),
        "gpt-4.1-mini":  (0.0004, 0.0016),
        "gpt-4.1-nano":  (0.0001, 0.0004),
        "gpt-4o-mini": (0.0025, 0.0075),
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


#############################
# Pre-Written Prompts       #
#############################
PROMPT_OPTIONS = {
    "--Select--": {
        "prompt": "",
        "recommended_model": "gpt-3.5-turbo",
        "description": "No pre-written prompt selected."
    },
    "Gluten Free Contextual Check": {
        "prompt": (
            "SYSTEM MESSAGE:\n"
            "You are a JSON-producing assistant for high-criticality compliance checking. Your task is to review the ingredient list of a product claimed to be \"gluten free\" and identify any ingredient entries that appear inconsistent with that claim.\n\n"
            "You must:\n"
            "1. Consider the context of each ingredient — do not flag just based on keyword matches.\n"
            "2. Understand phrases like \"gluten free oats\" or \"barley malt extract (gluten removed)\" and evaluate whether the **modifier clearly negates the gluten content**.\n"
            "3. Only flag ingredients where the gluten presence is **likely or uncertain**, even after context is considered.\n\n"
            "Return JSON **only** in this format:\n"
            "{\n"
            "  \"gluten_conflicts\": [\"exact ingredient(s) causing concern\"],\n"
            "  \"overall\": \"Pass\" | \"Fail\"\n"
            "}\n\n"
            "Examples of gluten-containing ingredients: wheat, rye, barley, oats (unless explicitly gluten free), spelt, kamut, triticale, malt extract, semolina, farro.\n\n"
            "If you find such an ingredient but it's clearly labeled as gluten free (e.g., \"gluten free oats\"), do **not** flag it.\n\n"
            "If no conflicting or uncertain ingredients are found, return an empty array and \"Pass\".\n"
            "No disclaimers or extra commentary — JSON only.\n\n"
            "USER MESSAGE:\n"
            "Evaluate the following ingredient list for gluten-related concerns:\n"
            "{full_ingredients}"
    ),
        "recommended_model": "gpt-4.1-mini",
        "description": "Reviews 'full_ingredients' of gluten-free flagged products and flags likely or uncertain gluten sources while respecting context like 'gluten free oats'."
    },
    "Image: Ingredient Scrape (HTML)": {
        "prompt": (
            "SYSTEM MESSAGE:\n"
            "You are a **regulatory-grade data-capture agent** for UK food labels.\n"
            "Your sole task is to extract the EXACT ingredient list from the supplied product-pack image.\n"
            "Accuracy is **safety-critical**. Follow every rule below to the letter.\n\n"

            "STEP-BY-STEP RULES (do not skip):\n"
            "1. Perform high-fidelity OCR of the INGREDIENTS section only. Capture ALL text, including\n"
            "   bracketed sub-ingredients, percentages, E-numbers, and processing aids.\n"
            "2. **DO NOT** invent, re-order, translate, normalise, paraphrase, or summarise anything.\n"
            "   • If a word is illegible, replace just that word with \"[???]\" (max 3 unknowns allowed).\n"
            "3. Preserve original punctuation and capitalisation **except**:\n"
            "   • Convert a trailing full-stop to none (most UK packs do not end with a period).\n"
            "4. Bold every occurrence of the 14 UK FIC allergens **and their obvious synonyms**:\n"
            "   celery; cereals containing gluten (wheat, rye, barley, oats, spelt, kamut); crustaceans;\n"
            "   eggs; fish; lupin; milk; molluscs; mustard; **tree nuts** (almond, hazelnut, walnut,\n"
            "   cashew, pecan, pistachio, macadamia, Brazil nut); peanuts; sesame; soy/soya; sulphur\n"
            "   dioxide / sulphites.\n"
            "   • Example: <b>almonds</b>; <b>wheat</b>; <b>soya lecithin</b> (bold only the allergen token).\n"
            "5. Deduplicate nothing. If the label repeats an ingredient, you must repeat it.\n"
            "6. After processing, output **ONLY** the final HTML string — no disclaimers, no commentary,\n"
            "   no markdown fences.\n"
            "7. If the INGREDIENTS section is genuinely unreadable (e.g. obscured, missing, <3 words),\n"
            "   output exactly the text:  IMAGE_UNREADABLE  (all caps, no HTML).\n\n"

            "BEGIN EXTRACTION NOW."
    ),
        "recommended_model": "gpt-4o",
        "description": (
            "Literal OCR-grade extraction of the INGREDIENTS list with allergen tokens wrapped in "
            "<b></b>. Returns only an HTML string or the sentinel IMAGE_UNREADABLE."
    )
},
    "Image: Directions for Use": {
        "prompt": (
            "SYSTEM MESSAGE:\n"
            "You are a product data capture assistant for UK food and supplement labels.\n"
            "Your task is to extract any clear instructions about how the product should be consumed or used, based on the cropped image provided.\n\n"
            "RULES:\n"
            "1. Look for a section that includes consumption guidance — including phrases like 'Directions', 'How to use', 'Recommended use', 'Usage', or other instructions that explain how or when to take the product.\n"
            "2. Even if there is no heading, extract the block of text if the meaning clearly implies instructions for use.\n"
            "3. Do NOT invent or paraphrase. Return only the visible, legible printed text.\n"
            "4. Do NOT return HTML, markdown, or formatting — just plain text.\n"
            "5. If the image is unreadable or no usage guidance is clearly found, return exactly: IMAGE_UNREADABLE\n\n"
            "Your output must be clean plain text only."
        ),
        "recommended_model": "gpt-4o",
        "description": "More flexible extraction of usage instructions from cropped label. Looks for any clearly implied direction, even without a heading."
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
    "Image: Storage Instructions": {
        "prompt": (
            "SYSTEM MESSAGE:\n"
            "You are a product data extraction assistant. Your job is to extract the STORAGE INSTRUCTIONS section\n"
            "from a cropped product label image used in the UK retail context.\n\n"
            "RULES:\n"
            "1. Look for headings like: 'Storage', 'Keep in a cool place', 'How to store', or similar.\n"
            "2. Extract all nearby or indented text that forms part of the storage instruction.\n"
            "3. Do not paraphrase, reword, or summarise.\n"
            "4. Return only the exact printed text — no bullet points, no formatting, no markdown or HTML.\n"
            "5. If you cannot find the storage section or it is unreadable, return exactly: IMAGE_UNREADABLE"
        ),
        "recommended_model": "gpt-4o",
        "description": "Extracts storage guidance from the cropped label. Returns plain text only or IMAGE_UNREADABLE."
    },
    "Image: Warnings and Advisory (JSON)": {
        "prompt": (
            "SYSTEM MESSAGE:\n"
            "You are a food safety and regulatory assistant. You will extract and classify 3 types of messages from a UK label image:\n"
            "- Warnings (e.g., health risks, dosage errors, legal warnings)\n"
            "- Advisory notes (e.g., guidance such as 'consult a doctor')\n"
            "- May Contain statements (e.g., 'may contain traces of milk')\n\n"
            "TASK:\n"
            "1. Review the label text and extract any sentences that clearly belong to one of these three types.\n"
            "2. Classify each item into the correct category.\n"
            "3. If multiple messages appear in one category, concatenate them with line breaks.\n\n"
            "OUTPUT:\n"
            "Return a valid minified JSON object using this structure:\n\n"
            "{\n"
            "  \"filename\": \"[EXACT_FILENAME].png\",\n"
            "  \"warnings\": \"...\",\n"
            "  \"advisory\": \"...\",\n"
            "  \"may_contain\": \"...\"\n"
            "}\n\n"
            "RULES:\n"
            "- Fill each field with only the text directly from the image.\n"
            "- Leave any missing category as an empty string.\n"
            "- If the text is unreadable or the section is missing, set all fields to blank except 'may_contain': \"IMAGE_UNREADABLE\"\n"
            "- Return JSON only. No markdown, no commentary, no extra notes."
        ),
        "recommended_model": "gpt-4o",
        "description": "Extracts warnings, advisory, and may contain info from label. Outputs valid JSON with four fields."
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
        "- ingredients: {Other ingredients}\n"
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
            "You are a JSON-producing assistant performing a high-criticality data audit. "
            "Your sole task is to verify that a product—already marked ‘suitable for vegans’—contains "
            "no animal-derived ingredients. Use ONLY the text provided in the field "
            "‘full_ingredients’. Never invent data or guesses.\n\n"

            "Return valid JSON in exactly this format:\n"
            "{{{{\n"                              # doubled braces
            "  \"ingredient_conflicts\": [],\n"
            "  \"overall\": \"Pass\" | \"Fail\"\n"
            "}}}}\n\n"                            # doubled braces

            "USER MESSAGE:\n"
            "Evaluate the following ingredient list for vegan compliance:\n"
            "{full_ingredients}\n\n"              # ← keep single-braced placeholder
            "If you find any animal-derived term—examples include but are not limited to: gelatin "
            "(beef, pork, fish), beeswax, whey, honey, lanolin, carmine, shellac, collagen, casein, "
            "egg, milk, lactose, albumin, tallow, anchovy, crab, lobster, oyster sauce—add the exact "
            "spelling(s) as they appear to the array `ingredient_conflicts` and set `overall` to \"Fail\". "
            "If none are present, return an empty array and set `overall` to \"Pass\".\n\n"

            "**Important rules**:\n"
            "1. Matching is case-insensitive, but echo the ingredient exactly as supplied.\n"
            "2. Flag ONLY real matches from the provided text—no placeholders or interpretations.\n"
            "3. If any conflict is found, `overall` MUST be \"Fail\".\n"
            "4. Output the JSON object only—no prose, no explanations."
        ),
        "recommended_model": "gpt-4.1-mini",
        "description": "Validates vegan status solely from ingredient text; flags any animal-derived terms."
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

# --- Determine if image-based ---
is_image_prompt = prompt_choice.startswith("Image:")
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
                box_color='#4a90e2',
                realtime_update=False,
                aspect_ratio=None,
                return_type="image"
            )

        if st.button("✅ Use this crop →"):
            buf = io.BytesIO()
            cropped_img.save(buf, format="PNG")
            st.session_state["cropped_bytes"] = buf.getvalue()
else:
    uploaded_file = st.file_uploader("📁 Upload your CSV", type=["csv"])


# ---------------------------------------------------------------
# Image-prompt flow -– two-pass high-accuracy extraction
# ---------------------------------------------------------------

if is_image_prompt and st.session_state.get("cropped_bytes"):
    st.markdown("### 📤 Processing image…")
    with st.spinner("Running high-accuracy two-pass extraction"):
        # Enforce the correct model
        if model_choice != "gpt-4o":
            st.error("🛑  Image prompts require the **gpt-4o** model. "
                     "Please choose it above and try again.")
            st.stop()

        try:
            # Use the general crop+prompt pipeline for non-ingredient prompts
            if "Ingredient Scrape" in prompt_choice:
                html_out = two_pass_extract(st.session_state["cropped_bytes"])
            else:
                import base64
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


