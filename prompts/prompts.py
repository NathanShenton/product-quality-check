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
    "Food Supplement Compliance Check": {
        "prompt": (
            "SYSTEM MESSAGE:\n"
            "You are a JSON-producing assistant for high-criticality compliance checking. Your task is to review all provided product information for a single SKU (which may include ingredients, descriptions or sell copy, nutritional details, directions, warnings, quantity, etc.) and:\n"
            "1. Determine whether the product is a food supplement based on any of the provided data.\n"
            "2. If the product is a food supplement, ensure the marketing/sell copy, directions, and warnings include all required UK food supplement labeling elements:\n"
            "   – An advised daily dose (e.g., “Take one capsule daily”).\n"
            "   – A “Do not exceed the recommended dose” clause.\n"
            "   – A statement that it “should not be used as a substitute for a varied diet.”\n"
            "   – “Keep out of reach of young children.”\n"
            "   – Any other mandatory warnings or qualifiers (e.g., “Consult your doctor if pregnant or breastfeeding”).\n"
            "If the product is not a food supplement, simply report that.\n\n"
            "Return JSON only in this exact format:\n"
            "{\n"
            "  \"is_food_supplement\": true | false,\n"
            "  \"justification\": \"<brief rationale based on all data>\",\n"
            "  \"compliance_check\": {\n"
            "    \"overall\": \"Pass\" | \"Fail\",\n"
            "    \"missing_elements\": [\n"
            "      \"<element_name>\",\n"
            "      ...\n"
            "    ]\n"
            "  }\n"
            "}\n\n"
            "If \"is_food_supplement\" is false, set \"justification\" accordingly and omit \"compliance_check\" or leave its fields empty.\n"
            "No disclaimers or extra commentary—JSON only.\n\n"
            "USER MESSAGE:\n"
            "Here is all the product data for the SKU:\n"
            "{all_fields}"
        ),
        "recommended_model": "gpt-4o-mini",
        "description": "Holistically determines if a product is a food supplement from any provided fields, then verifies required UK labeling elements."
    },
    "Medicinal Language Compliance Checker": {
        "prompt": (
            "SYSTEM MESSAGE:\n"
            "\"You are a JSON-producing assistant that evaluates product marketing copy for regulatory compliance. "
            "You must NOT hallucinate or assume context beyond what is written. "
            "Your task is to assess whether the product description contains *medicinal language* likely to be non-compliant for food and drink products under ASA/FSA guidance.\n\n"
    
            "You must respond with valid JSON in this exact format:\n\n"
            "{\n"
            "  \"medicinal_language_flag\": \"Yes\" | \"No\",\n"
            "  \"matched_category\": \"<reason category or 'None'>\",\n"
            "  \"explanation\": \"<brief explanation if flagged, or empty if 'No'>\"\n"
            "}\n\n"
    
            "Return only one of the allowed values for `medicinal_language_flag`: 'Yes' or 'No'. "
            "Return only the categories listed below. Never output any other string or free text. "
            "Return nothing except the JSON object.\"\n\n"
    
            "USER MESSAGE:\n"
            "Evaluate the following product description **in isolation**. Determine whether it includes language that clearly suggests a *medicinal function*, such as treating, preventing, or curing a disease or adverse health condition.\n\n"
    
            "Only flag if the text clearly implies a functional effect on health **beyond general wellness**. Avoid flagging vague lifestyle claims, traditional uses, or non-medical benefits.\n\n"
    
            "Flag as 'Yes' **only if** the description:\n"
            "- Clearly suggests treatment, relief, or reduction of a specific symptom or condition (e.g. pain, anxiety, IBS, hot flushes)\n"
            "- Claims to prevent or protect against a health condition (e.g. colds, viruses, stress-related illness)\n"
            "- References detoxification or cleansing of organs in a therapeutic sense\n"
            "- Mentions hormonal regulation or menopause symptom management with an implied health benefit\n"
            "- Indicates mood or psychological enhancement with medical framing (e.g. depression, burnout)\n"
            "- References named medical conditions in the context of effect or benefit (e.g. cancer, high blood pressure)\n"
            "- Uses words like 'heal', 'repair', 'fight', 'boost', or 'restore' **in a way that implies medical action**\n\n"
    
            "Do NOT flag if:\n"
            "- The effect is framed as traditional, cosmetic, or personal care (e.g. skin moisturising, used in Ayurveda)\n"
            "- The language is clearly compliant and aligned to approved health claim structure (e.g. 'contributes to the normal function of...')\n"
            "- The product describes general lifestyle positioning without linking to specific health improvements\n"
            "- No health-related benefit is actually claimed\n\n"
    
            "Allowed values for `matched_category`:\n"
            "• \"Symptom Relief\"\n"
            "• \"Disease Treatment or Cure\"\n"
            "• \"Disease Prevention\"\n"
            "• \"Detox or Organ Repair\"\n"
            "• \"Mental Health Claims\"\n"
            "• \"Hormonal or Menopause Claims\"\n"
            "• \"Immune Claims\"\n"
            "• \"Medical Condition References\"\n"
            "• \"Pharmacological Action\"\n"
            "• \"None\" ← if compliant\n\n"
    
            "Return nothing but the JSON object.\n\n"
            "Selected field:\n"
            "- Product Description: {{product_description}}\n"
        ),
        "recommended_model": "gpt-4o-mini",
        "description": "Contextually flags non-compliant medicinal language in food and drink product copy, reducing false positives by focusing on explicit therapeutic claims."
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
    "Image: Multi-Image Ingredient Extract & Compare": {
        "prompt": (
            "SYSTEM MESSAGE:\n"
            "You are a high-accuracy image OCR assistant for UK food product packaging.\n\n"
            "Your job is to extract only the INGREDIENTS section text from product label images provided as base64.\n\n"
            "RULES:\n"
            "1. Ignore all other sections (e.g. usage, warnings, storage).\n"
            "2. Preserve all original punctuation, capitalisation, E-numbers, and sub-ingredients.\n"
            "3. Bold every UK FIC allergen or known variant using <b>…</b> tags — even partial matches inside words.\n"
            "4. Return an HTML string or IMAGE_UNREADABLE if no INGREDIENTS are found.\n\n"
            "UK FIC allergens to bold: celery, cereals containing gluten (wheat, rye, barley, oats, spelt, kamut), crustaceans, eggs, fish, lupin, milk, molluscs, mustard, nuts (almond, hazelnut, walnut, cashew, pecan, pistachio, macadamia, Brazil nut), peanuts, sesame, soy/soya, sulphur dioxide/sulphites.\n\n"
            "Your output must be either:\n"
            "• a full HTML <p>…</p> string with allergens bolded, or\n"
            "• IMAGE_UNREADABLE (exact text) if the section is illegible."
        ),
        "recommended_model": "gpt-4o",
        "description": (
            "Extracts INGREDIENTS from multiple product image URLs and highlights allergens "
            "using <b>…</b>. Compares result to the 'full_ingredients' field from the same row."
        )
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
            "You are a product label transcription assistant. Your task is to extract the product's STORAGE instructions from the supplied UK food or supplement label image.\n\n"
            "RULES:\n"
            "1. Only copy text that clearly refers to **how or where the product should be stored**.\n"
            "   Examples: 'Store in a cool, dry place.', 'Keep refrigerated.', 'Do not freeze.'\n"
            "2. The text may or may not follow a heading like 'Storage', 'Keep', or 'How to store'. That’s okay.\n"
            "3. **Do NOT paraphrase, invent, summarise, or complete sentences** — only transcribe what's visible.\n"
            "4. Keep original punctuation and spelling. Do not correct grammar or add structure.\n"
            "5. Return plain text only. No HTML, markdown, or commentary.\n"
            "6. If you see no valid storage text, or the image is unreadable, return exactly: IMAGE_UNREADABLE\n\n"
            "Begin by copying the storage text exactly as it appears. Return nothing else."
    ),
    "recommended_model": "gpt-4o",
    "description": "Safely extracts storage text from label images, verbatim only. Avoids paraphrasing or guessing."
    },
    "Product Name & Variant Extractor": {
        "prompt": (
            "SYSTEM MESSAGE:\n"
            "\"You are a JSON-producing assistant. You never invent placeholder text. "
            "You must respond with valid JSON in this exact format:\n\n"
            "{\n"
            "  \"product_name\": \"<Product Name>\",\n"
            "  \"variant_name\": \"<Variant Name>\"\n"
            "}\n\n"
            "No other fields are allowed.\"\n\n"
            "USER MESSAGE:\n"
            "Using the product data in **Selected fields**, return values for the two attributes below.\n\n"
            "► **Product Name**\n"
            "• Brand + primary product description (product type).\n"
            "• Exclude strength, flavour or scent unless their removal makes the name unclear.\n"
            "• **MUST NOT repeat any words that appear in Variant Name. Remove them.**\n"
            "• Always Proper Case; drop trademark symbols (™, ®, &trade;, etc.).\n"
            "• Examples: Holland & Barrett Vitamin D3; Nakd Raw Fruit & Nut Bar.\n\n"
            "► **Variant Name**\n"
            "• Only the key distinguishing element *other than* size/pack-count.\n"
            "• Acceptable: flavour, strength, scent, functional claim.\n"
            "• **Never** include quantity (ml, g, L, kg) or pack size (e.g. “3×50 g”).\n"
            "• If size/pack is the only difference, set Variant Name to an empty string \"\".\n"
            "• Proper Case; no trademark symbols.\n"
            "• Examples:\n"
            "    – Salted Caramel\n"
            "    – 1000 mg\n"
            "    – Blood Orange & Rosemary\n"
            "    – \"\"   ← when no valid variant exists\n\n"
            "Return nothing but the JSON object.\n\n"
            "Selected fields:\n"
            "- Brand: {{brand}}\n"
            "- SKU Name: {{sku_name}}\n"
            "- Quantity: {{quantity_string}}\n"
        ),
        "recommended_model": "gpt-4o-mini",
        "description": "Extracts `product_name` and `variant_name`, ensuring variant never contains size/pack info and product name never repeats variant content."
},
    "Gelatin Source Classifier": {
        "prompt": (
            "SYSTEM MESSAGE:\n"
            "\"You are a JSON-producing assistant. You must not hallucinate or assume information. "
            "You must respond with valid JSON in this exact format:\n\n"
            "{\n"
            "  \"gelatin_source\": \"<Gelatin - Beef | Gelatin - Porcine | Gelatin - Unknown Origin>\"\n"
            "}\n\n"
            "Return only one of the three allowed values. Never output any other string or text. "
            "Only choose 'Beef' or 'Porcine' if the ingredient statement **explicitly** states the animal source. "
            "If no animal is stated, select 'Gelatin - Unknown Origin'.\"\n\n"
            "USER MESSAGE:\n"
            "Review the provided product data to determine the origin of gelatin used. "
            "Use the ingredient statement as the **primary source of truth**. "
            "Only use the variant description if it provides additional explicit information. "
            "Never infer or guess the source. Do not use brand reputation or assumed product category.\n\n"
            "Allowed values for `gelatin_source`:\n"
            "• \"Gelatin - Beef\"        ← if the ingredient statement mentions beef gelatin, bovine gelatin, or equivalent\n"
            "• \"Gelatin - Porcine\"     ← if the ingredient statement mentions pork gelatin, porcine gelatin, or equivalent\n"
            "• \"Gelatin - Unknown Origin\" ← if the source is not mentioned or unclear\n\n"
            "Return nothing but the JSON object.\n\n"
            "Selected fields:\n"
            "- SKU: {{sku}}\n"
            "- Full Ingredients: {{full_ingredients}}\n"
            "- Variant Description: {{variants_description}}\n"
        ),
        "recommended_model": "gpt-4o-mini",
        "description": "Classifies the animal origin of gelatin using only explicit evidence from the ingredient statement or variant description."
    },
    "Image: Warnings and Advisory (JSON)": {
        "prompt": (
            "SYSTEM MESSAGE:\n"
            "You are a food safety and regulatory extraction assistant. You will identify and classify any relevant text from a UK product label image into the following categories:\n"
            "- **warnings** (e.g., health risks, dosage risks, serious safety notices)\n"
            "- **advisory** notes (e.g., consult a doctor, not suitable for...)\n"
            "- **may_contain** statements (e.g., 'may contain traces of nuts')\n\n"
            "TASK:\n"
            "1. Read the entire image text. Do **not** rely only on headings — also check for standalone lines that match the meaning of each category.\n"
            "2. If a line clearly matches one of the three types, assign it to that category.\n"
            "3. If you're unsure where something fits, use 'advisory' as the fallback.\n"
            "4. If multiple items appear in one category, separate them using line breaks ('\\n').\n\n"
            "OUTPUT:\n"
            "Return a valid, minified JSON object using this structure:\n"
            "{\n"
            "  \"filename\": \"[EXACT_FILENAME].png\",\n"
            "  \"warnings\": \"...\",\n"
            "  \"advisory\": \"...\",\n"
            "  \"may_contain\": \"...\"\n"
            "}\n\n"
            "RULES:\n"
            "- Only copy exact label text. Do NOT invent or complete sentences.\n"
            "- Do not summarise or reword anything.\n"
            "- If a field is empty, use an empty string (\"\").\n"
            "- If nothing relevant is readable, use this output exactly:\n"
            "{ \"filename\": \"[FILENAME].png\", \"warnings\": \"\", \"advisory\": \"\", \"may_contain\": \"IMAGE_UNREADABLE\" }\n"
            "- Output JSON only. No extra text, no markdown."
        ),
        "recommended_model": "gpt-4o",
        "description": "Flexibly extracts warnings, advisory, and 'may contain' text based on content, not just headings. Outputs exact JSON."
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
    "Legal Category Classifier": {
        "prompt": (
            """SYSTEM MESSAGE:\\n
                "You are a JSON-producing assistant. Valid output:\\n\\n
                {\\n  \\\"legal_category\\\": \\\"Ambient Food\\\" | \\\"Chilled Food\\\" | \\\"Cosmetic\\\" | \\\"Food Supplement (Liquid)\\\" | \\\"Food Supplement (Solid)\\\" | 
                \\\"GSL Medicine\\\" | \\\"Homeopathic\\\" | \\\"Medical Device\\\" | \\\"Other - General Merchandise\\\" | \\\"Traditional Herbal Remedy\\\" | \\\"unsure\\\"\\n}\\n\\n
                Rules (read carefully):\\n
                • Return **exactly one** value. If any doubt remains, output \\\"unsure\\\".\\n
                • Use \\\"Other - General Merchandise\\\" only when the product is clearly a non-food, non-supplement, non-cosmetic, non-medical item.\\n
                • Assign \\\"Medical Device\\\", \\\"Homeopathic\\\", \\\"GSL Medicine\\\", or \\\"Traditional Herbal Remedy\\\" **only when there is explicit, authoritative evidence** (e.g. the text literally states the class or provides the correct regulatory code/number). Otherwise choose \\\"unsure\\\".\\n
                    – If is_medical_device is true **and** the text states \\\"medical device\\\", choose \\\"Medical Device\\\".\\n
                    – If thr contains a valid THR code (format \\\"THR00000/0000\\\"), choose \\\"Traditional Herbal Remedy\\\"; otherwise never assign that category.\\n
                    – If product_licence starts with \\\"PL \\\" (e.g. \\\"PL 01234/0567\\\") and the text confirms GSL status, choose \\\"GSL Medicine\\\".\\n
                    – If product_licence starts with \\\"NR \\\" or the text says \\\"homeopathic medicine\\\", choose \\\"Homeopathic\\\".\\n
                • Assign \\\"Food Supplement (Liquid/Solid)\\\" **only when** BOTH conditions hold:\\n
                    1. The product text (name, description, or warnings) explicitly calls itself a \\\"food supplement\\\" or \\\"dietary supplement\\\" (exact phrase).\\n
                    2. It provides a measured daily dose **and** lists NRVs/\\u0025RIs or says \\\"do not exceed the recommended intake\\\".\\n
                  Measured dose alone is **not** enough.\\n
                  Products described as \\\"tea\\\", \\\"herbal infusion\\\", \\\"tisane\\\", or sold in \\\"tea bags\\\" are **not** supplements unless they meet both criteria above.\\n
                • For supplements, distinguish Liquid vs Solid by keywords in lexmark_pack_size:\\n
                    \\\"ml\\\", \\\"l\\\", \\\"liquid\\\", \\\"shot\\\", \\\"syrup\\\" → Liquid; \\\"g\\\", \\\"tablet\\\", \\\"capsule\\\", \\\"sachet\\\", \\\"bar\\\" → Solid.\\n
                • Distinguish \\\"Ambient Food\\\" vs \\\"Chilled Food\\\" via storage cues (e.g. \\\"store below 5 °C\\\", \\\"keep refrigerated\\\").\\n\\n
                No explanations, disclaimers, or extra keys — output only the valid JSON."\\n\\n
                USER MESSAGE:\\n
                Classify the following product:\\n\\n
                - sku: {sku}\\n
                - sku_name: {sku_name}\\n
                - variants_description: {variants_description}\\n
                - full_ingredients: {full_ingredients}\\n
                - directions_info: {directions_info}\\n
                - warning_info: {warning_info}\\n
                - lexmark_pack_size: {lexmark_pack_size}\\n
                - is_medical_device: {is_medical_device}\\n
                - thr: {thr}\\n
                - product_licence: {product_licence}\\n\\n
                Only respond with the JSON described above."""
        ),
        "recommended_model": "gpt-4.1-mini",
        "description": "Classifies products into legally defined categories or returns 'unsure' when data are inadequate."
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
