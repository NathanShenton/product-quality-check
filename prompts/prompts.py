#############################
# Pre-Written Prompts       #
#############################
PROMPT_OPTIONS = {
    "--Select--": {
        "prompt": "",
        "recommended_model": "gpt-3.5-turbo",
        "description": "No pre-written prompt selected."
    },
    "INCOMPLETE: Gluten Free Contextual Check": {
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
    "COMPLETE: Nutrient Data Only": {
        "prompt": (
            "SYSTEM MESSAGE:\n"
            "\"You are a JSON-producing assistant that parses a raw `nutritionals_info` array for a single SKU and extracts the per-100 g values of key nutrients. "
            "Never hallucinate or assume facts; analyse ONLY the supplied data.\"\n\n"
    
            "Respond with valid JSON ONLY in exactly this shape:\n"
            "{\n"
            "    \"sku\": \"<the SKU from the input>\",\n"
            "    \"name\": \"<the product name from the input>\",\n"
            "    \"energy_kj_per_100g\": <number|null>,\n"
            "    \"saturated_fat_g_per_100g\": <number|null>,\n"
            "    \"sugars_g_per_100g\": <number|null>,\n"
            "    \"salt_mg_per_100g\": <number|null>,\n"
            "    \"fibre_g_per_100g\": <number|null>,\n"
            "    \"protein_g_per_100g\": <number|null>\n"
            "}\n\n"
    
            "RULES:\n"
            "1. Identify per-100 g values: if a `value` contains two parts separated by '/', take the first part as the per-100 g value; if only one value is present, assume it refers to per 100 g.\n"
            "2. Normalise units:\n"
            "   • 'g' → grams\n"
            "   • 'kJ' → kilojoules\n"
            "   • 'mg' → milligrams\n"
            "   • 'µg' or 'mcg' → micrograms (ignore for this task)\n"
            "3. Salt handling:\n"
            "   • If sodium is provided instead of salt, convert sodium (mg) to salt (mg) using: `salt_mg = sodium_mg × 2.5`\n"
            "   • Prioritise 'salt' if both 'salt' and 'sodium' are present\n"
            "4. Extract and report per-100 g values for:\n"
            "   • Energy (kJ)\n"
            "   • Saturated fat (g)\n"
            "   • Total sugars (g)\n"
            "   • Salt (in milligrams per 100 g)\n"
            "   • Fibre (g)\n"
            "   • Protein (g)\n"
            "5. If any nutrient is missing, unparseable, or not applicable, set its value to `null`.\n\n"
    
            "INPUT DATA:\n"
            "{{product_data}}\n"
        ),
        "recommended_model": "gpt-4o-mini",
        "description": (
            "Extracts per-100 g nutrients from `nutritionals_info`, normalises units "
            "(including sodium→salt conversion), and outputs in structured JSON format without NPM scoring."
        )
    },
    "COMPLETE: Age Restriction Compliance Checker": {
        "prompt": (
            "SYSTEM MESSAGE:\n"
            "You are a JSON-producing assistant that decides whether a product sold by "
            "Holland & Barrett must be age-restricted. Base your answer **only** on the supplied "
            "product data and the decision rules below. Never hallucinate.\n\n"
    
            "GENERAL ANALYSIS INSTRUCTIONS (read carefully):\n"
            "• Treat {{product_data}} as a dictionary that may contain fields such as "
            "  name, description, ingredients, nutrition_table, directions, warnings, packaging_size.\n"
            "• **Search ALL fields case-insensitively** for keywords and numeric values; do not rely on just one.\n"
            "• When a rule depends on a quantity (e.g. caffeine mg/100 ml) you must:\n"
            "    – Extract every occurrence of a number-unit pair (mg, g, %, ml, kcal,…).\n"
            "    – Normalise units (e.g. convert 80 mg/330 ml into 24.24 mg/100 ml).\n"
            "    – Compare the normalised value to the threshold.\n"
            "• If a nutrient panel gives caffeine **per serving**, first derive mg/100 ml using the declared "
            "serving size or pack volume; if impossible, treat as ambiguous and note it in debug_notes.\n"
            "• Apply the rules **holistically**: a caffeine claim in the description AND an explicit "
            "nutrition value together can confirm the threshold, even if each alone is ambiguous.\n"
            "• Be liberal in recognising synonyms and inflections (e.g. “appetite-control”, "
            "“supports testosterone”).\n\n"
    
            "Respond with **valid JSON ONLY** in exactly this shape:\n"
            "{\n"
            "  \"age_restriction_flag\": \"Yes\" | \"No\",\n"
            "  \"required_minimum_age\": \"16+\" | \"18+\" | \"None\",\n"
            "  \"matched_policy_section\": \"<one category below or 'None'>\",\n"
            "  \"explanation\": \"<brief reason if flagged, else empty>\",\n"
            "  \"debug_notes\": [\"<optional low-confidence signals considered>\"]\n"
            "}\n\n"
    
            "Allowed values for `matched_policy_section` (policy-mandated ages in brackets):\n"
            "• \"Weight-Loss / Appetite Suppressant\" (18+)\n"
            "• \"Sexual Vitality / Performance\" (18+)\n"
            "• \"Energy Drink ≥15 mg/100 ml Caffeine\" (18+)\n"
            "• \"High-Caffeine Food ≥150 mg/serving\" (18+)\n"
            "• \"Creatine\" (18+)\n"
            "• \"Restricted CBD\" (18+)\n"
            "• \"OTC Medicine\" (16+)\n"
            "• \"Medical Device (incl. THR)\" (16+)\n"
            "• \"Traditional Herbal Remedy – THR\" (18+)\n"
            "• \"CLP-Regulated Chemical / Corrosive\" (18+)\n"
            "• \"None\"\n\n"
    
            "DECISION RULES (apply top-down; stop at first clear match):\n"
            "1. **Weight-loss signals**\n"
            "   • Keywords: burn fat, appetite suppress/control, weight-loss, diet pill, keto-burner…\n"
            "   • Look in name, description, directions, claims.\n"
            "2. **Sexual vitality signals**\n"
            "   • Libido, testosterone boost, virility, sperm count, erectile support…\n"
            "3. **Caffeine thresholds**\n"
            "   • Drinks: ≥15 mg/100 ml (after unit normalisation).\n"
            "   • Foods / shots / tablets: ≥150 mg per declared serving.\n"
            "4. **Creatine present** (any creatine salt or “creatine” keyword in ingredients).\n"
            "5. **Restricted CBD cannabinoids** as per policy list.\n"
            "6. **Regulatory flags**\n"
            "   • Explicit \"OTC\", \"THR\", “medical device class…”.\n"
            "7. **Corrosive / CLP chemicals** (e.g. sodium hydroxide, hydrochloric acid).\n"
            "8. If more than one category matches, choose the **highest** minimum age.\n"
            "9. If nothing matches with high confidence, return:\n"
            "   {\"age_restriction_flag\":\"No\",\"required_minimum_age\":\"None\",\"matched_policy_section\":\"None\"}\n\n"
    
            "DEBUG NOTES:\n"
            "• List any partial or ambiguous cues not strong enough to trigger a flag.\n"
            "  Example: [\"caffeine 80 mg/250 ml can → 32 mg/100 ml (<15 mg), no restriction\"]\n"
            "• Use [] if no such notes.\n\n"
    
            "PRODUCT DATA:\n"
            "{{product_data}}\n"
        ),
        "recommended_model": "gpt-4o-mini",
        "description": "Uses holistic field parsing and numeric normalisation to flag weight-loss, libido, high-caffeine, CBD, OTC/THR & similar items requiring an age gate."
    },
    "INCOMPLETE: HFSS Checker": {
        "prompt": "(no prompt needed – handled via 4-pass logic)",
        "recommended_model": "gpt-4.1-mini",
        "description": "Four-pass NPM & HFSS classifier using structured GPT logic."
    },
    "COMPLETE: FNV Line-by-Line Estimator": {
        "prompt": (
            "    ### ROLE\n"
            "    Return STRICT JSON estimating Fruit-Nut-Veg share. Favour slight under-counts.\n\n"
    
            "    ### JSON SHAPE (copy exactly)\n"
            "    {\n"
            "        \"certain_fnv\": <0-100 int>,\n"
            "        \"presumptive_fnv\": <0-100 int>,\n"
            "        \"fnv_ingredients\": [\"<ingredient1>\", \"<ingredient2>\", ...],  # unique, lowercase, no excluded words\n"
            "        \"debug_notes\": [\"<≤20-word note>\", ...]\n"
            "    }\n\n"
    
            "    ### FNV vs EXCLUSIONS (memorise!)\n"
            "    Fruits: culinary fruits, berries, citrus, coconut, cocoa, tomato.\n"
            "    Nuts/Seeds: tree nuts, peanuts, sesame, chia, flax, sunflower, pumpkin.\n"
            "    Veg & Legumes: horticultural veg, pulses, tubers, herbs.\n"
            "        • Tea & herbal infusions count **only if the edible leaf/flower is consumed** (e.g. matcha powder, dried nettle in soup);\n"
            "          exclude when merely steeped then discarded (typical tea-bag infusion).\n"
            "    EXCLUDE always: wheat, barley, rye, oats, corn/maize, rice, quinoa, spelt, sorghum, millet, teff; "
            "any bran/flour; refined protein isolates; dairy; water; **oils or butters**; **spices/bark (cinnamon, nutmeg, etc.)**; all additives.\n\n"
    
            "    ### RULE 1 – CERTAIN_FNV  (collect, SUM, then verify)\n"
            "    • Collect **ALL** printed percentages attached to eligible FNV ingredients and SUM them → certain_fnv.\n"
            "      Example: “green tea (36%), peppermint (36%)” → +36 +36 (=72).\n"
            "    • If a bracket lists ONLY excluded items **and no other FNV** → subtract that % and add note “❗ % removed (non-FNV blend)”.\n"
            "    • Never invent or apportion unseen percentages.\n\n"
    
            "    ### RULE 2 – PRESUMPTIVE_FNV  (runs only if certain_fnv is still 0)\n"
            "    1st eligible FNV   → +30 %.\n"
            "    2nd eligible FNV   → +15 %   (cap presumptive at 45 %).\n"
            "    Seasonings / extracts / colour powders → ≤1 % each.\n"
            "    Veg crisps → 50 %.\n"
            "    Water-first beverage: first fruit/veg concentrate after water → 10 %.\n\n"
    
            "    ### RULE 3 – SPECIAL CASES\n"
            "    Stand-alone nut/seed + salt/oil → certain_fnv = 0, presumptive_fnv = 90.\n"
            "    Multi-flavour packs: score each sub-recipe, keep the single highest FNV % as certain_fnv.\n\n"
    
            "    ### RULE 4 – ROUND & CAP\n"
            "    Round BOTH numbers **UP** to whole integers; ensure certain + presumptive ≤ 100.\n\n"
    
            "    ### RULE 5 – FINAL SELF-AUDIT (run AFTER rules 1-4)\n"
            "    – Purge any EXCLUDE word (or token ending oil/butter/isolate/spice/bark) from fnv_ingredients and fix the maths.\n"
            "    – Assert every ingredient remaining in fnv_ingredients contributed to certain_fnv or presumptive_fnv; if not, move it to EXCLUDE and explain.\n"
            "    – Re-run RULE 4.\n"
            "    – Ensure debug_notes match the final figures.\n\n"
    
            "    ### HARD TESTS (fail ⇒ fix and re-run audit)\n"
            "    1. certain_fnv > 0 ⇒ presumptive_fnv **MUST** be 0.\n"
            "    2. certain_fnv + presumptive_fnv ≤ 100 & both whole ints.\n"
            "    3. certain_fnv equals the **sum** of declared % attached to items in fnv_ingredients.\n"
            "    4. No invented % values.\n"
            "    5. Debug notes explain the final maths in ≤20 words and reference any exclusions.\n\n"
    
            "    Respond **ONLY** with the JSON.\n\n"
            "    ### PRODUCT DATA\n"
            "    {{product_data}}\n"
        ),
        "recommended_model": "gpt-4.1-mini",
        "description": "Low-temp prompt (0.1) that sums all printed % for eligible FNV items, excludes spices/bark and non-consumed infusions, and keeps presumptive caps (≤45 %) to reduce HFSS false-compliance risk."
    },
    "INCOMPLETE: NPM & HFSS Classification": {
        "prompt": (
            "SYSTEM MESSAGE:\n"
            "You are a JSON-only assistant for calculating UK Nutrient Profiling Model (2004/05) "
            "scores and HFSS compliance.\n"
            "\n"
            "# DATA NORMALISATION RULES\n"
            "- Any value with no 'per serving' qualifier must be treated as per-100 g "
            "(or per-100 ml for drinks).\n"
            "- Accept flexible nutrient keys (e.g. 'Energy (kj)' -> energy_kj).\n"
            "- If salt_g is present but sodium_mg is not, compute sodium_mg = round(salt_g * 400).\n"
            "- Never invent nutrients that are missing from the input.\n"
            "\n"
            "# FOOD POINT TABLES (cut-offs for 1-10 pts)\n"
            "Energy_kJ: 335 670 1005 1340 1675 2010 2345 2680 3015 3350\n"
            "SatFat_g : 1 2 3 4 5 6 7 8 9 10\n"
            "Sugars_g : 4.5 9 13.5 18 22.5 27 31 36 40.5 45\n"
            "Sodium_mg: 90 180 270 360 450 540 630 720 810 900\n"
            "FVN_%    : 40 60 80 -> 1 2 5 pts\n"
            "Fibre_g  : 0.9 1.9 2.8 3.7 4.7 -> 1-5 pts\n"
            "Protein_g: 1.6 3.2 4.8 6.4 8   -> 1-5 pts\n"
            "\n"
            "# DRINK POINT TABLES (only if is_drink true)\n"
            "Energy_kJ: 30 60 90 120 150 180 210 240 270 300\n"
            "... (add remaining drink tables here) ...\n"
            "\n"
            "# TASK\n"
            "1. Parse nutritionals; convert per-serving to per-100.\n"
            "2. Extract per-100 energy_kj, saturated_fat_g, total_sugars_g, "
            "sodium_mg|salt_g, fibre_g, protein_g.\n"
            "3. Detect fruit_veg_nut_pct:\n"
            "   - If full_ingredients starts with '100 %', set to 100.\n"
            "   - Else if explicit 'xx %' present, use that value.\n"
            "4. Choose FOOD tables unless is_drink == true.\n"
            "5. Score A-points and C-points.\n"
            "6. Protein cap: if A >= 11 and FVN < 5 then protein_pts = 0.\n"
            "7. Compute:\n"
            "   - worse_case_npm_score (missing C nutrients = 0 pts)\n"
            "   - npm_with_fnv_consideration (include FVN pts if % known)\n"
            "8. Derive hfss_legislation and hfss_category per UK thresholds.\n"
            "9. INTERNAL SELF-CHECK: ensure 0 <= each point <= 10 and score <= 40.\n"
            "\n"
            "Respond ONLY with JSON:\n"
            "{\n"
            "  \"worse_case_npm_score\": <number>,\n"
            "  \"hfss_legislation\": \"In Scope\" | \"Out of Scope\",\n"
            "  \"hfss_category\": \"Less healthy\" | \"Not HFSS\",\n"
            "  \"npm_with_fnv_consideration\": <number|null>,\n"
            "  \"debug_notes\": \"<concise explanation>\"\n"
            "}\n"
            "\n"
            "USER MESSAGE:\n"
            "{{PRODUCT_DATA}}"
        ),
        "recommended_model": "gpt-4.1-mini",
        "description": (
            "Normalises nutritionals, handles food vs drink tables, converts salt->sodium, "
            "computes NPM scores, checks HFSS status, and self-verifies."
        )
    },
    "COMPLETE: Competitor SKU Match": {
        "prompt": "(auto-generated, not used directly)",
        "recommended_model": "gpt-4.1-mini",
        "description": "Find the best competitor product match for each SKU"
    },
    "COMPLETE: Prohibited Marketplace Compliance Checker": {
        "prompt": (
            "SYSTEM MESSAGE:\n"
            "\"You are a JSON-producing assistant that evaluates product data for compliance with "
            "Holland & Barrett’s Prohibited Marketplace List. Never hallucinate or assume facts. "
            "Analyse ONLY the supplied data.\"\n\n"
    
            "Respond with valid JSON ONLY in exactly this shape:\n"
            "{\n"
            "  \"prohibited_flag\": \"Yes\" | \"No\",\n"
            "  \"matched_category\": \"<one category below or 'None'>\",\n"
            "  \"explanation\": \"<brief reason if flagged, else empty>\",\n"
            "  \"debug_notes\": [\"<optional list of partial or low-confidence concerns>\"]\n"
            "}\n\n"
    
            "Allowed values for `matched_category`:\n"
            "• \"Stolen or Unauthorised Products\"\n"
            "• \"Counterfeit or IP Violations\"\n"
            "• \"Offensive or Reputational Risk\"\n"
            "• \"Illegal or Regulatory Risk\"\n"
            "• \"Second Hand Product\"\n"
            "• \"Infant Food or Weaning Aid\"\n"
            "• \"Medicinal Product\"\n"
            "• \"Contains Microbeads\"\n"
            "• \"Contains Single Use Plastic\"\n"
            "• \"CBD Not on FSA List\"\n"
            "• \"Age Restricted: Alcohol\"\n"
            "• \"Age Restricted: Tobacco\"\n"
            "• \"Age Restricted: Vape or E-Cigarette\"\n"
            "• \"Age Restricted: Matches or Lighters\"\n"
            "• \"Age Restricted: Weapons or Fireworks\"\n"
            "• \"Age Restricted: Knives or Blades\"\n"
            "• \"Age Restricted: Corrosive Substances\"\n"
            "• \"Age Restricted: Videos or Games\"\n"
            "• \"None\"\n\n"
    
            "Set `prohibited_flag` = \"Yes\" ONLY when the match is clear and specific. "
            "If uncertain, return \"No\" and `matched_category` = \"None\".\n\n"
    
            "–––––  MEDICINAL PRODUCT LOGIC  –––––\n"
            "Flag as \"Medicinal Product\" **if at least ONE** of these is true:\n"
            "1. The text claims to **treat, cure, relieve, or prevent** a disease, symptom, or dysfunction "
            "(e.g. arthritis, insomnia, PCOS, liver disease, testosterone deficiency).\n"
            "2. It uses explicit medicinal verbs: \"treats\", \"heals\", \"cures\", \"eliminates pain\", "
            "\"anti-inflammatory\", \"antiviral\", \"antibacterial\", \"lowers blood sugar\", etc.\n"
            "3. It references an MHRA/EMA regulatory status: \"Traditional Herbal Registration (THR)\", "
            "\"PL 12345/0001\", \"homeopathic remedy\", \"herbal medicine\", medical-device class, etc.\n"
            "4. The product is **explicitly marketed for a specific medical condition** "
            "(e.g. phrases like \"PCOS supplement\", \"menopause support\", \"for eczema\") even without verbs; "
            "the condition must be medical, not generic wellbeing.\n\n"
    
            "Do **NOT** flag as medicinal when the ONLY evidence is:\n"
            "• **Permitted EFSA structure-function claims** such as:\n"
            "  – \"Contributes to the maintenance of normal bones/teeth/muscle function\"\n"
            "  – \"Contributes to normal energy-yielding metabolism\"\n"
            "  – \"Supports normal psychological function\"\n"
            "  – \"Reduces tiredness and fatigue\"\n"
            "  – \"Helps maintain normal testosterone levels\" (zinc), etc.\n"
            "• **Traditional or historical framing** (e.g. \"used for 400 years\", \"Ayurvedic herb\").\n"
            "• Generic wellbeing language: \"balance\", \"vitality\", \"overall health\", \"adaptogen\".\n"
            "• Standard food-supplement disclaimers: pregnancy / children / consult GP.\n"
            "• Common supplement ingredients (magnesium, vitamin C, turmeric, etc.) **without** disease claims.\n\n"
    
            "–––––  ADDITIONAL CATEGORY RULES  –––––\n"
            "AGE-RESTRICTED BLADES\n"
            "• Flag \"Age Restricted: Knives or Blades\" if any field contains: knife, knives, blade, razor, scalpel, "
            "machete, axe, cleaver, chopper, box-cutter, stanley, sword, katana, kukri, sharpened, "
            "or a size reference such as \"5\" blade\".\n\n"
            "INFANT FOOD / WEANING AID\n"
            "• Flag \"Infant Food or Weaning Aid\" when BOTH:\n"
            "  1. The product is edible (puree, pouch, cereal, milk, formula, snack, drink).\n"
            "  2. Any field references baby, infant, toddler, weaning, \"4 m+\", \"6 m+\", \"stage 1/2/3\".\n"
            "• Ignore \"baby\" used purely as a cosmetic adjective for adults.\n\n"
            "COUNTERFEIT OR IP VIOLATIONS\n"
            "• Flag if wording like replica, inspired by, type, dupe, \"smells like\", \"compatible with <brand>\" "
            "appears **alongside** a well-known trade mark (e.g. Chanel No. 5, Nike, Apple). "
            "• Genuine parallel-import products with original branding are **NOT** counterfeit—only flag explicit replica/inspired cases.\n\n"
    
            "DEBUGGING GUIDANCE:\n"
            "• Always populate `debug_notes` with any partial signals that were considered but did **not** trigger "
            "a full prohibition. Examples: [\"mentions PCOS\", \"contains 'knife' but refers to butter-knife in recipe\"]\n"
            "• Keep each note short; if no partial concerns, return an empty list [].\n\n"
    
            "PRODUCT DATA:\n"
            "{{product_data}}\n"
        ),
        "recommended_model": "gpt-4o-mini",
        "description": "Prohibited-item checker with expanded medicinal-condition logic and debugging output."
    },
    "COMPLETE: Food Supplement Compliance Check": {
        "prompt": (
            "SYSTEM MESSAGE:\n"
            "You are a JSON-producing assistant for **mission-critical UK food-supplement compliance checks**. "
            "Examine ALL available data for one SKU (ingredients, descriptions, nutritionals, sell copy, warnings, "
            "directions, quantity, etc.) and:\n"
            "  1. **Classification** – Decide whether the product is a *food supplement* based on its overall context.\n"
            "  2. **If it *is* a food supplement**, confirm that the labelling text (any field) explicitly contains **ALL** mandatory UK statements:\n"
            "       – An *advised daily dose* (e.g. “Take one capsule daily”).\n"
            "       – A *“Do not exceed the recommended dose”* clause.\n"
            "       – A statement that it *“should not be used as a substitute for a varied diet.”*\n"
            "       – *“Keep out of reach of young children.”*\n"
            "       – Any **additional legally required qualifiers**, including but not limited to:\n"
            "            • “Consult your doctor if pregnant or breastfeeding” (especially if vitamin A, iron, or other high-risk nutrients are present).\n"
            "            • Warnings for children, allergens, or high-dose nutrients (e.g. vitamin D > 25 µg, zinc > 25 mg).\n"
            "  3. **If it is *not* a food supplement**, report that and **skip** the compliance check.\n\n"
            "Important constraints:\n"
            "  • **No assumptions** – treat any ambiguous or absent statement as *missing*.\n"
            "  • **Independently verify** each mandatory element; only mark *overall* as \"Pass\" when **every** required statement is clearly present.\n"
            "  • The **justification** field must *only* explain the *classification* decision (why it is / isn’t a supplement) – not the compliance outcome.\n\n"
            "Return **JSON ONLY** in exactly this format:\n"
            "{\n"
            "    \"is_food_supplement\": true | false,\n"
            "    \"justification\": \"<brief rationale for classification>\",\n"
            "    \"compliance_check\": {\n"
            "        \"overall\": \"Pass\" | \"Fail\",          // include only when is_food_supplement is true\n"
            "        \"missing_elements\": [                   // list each absent or incorrect mandatory statement\n"
            "            \"<element_name>\",\n"
            "            ...\n"
            "        ]\n"
            "    }\n"
            "}\n\n"
            "If \"is_food_supplement\" is false, still return the field but **omit** the entire \"compliance_check\" object.\n"
            "No extra commentary, markdown, or disclaimers – **strict JSON output only**.\n\n"
            "USER MESSAGE:\n"
            "Here is all available data for one SKU (field names may vary):\n"
            "{{all_product_fields}}\n"
        ),
        "recommended_model": "gpt-4.1-mini",
        "description": "Classifies a SKU as a food supplement or not, then checks for every mandatory UK food-supplement label statement with zero reliance on specific field names."
    },
    "COMPLETE: Novel Food Checker (EU)": {
        "prompt": "",  # We'll generate this dynamically per row
    "recommended_model": "gpt-4o-mini",
    "description": "Flags presence of authorised EU novel foods by comparing ingredient statements to the consolidated EU 2017/2470 list."
},
    "INCOMPLETE: Medicinal Language Compliance Checker": {
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
    "COMPLETE: Image: Ingredient Scrape (HTML)": {
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
    "INCOMPLETE: shelf_label_validation": {
        "prompt": (
            "SYSTEM MESSAGE:\n"
            "You are a JSON-producing assistant for high-criticality shelf-label compliance checking. "
            "For every SKU you receive, examine these fields:\n"
            "sku, product_description_en, lexmark_pack_size, sel_description, "
            "lexmark_uom, price_mult, brand_name\n"
            "Use sku_name only to help identify the product type (e.g. cosmetic) and do not use it in any validation checks.\n\n"

            "Perform ALL of the following validation rules and list every failure:\n\n"

            "1. lexmark_pack_size\n"
            "   • Must be proper-case for word-based descriptors (Capsules, Tablets, Tea Bags, etc.).  \n"
            "   • For g or ml units, numeric values with case-insensitive UOM are accepted (e.g. \"25g\", \"25G\", \"150ml\", \"150ML\").  \n"
            "   • Must be either:\n"
            "       – Exactly one descriptor from this allowed set:\n"
            "         {Bags, Candles, Caplets, Capsules, Chewables, Condoms, Cubes, Footpads, g, Gummies, Infusers,\n"
            "          Inhalators, Item, Items, Jellies, Liners, Lozenges, Melts, ml, Nuggets, Packs, Pad, Pads,\n"
            "          Pastilles, Patches, Pieces, Pillules, Plasters, Sachets, Softgels, Sticks, Strips,\n"
            "          Suppositories, Tablets, Tampons, Tea Bags, Wipes}\n"
            "       – Or a numeric value immediately followed (no space) by one of the allowed UOM entries above.\n"
            "   • “Items” is valid only when the contents are genuinely assorted / non-typical.\n"
            "   • Immediate failure if the unit is any form of kilogram or litre (kg, kilogram(s), l, litre(s), ltr).\n\n"

            "2. sel_description\n"
            "   • Must be Proper Case.\n"
            "   • Renders on two rows of 20 characters each (40 total).\n"
            "   • Split at nearest space ≤20 chars; if row 2 overflows it is auto-truncated with “…”—any truncation is a failure.\n"
            "   • Must NOT contain the brand name or the pack size (e.g. \"60 Tablets\"), but may include strength values such as \"1000 mg\" or \"15 SPF\".\n\n"

            "3. lexmark_uom (price-per unit)\n"
            "   • Must be the singular Proper-Case family unit matching the pack size (\"Per Tablet\", \"Per Capsule\", etc.).\n"
            "   • For pack sizes in g or ml, default to \"Per 100 g\" or \"Per 100 ml\" unless the product is a cosmetic under the UK Price-Marking Order 2004 (override only when certain from context).\n\n"

            "4. price_mult\n"
            "   • Must equal reference_qty ÷ pack_size_qty, where reference_qty = 100 for “Per 100 g/ml” and 1 for singular units.\n"
            "   • Accept a rounding tolerance of ±0.0001.\n\n"

            "5. brand_name\n"
            "   • Must be populated (non-blank).\n"
            "   • Brand name must NOT appear anywhere in sel_description (case-insensitive).\n\n"

            "6. Self-Validation\n"
            "   After applying rules 1–5, review each listed failure against the original data to confirm its validity. "
            "Remove any false positives so only confirmed failures remain.\n\n"

            "Return JSON ONLY in the following format—no extra keys, comments or text:\n"
            "{\n"
            "  \"overall\": \"Pass\" | \"Fail\",\n"
            "  \"failures\": [                      // empty if overall == \"Pass\"\n"
            "    {\n"
            "      \"field\": \"<field_name>\",     // lexmark_pack_size | sel_description | lexmark_uom | price_mult | brand_name\n"
            "      \"reason\": \"<concise reason>\"\n"
            "    }\n"
            "  ],\n"
            "  \"notes\": \"<optional brief note>\" // omit or leave blank if not needed\n"
            "}\n\n"

            "Do NOT output anything except the JSON.\n\n"
            "USER MESSAGE:\n"
            "Here is all available data for one SKU (fields may vary):\n"
            "{{all_product_fields}}\n"
        ),
        "recommended_model": "gpt-4.1-mini",
        "description": (
            "Validates shelf-label data (pack size, SEL proper case & length, brand presence, "
            "price-per UOM, price multiplier) with special-case handling for g/ml case and a self-validation pass."
        )
    },
    "COMPLETE: AUDIT: Spelling and Grammar Checker": {
        "prompt": (
            "SYSTEM MESSAGE:\n"
            "\"You are a JSON-producing assistant that evaluates product data for spelling and grammar issues "
            "in UK English. Use the supplied `sku_name` and `brand_name` to avoid flagging valid brand terms or SKUs as errors. "
            "Never hallucinate or assume facts; analyse ONLY the supplied data.\"\n\n"
    
            "Respond with valid JSON ONLY in exactly this shape:\n"
            "{\n"
            "  \"grammar_flag\": \"Pass\" | \"Fail\",\n"
            "  \"summary\": \"<brief human-readable summary of issues or 'No issues found'>\",\n"
            "  \"errors\": [\n"
            "    {\n"
            "      \"field\": \"<name of the field where the issue was found>\",\n"
            "      \"type\": \"Spelling\" | \"Grammar\",\n"
            "      \"text\": \"<the exact text snippet with the issue>\",\n"
            "      \"suggestion\": \"<corrected text suggestion>\"\n"
            "    },\n"
            "    …\n"
            "  ],\n"
            "  \"debug_notes\": [\"<optional list of low-confidence or style observations>\"]\n"
            "}\n\n"
    
            "RULES:\n"
            "1. Set `grammar_flag` = \"Fail\" if **any** spelling or grammar issue is detected; else \"Pass\".\n"
            "2. Only flag genuine English spelling or grammatical errors; do **not** flag:\n"
            "   • Proper nouns or trademarks in `brand_name` or `sku_name`.\n"
            "   • Alphanumeric SKUs or model numbers.\n"
            "   • Industry-specific jargon flagged as correct UK usage.\n"
            "3. Use UK conventions (e.g. colour, organise, centre) for all corrections.\n\n"
    
            "OUTPUT DETAILS:\n"
            "• `summary` should list issue count and a short overview (e.g. \"2 spelling errors and 1 grammar issue found\").\n"
            "• `errors` must include each issue’s location, type, original snippet, and a suggested correction.\n"
            "• `debug_notes` can capture borderline cases or style notes (e.g. [\"'utilise' is correct UK variant but consider 'use'\"]).\n\n"
    
            "PRODUCT DATA:\n"
            "{{product_data}}\n"
        ),
        "recommended_model": "gpt-4o-mini",
        "description": "Checks variants_description for UK-English spelling & grammar, respecting sku_name and brand_name to avoid false positives."
    },
    "INCOMPLETE: AUDIT: Free-From Claim Check": {
        "prompt": (
            "SYSTEM MESSAGE:\n"
            "You are a deterministic JSON auditor. You must follow every rule below exactly. "
            "If any rule is broken, you must output the fallback object:\n"
            "{ \"sku\": \"\", \"claim\": \"\", \"violated\": false, \"matched_term\": \"\", \"explanation\": \"\", \"debug\": \"\" }\n\n"
    
            "INPUT FIELDS (runtime):\n"
            "  sku               - string (the product identifier)\n"
            "  claim             - string (one free-from claim to check, e.g. \"Dairy Free\")\n"
            "  full_ingredients  - string (may contain HTML)\n"
            "  warning_info      - string (may contain HTML)\n\n"
    
            "STEP 1 – NORMALISE TEXT:\n"
            "  - Strip all HTML tags from full_ingredients and warning_info.\n"
            "  - Convert both strings to lowercase.\n\n"
    
            "STEP 2 – CONTEXTUAL CLAIM CHECK:\n"
            "  - You will see exactly one claim.\n"
            "  - Using your contextual understanding of that claim, decide whether any ingredient or warning phrase violates it.\n"
            "    • If you determine a violation, set violated=true, and capture:\n"
            "        matched_term = the exact word or phrase from the ingredients or warnings that triggered the violation\n"
            "        explanation  = a brief rationale (e.g. “‘milk powder’ indicates dairy”).\n"
            "        debug        = a single message describing the match (e.g. “warning_info contains 'milk powder'”).\n"
            "    • If you determine no violation, set violated=false and leave matched_term, explanation, and debug empty.\n\n"
    
            "STEP 3 – OUTPUT (strict JSON ONLY):\n"
            "{\n"
            "  \"sku\": \"<same as input>\",\n"
            "  \"claim\": \"<same as input>\",\n"
            "  \"violated\": <true|false>,\n"
            "  \"matched_term\": \"<if violated, the triggering phrase; otherwise empty>\",\n"
            "  \"explanation\": \"<brief rationale; empty if no violation>\",\n"
            "  \"debug\": \"<single debug message; empty if no violation>\"\n"
            "}\n"
            "Output nothing else."
        ),
        "recommended_model": "gpt-4.1-mini",
        "description": "Single-claim free-from auditor: contextually assesses one claim per call, returns strict JSON with a single debug string."
    },
    "COMPLETE: AUDIT: Ingredient Spelling": {
        "prompt": (
            "SYSTEM MESSAGE:\\n"
            "\"You are a JSON-producing assistant that audits the `full_ingredients` field of product data for two things: "
            "(A) clear spelling errors in ingredient names, and (B) residual **non-ingredient** text such as marketing blurbs or usage claims. "
            "Ignore boilerplate headings like ‘Ingredients:’ or ‘Full Ingredients:’ and benign serving phrases such as ‘per gummy’, ‘per gummies’, "
            "‘per tablet’, ‘per capsule’. Use only the supplied `sku`, `sku_name`, and `full_ingredients` — never hallucinate or assume facts.\"\\n\\n"

            "Respond with **valid JSON ONLY** in exactly this shape:\\n"
            "{\\n"
            "  \\\"spelling_flag\\\": \\\"Pass\\\" | \\\"Fail\\\",\\n"
            "  \\\"residual_flag\\\": \\\"Pass\\\" | \\\"Fail\\\",\\n"
            "  \\\"summary\\\": \\\"<brief human-readable overview of findings or 'No issues found'>\\\",\\n"
            "  \\\"misspellings\\\": [{ \\\"field\\\": \\\"full_ingredients\\\", \\\"text\\\": \\\"<token>\\\", \\\"suggestion\\\": \\\"<fix>\\\" }],\\n"
            "  \\\"residuals\\\": [{ \\\"field\\\": \\\"full_ingredients\\\", \\\"text\\\": \\\"<snippet>\\\", \\\"position\\\": \\\"<offset>\\\" }],\\n"
            "  \\\"debug_notes\\\": [\\\"<optional notes>\\\"]\\n"
            "}\\n\\n"

            "RULES:\\n"
            "1. **Spelling check** – flag alphabetic tokens ≥ 4 chars that a UK-English spell-checker marks as incorrect (high confidence). "
            "Skip proper nouns/trademarks from `sku_name`, hyphen-ated domain terms, or tokens with edit-distance > 1. "
            "Set `spelling_flag` = \\\"Fail\\\" if any misspelling is found; otherwise \\\"Pass\\\".\\n"
            "2. **Residual detection** – flag strings that are clearly **not ingredients** (e.g. flavour descriptors “rich chocolate flavour”, marketing claims "
            "“supports immunity”). Exclude heading labels /^(ingredients|full ingredients):?$/i and serving strings "
            "/^per (gummy|gummies|tablet|capsule|serving|dose)s?$/i. Heuristics (case-insensitive): adjectives like tasty, refreshing, premium; "
            "verbs like supports, boosts; stand-alone flavour words not followed by ‘flavouring’ or ‘extract’. "
            "**Set `residual_flag` = \\\"Fail\\\" only when at least one residual snippet is detected; otherwise set it to \\\"Pass\\\".**\\n"
            "3. The `summary` must report counts (e.g. \\\"1 misspelling, 2 residual snippets\\\" or \\\"No issues found\\\").\\n"
            "4. Omit the `misspellings` or `residuals` array if it is empty.\\n"
            "5. Output **only** the JSON object — no prose, no markdown.\\n\\n"

            "PRODUCT DATA:\\n"
            "{{product_data}}\\n"
        ),
        "recommended_model": "gpt-4o-mini",
        "description": "Audits `full_ingredients` for misspelled ingredient tokens and filters out marketing/descriptive residue, ignoring header or serving phrases."
},
    "COMPLETE: Image: Directions for Use": {
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
    "INCOMPLETE: Image: Multi-Image Ingredient Extract & Compare": {
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
    "COMPLETE: Grammar & Spelling Summary Checker": {
        "prompt": (
            "SYSTEM MESSAGE:\n"
            "\"You are a JSON-producing assistant that checks short product descriptions (sell copy) for *real* spelling, grammar, punctuation, and capitalisation issues. "
            "Always use British English conventions (e.g. 'colour', not 'color'; 'organised', not 'organized'). "
            "Focus strictly on objectively incorrect language. Do NOT suggest stylistic changes, placeholder corrections, or domain-specific tweaks. "
            "If the description is already correct, return an empty summary.\"\n\n"
    
            "**Respond ONLY with valid JSON in exactly this format:**\n"
            "{\n"
            "  \"summary\": \"<short, human-readable list of corrections made>\",\n"
            "  \"debug_notes\": [\"<optional corrections shown as (wrong->correct)\"]\n"
            "}\n\n"
    
            "**Guidelines:**\n"
            "• Write the `summary` as a brief bullet-style list (starting each item with '- ').\n"
            "• Mention the type of correction if helpful (e.g. [Spelling], [Punctuation]).\n"
            "• Return `summary: \"\"` and `debug_notes: []` if no issues found.\n"
            "• Use `debug_notes` for optional exact substitutions in the form (wrong->correct).\n\n"
    
            "**EXAMPLES**\n"
            "- If the input is perfect:\n"
            "{ \"summary\": \"\", \"debug_notes\": [] }\n\n"
            "- If there are issues:\n"
            "{\n"
            "  \"summary\": \"- [Spelling] Corrected 'recieve' to 'receive'.\\n- [Grammar] Added missing apostrophe in 'mens' to form 'men's'.\",\n"
            "  \"debug_notes\": [\"recieve->receive\", \"mens->men's\"]\n"
            "}\n\n"
    
            "PRODUCT DATA:\n"
            "{{product_data}}\n"
        ),
        "recommended_model": "gpt-4o",
        "description": "UK-English grammar, spelling, punctuation, and case checker with readable summaries and debug trace."
    },
    "COMPLETE: Image: Storage Instructions": {
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
    "COMPLETE: Product Name & Variant Extractor": {
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
    "COMPLETE: Gelatin Source Classifier": {
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
    "COMPLETE: Image: Warnings and Advisory (JSON)": {
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
    "INCOMPLETE: Price Marking Order Category": {
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
    "INCOMPLETE: Free From Quick-Check": {
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
    "INCOMPLETE: French Sell Copy Translator": {
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
    "INCOMPLETE: Product Usage Type Checker": {
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
    "INCOMPLETE: Vegan Flag Check": {
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
    "COMPLETE: Methylated Vitamin Check": {
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
    "INCOMPLETE: food_supplement_classifier": {
        "prompt": (
            "SYSTEM MESSAGE:\n"
            "You are a JSON-producing assistant that classifies whether a product SKU is a food supplement.  \n"
            "Review **all** provided product data holistically and contextually; do not restrict to specific fields.\n\n"

            "Rules:\n"
            "• Classify as a food supplement only if the text explicitly uses the phrase \"food supplement\" or \"dietary supplement\" "
            "AND includes a measured daily dose, mentions NRV/%RI, or says \"do not exceed the recommended intake\".\n"
            "• Products not meeting both criteria are not food supplements.\n\n"

            "Process:\n"
            "1. Analyze the data and make an initial decision.\n"
            "2. Perform a self-validation: re-examine the original data to confirm or revise the decision.\n"
            "3. Provide a succinct debug explanation of why you determined it is (or is not) a food supplement.\n\n"

            "Return JSON ONLY in this format:\n"
            "{\n"
            "  \"is_food_supplement\": \"Yes\" | \"No\",\n"
            "  \"confirmed\": \"Yes\" | \"No\",\n"
            "  \"debug\": \"<succinct explanation>\"\n"
            "}\n\n"

            "Do NOT output anything except the JSON.\n\n"
            "USER MESSAGE:\n"
            "{{PRODUCT_DATA}}\n"
        ),
        "recommended_model": "gpt-4.1-mini",
        "description": "Holistically classifies a product as a food supplement, confirms its decision via self-validation, and returns a concise debug reason."
    },
    "INCOMPLETE: Legal Category Classifier": {
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
    "INCOMPLETE: AUDIT: Nutritionals": {
        "prompt": (
            """SYSTEM MESSAGE:\\n
            \"You are a JSON-producing assistant that performs a **three-step nutritional audit**. "
            "Never hallucinate or assume facts — analyse ONLY the composite object passed in `{{PRODUCT_DATA}}`. "
            "Field names may vary, so parse holistically (SKU, SKU name, description, ingredients, nutritionals array, free-text, etc.).\"\\n\\n

            Respond with **valid JSON ONLY** in exactly this shape:\\n\\n
            {\\n
              \\\"nutrition_flag\\\": \\\"Pass\\\" | \\\"Fail\\\",\\n
              \\\"nrv_flag\\\":      \\\"Pass\\\" | \\\"Fail\\\",\\n
              \\\"summary\\\":       \\\"<brief human-readable overview or 'All checks passed'>\\\",\\n
              \\\"errors\\\": [\\n
                {\\n
                  \\\"field\\\":   \\\"nutritionals\\\",\\n
                  \\\"type\\\":    \\\"Missing Data\\\" | \\\"Missing NRV\\\",\\n
                  \\\"message\\\": \\\"<description of the issue>\\\"\\n
                }\\n
              ]\\n
            }\\n\\n

            ### AUDIT LOGIC (rigid three-step process) ###\\n
            **STEP 1 – Classify product** (holistic search)\\n
              • Build one lower-case string from *all* fields.\\n
              • Flags:\\n
                – *consumable?*  → true if it contains any of: vitamin, supplement, tablet, gummy, effervescent, tea, honey, powder, drink, food.\\n
                – *supplement?* → true if it also contains stricter cues: vitamin, supplement, tablet, gummy, effervescent.\\n\\n
            **STEP 2 – If supplement?**\\n
              • Locate a nutrition panel (JSON array, table, or heading “nutrition”, “nutrition facts”, “typical values”).\\n
              • If none found ⇒ add error {Missing Data} and set `nutrition_flag = Fail`.\\n
              • Else set `nutrition_flag = Pass`.\\n
              • Scan the panel text for ANY occurrence of “NRV” or “RI” (case-insensitive, with or without “%”).\\n
                – **If ZERO matches are found, you MUST set `nrv_flag = Fail` and add error {Missing NRV}.**\\n
                – Else set `nrv_flag = Pass`.\\n\\n
            **STEP 3 – If consumable but NOT supplement**\\n
              • Locate a nutrition panel as above.\\n
              • If none found ⇒ error {Missing Data} and `nutrition_flag = Fail`; otherwise `nutrition_flag = Pass`.\\n
              • Always set `nrv_flag = Pass` (NRV not required for non-supplement foods/drinks).\\n\\n
            **If not consumable at all**\\n
              • Set `nutrition_flag = Pass`, `nrv_flag = Pass`, and summary “Not a consumable product”.\\n\\n

            OUTPUT RULES:\\n
            • `summary` must be concise (e.g. “Supplement missing NRV” or “All checks passed”).\\n
            • Omit the `errors` array when there are no failures.\\n
            • Output **only** the JSON object — no extra keys, no markdown.\\n\\n

            USER MESSAGE:\\n
            Audit the following product (all fields bundled):\\n\\n
            {{PRODUCT_DATA}}\\n\\n
            Only respond with the JSON described above."""
        ),
        "recommended_model": "gpt-4o-mini",
        "description": "Three-step audit: classify, check nutrition panel, then NRV requirement for supplements."
},
    "COMPLETE: AUDIT: Allergen Bold Check": {
        "prompt": (
            "SYSTEM MESSAGE:\n"
            "You are a JSON-producing assistant. You never invent or assume allergen matches. You only report real, verified findings based on bold tag logic in an HTML-coded ingredient list. No disclaimers, no explanation — just valid JSON.\n\n"
            "Follow these rules carefully:\n\n"
            "1) You are scanning HTML-coded ingredient lists for unbolded mentions of these 14 regulated allergens:\n\n"
            "   - celery\n"
            "   - cereals containing gluten (wheat, rye, barley, oats)\n"
            "   - crustaceans\n"
            "   - eggs\n"
            "   - fish\n"
            "   - lupin\n"
            "   - milk\n"
            "   - molluscs\n"
            "   - mustard\n"
            "   - nuts (almonds, hazelnuts, walnuts, cashews, pecans, Brazil nuts, pistachios, macadamias)\n"
            "   - peanuts\n"
            "   - sesame\n"
            "   - soy (soya, soja)\n"
            "   - sulphites (SO2, sulfur dioxide)\n\n"
            "2) Only flag an allergen if BOTH of the following are true:\n"
            "   - The allergen word (or its synonym) appears fully outside any <b>, <B>, <strong>, or <STRONG> tag.\n"
            "   - The allergen is not part of a 'may contain', 'traces of', or similar precautionary statement.\n\n"
            "3) Evaluate each allergen synonym **independently**:\n"
            "   - Do not treat phrases like 'almond and hazelnut' as a single match.\n"
            "   - Check the bold status of each allergen word on its own.\n"
            "   - Use the following mappings:\n"
            "     • cereals containing gluten = wheat, rye, barley, oat, oats\n"
            "     • milk = milk powder, skimmed milk, whey (milk), casein, etc.\n"
            "     • soy = soy, soya, soja\n"
            "     • nuts = almonds, hazelnuts, walnuts, cashews, pecans, Brazil nuts, pistachios, macadamias\n"
            "     • sulphites = sulphites, SO2, sulfur dioxide\n\n"
            "4) Tag logic rules:\n"
            "   - A match is **not** a violation if the allergen word is fully wrapped in bold tags.\n"
            "   - Do NOT flag allergen matches that are surrounded by <b> or <strong> tags, even if connectors like 'and' or commas between them are unbolded.\n"
            "   - If **any part of the allergen word** lies outside bold tags (e.g., <strong>alm</strong>ond), treat it as unbolded.\n\n"
            "5) You must perform a strict two-step verification process for every potential allergen match:\n"
            "   Step 1 — Candidate Detection:\n"
            "     - Identify any word from the allergen list that may be outside of bold tags.\n"
            "   Step 2 — Self-Audit:\n"
            "     - Re-check the actual HTML to confirm the allergen is fully outside bold tags and not part of a 'may contain' disclaimer.\n"
            "     - If the allergen is even partially within bold tags, or bolded elsewhere in the sentence, exclude it.\n"
            "     - Do not flag matches based on phrases — only bold tag positions.\n\n"
            "6) Return a strict JSON response in exactly the following structure:\n\n"
            "{\n"
            "  \"unbolded_allergens\": \"milk, fish, celery\",\n"
            "  \"debug_matches\": [\n"
            "    \"Confirmed unbolded 'milk': found 'milk powder' outside tags in: '<p>milk powder, sugar, cocoa</p>'\",\n"
            "    \"Confirmed unbolded 'wheat': found 'wheat flour' outside tags in: 'wheat flour, water, salt'\"\n"
            "  ]\n"
            "}\n\n"
            "If no unbolded allergens are found:\n\n"
            "{\n"
            "  \"unbolded_allergens\": \"\",\n"
            "  \"debug_matches\": []\n"
            "}\n\n"
            "7) debug_matches requirements:\n"
            "   - Include one entry per allergen flagged.\n"
            "   - Quote the exact synonym matched (e.g. 'hazelnuts', 'milk powder').\n"
            "   - Show the relevant HTML snippet where it appeared.\n"
            "   - Confirm the result of the self-audit. Examples:\n"
            "     • \"Confirmed unbolded 'wheat': found outside tags in: 'wheat flour, salt'\"\n"
            "     • \"Excluded 'almond': fully bolded in: '<strong>Almond</strong> and <strong>Hazelnut</strong>'\"\n"
            "     • \"Excluded 'hazelnuts': wrapped in <strong> tag inside: '[sugar, <strong>hazelnuts</strong>]'\n\n"
            "8) Final validation check (mandatory before returning results):\n"
            "   - Every allergen in \"unbolded_allergens\" MUST match one of the 14 allergen categories or their approved synonyms (as defined in Rule 3).\n"
            "   - If a term does not map to a known allergen group, it must NOT appear in the result.\n"
            "   - Each allergen listed must also have a corresponding entry in \"debug_matches\".\n"
            "   - If any allergen is missing a debug trace, exclude it from the output.\n"
        ),
        "recommended_model": "gpt-4.1-mini",
        "description": "Checks HTML-coded ingredients for unbolded allergens using strict two-step logic with verified debug outputs."
    },
    "COMPLETE: GHS Pictogram Detector": {
        "prompt": (
            "SYSTEM MESSAGE:\n"
            "You are an image‑analysis assistant that identifies GB CLP / GHS hazard pictograms on retail packaging.\n"
            "✔️ Allowed pictograms (exact spelling only):\n"
            "    • Explosive              (exploding bomb)\n"
            "    • Flammable              (flame)\n"
            "    • Oxidising              (flame over circle)\n"
            "    • Corrosive              (corrosion)\n"
            "    • Acute toxicity         (skull and crossbones)\n"
            "    • Hazardous to the environment (environment)\n"
            "    • Health hazard          (exclamation mark)\n"
            "    • Serious health hazard  (silhouette with star‑burst chest)\n"
            "    • Gas under pressure     (gas cylinder)\n"
            "Rules you MUST follow:\n"
            "  • Look for each symbol’s distinctive inner graphic **and** the red diamond border; ignore icons without both features.\n"
            "  • If the same pictogram appears more than once, list it only once.\n"
            "  • If no pictogram is confidently visible, return an empty string in `pictograms`.\n"
            "  • Before answering, run an internal self‑check to ensure detections are correct (e.g. no mis‑identifying nutrition or recycling logos).\n"
            "\n"
            "Respond **only** with valid JSON exactly matching this schema (no markdown, no extra keys):\n"
            "{\n"
            "  \"pictograms\": \"<comma‑delimited list in the order above, or empty string>\",\n"
            "  \"debug_notes\": \"<short explanation of why each icon was or was not flagged>\"\n"
            "}\n"
            "\n"
            "USER MESSAGE:\n"
            "{{PRODUCT_DATA}}"
        ),
        "recommended_model": "gpt-4o",
        "description": (
            "Analyses up to several product‑image URLs for GB CLP / GHS hazard pictograms and returns a comma‑"
            "separated list of any icons found, together with concise debug notes explaining each decision."
        )
    },
    "Custom": {
        "prompt": "",
        "recommended_model": "gpt-4.1-mini",
        "description": "Write your own prompt below."
    }
}
