# prompts/ingredient_presence_fast.py
from __future__ import annotations
import re, unicodedata
from typing import Dict, List, Tuple

# ===== 1) Synonyms dictionary (canonical_key -> list of synonyms) =====
# Paste your full dict here (exactly as in your prompt).
TARGET_SYNONYMS: Dict[str, List[str]] = {
    "turmeric": ["turmeric", "curcuma longa", "turmeric root", "turmeric powder"],
    "turmeric_extract": ["turmeric extract", "curcuma longa extract", "turmeric root extract", "turmeric oleoresin"],
    "curcumin": ["curcumin", "curcuminoid", "curcuminoids"],
    "co_q10": ["coq10","co-q10","co q10","co q 10","coenzyme q10","co-enzyme q10","q10","ubiquinone","ubiquinone-10","ubidecarenone"],
    "milk_thistle": ["milk thistle","silybum marianum","silymarin","silybin"],
    "bovine_collagen": ["bovine collagen","hydrolysed bovine collagen","hydrolyzed bovine collagen","bovine type i collagen","bovine type iii collagen","beef collagen"],
    "marine_collagen": ["marine collagen","fish collagen","type i fish collagen","collagen from fish","cod collagen","tilapia collagen"],
    "chicken_collagen_uc_ii": ["uc-ii","uc ii","ucii","undenatured type ii collagen","chicken collagen","chicken sternum cartilage","type ii collagen (undenatured)"],
    "apple_cider_vinegar": ["apple cider vinegar","acv","cider vinegar","apple vinegar"],
    "ashwagandha": ["ashwagandha","withania somnifera","ksm-66","sensoril"],
    "omega_3": ["omega-3","omega 3","ω-3","n-3","epa","dha","fish oil","cod liver oil","omega3"],
    "omega_3_children": ["children's omega-3","kids omega-3","junior omega 3","child omega 3"],
    "omega_3_vegan": ["vegan omega-3","algal oil","algae oil","microalgae oil","schizochytrium","schizochytrium sp.","algal dha","algal epa"],
    "ginkgo_biloba": ["ginkgo biloba","gingko biloba","ginkgo","gingko"],
    "glucosamine": ["glucosamine","glucosamine sulfate","glucosamine sulphate","glucosamine hcl","n-acetyl-d-glucosamine","nag"],
    "maca": ["maca","lepidium meyenii","peruvian ginseng"],
    "cranberry_extract": ["cranberry","cranberry extract","vaccinium macrocarpon"],
    "chondroitin": ["chondroitin","chondroitin sulfate","chondroitin sulphate"],
    "blueberry": ["blueberry","blueberries","vaccinium corymbosum"],
    "bilberry": ["bilberry","bilberries","vaccinium myrtillus"],
    "tribulus_terrestris": ["tribulus terrestris","tribulus","puncture vine","puncturevine"],
    "rose_hips": ["rose hips","rosehip","rosa canina","rosehip extract"],
    "black_seed_oil": ["black seed oil","nigella sativa","black cumin seed oil","black cumin oil","kalonji oil"],
    "pumpkin_seed_oil": ["pumpkin seed oil","cucurbita pepo seed oil","pepita oil"],
    "korean_ginseng": ["korean ginseng","panax ginseng","red ginseng","asian ginseng"],
    "american_ginseng": ["american ginseng","panax quinquefolius"],
    "siberian_ginseng": ["siberian ginseng","eleutherococcus senticosus","eleuthero"],
    "beetroot_extract": ["beetroot","beet root","beetroot extract","beta vulgaris"],
    "hyaluronic_acid": ["hyaluronic acid","sodium hyaluronate","hyaluronan"],
    "flaxseed_oil": ["flaxseed oil","linseed oil","linseed","flax oil"],
    "soya_isoflavones": ["soya isoflavones","soy isoflavones","isoflavones (soy)","genistein","daidzein"],
    "primrose_oil": ["evening primrose oil","primrose oil","oenothera biennis oil","epo"],
    "starflower_oil": ["starflower oil","borage oil","borago officinalis seed oil","gla from borage"],
    "inulin": ["inulin","oligofructose","fructooligosaccharides","fos","chicory root fiber","chicory inulin"],
    "aloe_vera": ["aloe vera","aloe barbadensis","aloe gel","aloe juice"],
    "ginger_root": ["ginger","ginger root","zingiber officinale"],
    "echinacea": ["echinacea","echiflu","echinacea purpurea","echinacea angustifolia"],
    "artichoke_extract": ["artichoke extract","cynara scolymus","artichoke leaf extract"],
    "chlorella": ["chlorella","chlorella pyrenoidosa","chlorella vulgaris","chinese chlorella"],
    "acai_berry": ["acai","açai","euterpe oleracea","acai berry"],
    "cinnamon": ["cinnamon","cinnamomum verum","cinnamomum zeylanicum","cassia"],
    "alpha_lipoic_acid": ["alpha lipoic acid","α-lipoic acid","lipoic acid","thioctic acid","ala (lipoic)"],
    "bee_pollen": ["bee pollen","pollen (bee)"],
    "sea_kelp": ["sea kelp","kelp","ascophyllum nodosum","laminaria","kombu"],
    "black_garlic": ["black garlic"],
    "aged_garlic": ["aged garlic","aged garlic extract","kyolic"],
    "garlic_extract": ["garlic extract","allium sativum extract"],
    "pycnogenol": ["pycnogenol","french maritime pine bark extract","pinus pinaster bark extract"],
    "glucomannan": ["glucomannan","konjac","amorphophallus konjac"],
    "chromium": ["chromium","chromium picolinate","chromium chloride","chromium polynicotinate","chromium(iii)"],
    "rutin": ["rutin","rutoside","quercetin-3-rutinoside"],
    "lycopene": ["lycopene","tomato extract (lycopene)"],
    "matcha_tea": ["matcha","matcha tea","green tea powder"],
    "guarana": ["guarana","paullinia cupana"],
    "spirulina": ["spirulina","arthrospira platensis"],
    "boswellia": ["boswellia","boswellia serrata","indian frankincense","akba"],
    "horny_goat_weed": ["horny goat weed","epimedium","icariin"],
    "soy_lecithin": ["soy lecithin","soya lecithin","lecithin (soya)","lecithins (soy)","lecithin - soya"],
    "pomegranate_extract": ["pomegranate","pomegranate extract","punica granatum"],
    "resveratrol": ["resveratrol","trans-resveratrol","polygonum cuspidatum","japanese knotweed"],
    "red_clover_extract": ["red clover","red clover extract","trifolium pratense"],
    "fenugreek": ["fenugreek","trigonella foenum-graecum","methi"],
    "activated_charcoal": ["activated charcoal","activated carbon","vegetable carbon (e153)"],
    "lutein": ["lutein","marigold extract","tagetes erecta (lutein)"],
    "saw_palmetto": ["saw palmetto","serenoa repens"],
    "five_htp": ["5-htp","5 htp","5htp","5-htp","5 hydroxytryptophan","5-hydroxytryptophan","griffonia simplicifolia"],
    "saffron": ["saffron","crocus sativus","safranal","saffron extract"],
    "psyllium_husk": ["psyllium","psyllium husk","ispaghula husk","plantago ovata"],
    "shilajit": ["shilajit","mumijo","mumiyo","asphaltum","mineral pitch"],
    "lactase_enzyme": ["lactase","lactase enzyme","β-galactosidase","beta-galactosidase"],
    "valerian": ["valerian","valeriana officinalis","valerian root"],
    "plant_sterols": ["plant sterols","phytosterols","beta-sitosterol","sitosterol","campesterol","stigmasterol","plant stanols","phytostanols"],
    "black_cohosh": ["black cohosh","cimicifuga racemosa","actaea racemosa"],
    "grapeseed_extract": ["grapeseed extract","grape seed extract","vitis vinifera seed extract","opc"],
    "bromelain": ["bromelain","bromelin","bromelains"],
    "oregano_oil": ["oregano oil","origanum vulgare oil","carvacrol"],
    "manuka_honey": ["manuka honey","leptospermum scoparium","mgo","umf"],
    "honey": ["honey","acacia honey","forest honey"],
    "msm": ["msm","methylsulfonylmethane","dimethyl sulfone","dmso2"],
    "quercetin": ["quercetin","quercetin dihydrate","quercetin anhydrous"],
}

# ===== 2) Normalisation + pattern helpers =====

GREEK_MAP = {"ω": "omega", "α": "alpha", "β": "beta"}

def normalise_text(s: str) -> str:
    if not s: return ""
    s = s.casefold()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    for g, latin in GREEK_MAP.items():
        s = s.replace(g, latin)
    s = s.replace("–","-").replace("—","-").replace("−","-")
    s = s.replace("®","").replace("™","")
    return re.sub(r"\s+", " ", s).strip()

def term_to_pattern(term: str) -> str:
    raw = normalise_text(term)
    esc = re.escape(raw).replace(r"\ ", r"[-\s]*").replace(r"\-", r"[-\s]*")
    start_b = r"\b" if raw and raw[0].isalnum() else ""
    end_b   = r"\b" if raw and raw[-1].isalnum() else ""
    return f"{start_b}{esc}{end_b}"

def compile_patterns(syns: Dict[str, List[str]]) -> Dict[str, List[re.Pattern]]:
    out = {}
    for key, lst in syns.items():
        pats = []
        for t in lst:
            # 'ALA' is handled by a special rule; skip if someone added it.
            if key == "alpha_lipoic_acid" and normalise_text(t) == "ala":
                continue
            pats.append(re.compile(term_to_pattern(t), re.IGNORECASE))
        out[key] = pats
    return out

COMPILED = compile_patterns(TARGET_SYNONYMS)

# ===== 3) Core scan =====

def _collect_text(product_data: dict) -> str:
    # Join all fields you pass in – we’ll normalise soon after.
    parts = []
    for v in product_data.values():
        if v is None: continue
        parts.append(str(v))
    return normalise_text(" | ".join(parts))

def _window_has(text: str, start: int, end: int, words: List[str], radius: int = 60) -> bool:
    lo = max(0, start - radius); hi = min(len(text), end + radius)
    win = text[lo:hi]
    return any(re.search(term_to_pattern(w), win, re.I) for w in words)

def scan_product(product_data: dict) -> dict:
    text = _collect_text(product_data)

    presence = {k: False for k in TARGET_SYNONYMS}
    debug    = {k: []    for k in TARGET_SYNONYMS}

    omega3_hits: List[Tuple[int,int,str]] = []
    lecithin_hits: List[Tuple[int,int]]   = []

    # 1) Generic pattern matches
    for key, pats in COMPILED.items():
        for pat in pats:
            for m in pat.finditer(text):
                presence[key] = True
                debug[key].append(m.group(0))
                if key == "omega_3":
                    omega3_hits.append((m.start(), m.end(), m.group(0)))
                if key == "soy_lecithin" and "lecithin" in m.group(0):
                    lecithin_hits.append((m.start(), m.end()))

    # 2) Special rules
    # 2a) ALA must be near 'lipoic' or 'thioctic'
    if not presence["alpha_lipoic_acid"]:
        for m in re.finditer(r"\bala\b", text):
            if _window_has(text, m.start(), m.end(), ["lipoic","thioctic"]):
                presence["alpha_lipoic_acid"] = True
                debug["alpha_lipoic_acid"].append(text[m.start():m.end()])
                break

    # 2b) Soy lecithin only if soy/soya near 'lecithin'
    if presence["soy_lecithin"]:
        soy_words = ["soy","soya"]
        confirmed = False
        for s,e in lecithin_hits:
            win = text[max(0, s-40):min(len(text), e+40)]
            tokens = re.findall(r"[a-z0-9]+", win)
            if any(t in soy_words for t in tokens):
                confirmed = True
                break
        if not confirmed:
            presence["soy_lecithin"] = False
            debug["soy_lecithin"].append("retracted: lecithin found without soy/soya proximity")

    # 2c) Omega-3 children
    if presence["omega_3"]:
        child_words = ["child","children","kids","junior"]
        presence["omega_3_children"] = any(_window_has(text, s, e, child_words, radius=80) for s,e,_ in omega3_hits)
        if presence["omega_3_children"]:
            debug["omega_3_children"].append("omega-3 + child/kids/junior context")

    # 2d) Omega-3 vegan (algal/microalgae context)
    if presence["omega_3"]:
        vegan_words = ["algal","microalgae","algae","schizochytrium"]
        presence["omega_3_vegan"] = any(re.search(term_to_pattern(w), text, re.I) for w in vegan_words)
        if presence["omega_3_vegan"]:
            debug["omega_3_vegan"].append("omega-3 + algal/microalgae context")

    # 2e) Manuka implies honey (but avoid weak 'mgo/umf' alone)
    if presence["manuka_honey"]:
        if not re.search(r"\bmanuka\b", text) and not re.search(r"\bhoney\b", text):
            if any(re.search(r"\b(?:mgo|umf)\b", hit, re.I) for hit in debug["manuka_honey"]):
                presence["manuka_honey"] = False
                debug["manuka_honey"].append("retracted: weak MGO/UMF mention without manuka/honey context")
    if presence["manuka_honey"]:
        presence["honey"] = True
        debug["honey"].append("implied by manuka_honey")

    # 3) Assemble result
    candidates = [k for k, v in presence.items() if v]
    result = {
        "ingredients_presence": presence,
        "debug_matches": {k: v for k, v in debug.items() if v} or {},
        "candidates": candidates
    }
    return result
