import streamlit as st

LOGO_URL = (
    "https://cdn.freelogovectors.net/wp-content/uploads/2023/04/"
    "holland_and_barrett_logo-freelogovectors.net_.png"
)

SIDEBAR_MD = """
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


def render_sidebar():
    """
    Render the application sidebar with logo and description.
    """
    st.sidebar.markdown("# Flexible AI Checker")
    st.sidebar.image(LOGO_URL, use_container_width=True)
    st.sidebar.markdown(SIDEBAR_MD, unsafe_allow_html=True)
