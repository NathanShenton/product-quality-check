import streamlit as st

# CSS styles for the app
STYLE_CSS = """
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

/* Buttons (e.g. st.button, st.download_button) use Lime Green background and white text */
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
"""

def inject_css():
    """
    Inject custom CSS styles into the Streamlit app.
    """
    st.markdown(STYLE_CSS, unsafe_allow_html=True)
