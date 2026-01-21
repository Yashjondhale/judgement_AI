# =========================================================
# ⚖️ JUDICIARY AI – ADVANCED LEGAL FACT EXTRACTION SYSTEM
# =========================================================

import os
import re
import pandas as pd
import numpy as np
import PyPDF2
import nltk
import gradio as gr

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# 1. INITIAL SETUP (LOGIC SAME AS REFERENCE)
# ---------------------------------------------------------

nltk.download("stopwords")
STOP_WORDS = set(stopwords.words("english"))

# ⚠️ UPDATE THESE PATHS FOR YOUR SYSTEM
CSV_PATH = r"C:\Users\hp\OneDrive\Desktop\judiciary\justice\judgments.csv"
PDF_FOLDER_PATH = r"C:\Users\hp\OneDrive\Desktop\judiciary\justice\pdfs"

# ---------------------------------------------------------
# 2. TEXT PREPROCESSING (LOGIC SAME AS REFERENCE)
# ---------------------------------------------------------

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return " ".join(w for w in text.split() if w not in STOP_WORDS)

# ---------------------------------------------------------
# 3. LOAD CSV (LOGIC SAME AS REFERENCE)
# ---------------------------------------------------------

print("📥 Loading CSV...")
try:
    df = pd.read_csv(CSV_PATH)
    columns = ["case_no", "pet", "res", "pet_adv", "res_adv", "bench", "judgement_by"]
    for c in columns:
        df[c] = df[c].fillna("")

    df["case_signature"] = (
        df["case_no"] + " " +
        df["pet"] + " " +
        df["res"] + " " +
        df["bench"] + " " +
        df["judgement_by"]
    ).apply(preprocess_text)

    # ---------------------------------------------------------
    # 4. VECTORIZE (LOGIC SAME AS REFERENCE)
    # ---------------------------------------------------------
    vectorizer = TfidfVectorizer(max_features=7000, ngram_range=(1, 2))
    case_vectors = vectorizer.fit_transform(df["case_signature"])
    print(f"✅ Cases loaded: {len(df)}")
except Exception as e:
    print(f"⚠️ Error loading data: {e}")
    df = pd.DataFrame()
    case_vectors = None

# ---------------------------------------------------------
# 5. MATCH CASE (LOGIC SAME AS REFERENCE)
# ---------------------------------------------------------

def match_case(data, threshold=0.65):
    if df.empty: return None, 0.0
    
    query = " ".join([
        preprocess_text(data["case_no"][0]),
        preprocess_text(data["pet"][0]),
        preprocess_text(data["res"][0]),
        preprocess_text(data["bench"][0]),
        preprocess_text(data["judgement_by"][0]),
    ])

    vec = vectorizer.transform([query])
    scores = cosine_similarity(vec, case_vectors)[0]

    idx = np.argmax(scores)
    score = scores[idx]

    if score >= threshold:
        # Score is 0.0 to 1.0, we convert to 0-100
        return df.iloc[idx]["temp_link"], round(score * 100, 2)

    return None, 0.0

# ---------------------------------------------------------
# 6. PDF HANDLING (LOGIC SAME AS REFERENCE)
# ---------------------------------------------------------

def extract_pdf_text(path):
    try:
        reader = PyPDF2.PdfReader(path)
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    except:
        return ""

def locate_pdf(predicted_link):
    if not predicted_link: return None
    filename = os.path.basename(predicted_link).lower()
    for root, _, files in os.walk(PDF_FOLDER_PATH):
        for f in files:
            if filename in f.lower():
                return os.path.join(root, f)
    return None

# ---------------------------------------------------------
# 7. ADVANCED LEGAL FACT EXTRACTION (UPDATED FOR HTML UI)
# ---------------------------------------------------------

def extract_legal_facts(text, limit=25):
    if not text:
        return "❌ <b>System Alert:</b> Unable to extract content from PDF."

    lines = text.split("\n")
    facts = []
    start = False

    start_keys = ["facts", "brief facts", "facts of the case"]
    stop_keys = ["arguments", "analysis", "discussion", "order", "judgment"]

    for line in lines:
        low = line.lower()
        if any(k in low for k in start_keys):
            start = True
            continue
        if start and any(k in low for k in stop_keys):
            break
        if start and line.strip():
            facts.append(line.strip())
        if len(facts) >= limit:
            break

    if not facts:
        facts = lines[:limit]

    text_content = " ".join(facts)

    # Highlighting Logic (Yellow Highlights)
    highlights = {
        r"\b\d{1,2}\s(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s\d{4}\b": "📅 Date",
        r"\b(appellant|respondent|accused|petitioner|victim)\b": "👤 Party",
        r"\b(murder|rape|fraud|cheating|theft|robbery|assault|kidnapping)\b": "⚖️ Crime",
        r"\b(police station|ps|district|city|village)\b": "📍 Location"
    }

    for pat, tag in highlights.items():
        text_content = re.sub(
            pat,
            lambda m: f"<span style='background-color: #ffd700; color: #000; padding: 0 2px; font-weight: 600;'>{m.group(0)}</span>",
            text_content,
            flags=re.IGNORECASE
        )

    # 🔹 CHANGED HEADER AS REQUESTED
    # 🔹 Added HTML structuring for the "Summary" box seen in screenshot
    return f"""
    <div style="font-family: 'Segoe UI', sans-serif; color: #333;">
        <h3 style="border-bottom: 2px solid #1a365d; padding-bottom: 8px; color: #1a365d; margin-top:0;">
            ⚖️ Legal Facts from PDF
        </h3>
        <p style="text-align: justify; line-height: 1.6; font-size: 14px;">
            {text_content}
        </p>
        <br>
        <div style="background-color: #ebf8ff; padding: 15px; border-left: 5px solid #3182ce; border-radius: 4px; font-size: 13px;">
            <strong style="color: #2c5282;">🔍 AI Legal Analysis Summary:</strong>
            <ul style="margin: 5px 0 0 15px; padding: 0;">
                <li>Focused factual background extracted.</li>
                <li>Key legal entities and dates highlighted.</li>
                <li>Procedural arguments excluded for clarity.</li>
            </ul>
        </div>
    </div>
    """

# ---------------------------------------------------------
# 8. BRIDGE FUNCTION (ADAPTED FOR NEW UI)
# ---------------------------------------------------------

def match_case_ui(case_no, pet, res, pet_adv, res_adv, bench, judgement_by):
    # Prepare Data
    data = {
        "case_no": [case_no], "pet": [pet], "res": [res],
        "pet_adv": [pet_adv], "res_adv": [res_adv],
        "bench": [bench], "judgement_by": [judgement_by],
    }

    # Run Match Logic
    link, score_val = match_case(data)

    if not link:
        # FAILED MATCH
        return {
            status_output: "❌ NO MATCH FOUND",
            conf_output: {"Confidence": 0},
            dl_output: None,
            facts_output: "### ❌ No judgment matched in the database."
        }

    # SUCCESS MATCH
    pdf_path = locate_pdf(link)
    pdf_text = extract_pdf_text(pdf_path)
    facts_html = extract_legal_facts(pdf_text)
    
    # Return 0-100% confidence to the Label component
    # The 'score_val' is already multiplied by 100 in match_case()
    return {
        status_output: "✅ MATCH FOUND",
        conf_output: {"Confidence": score_val / 100}, # gr.Label expects 0.0-1.0 usually, or we pass dict for bars
        dl_output: pdf_path,
        facts_output: facts_html
    }

# ---------------------------------------------------------
# 9. PROFESSIONAL UI CONSTRUCTION
# ---------------------------------------------------------

# Custom CSS to replicate the "Card" look and colors from your screenshot
custom_css = """
.gradio-container { background-color: #f8f9fa; }
.header-area { text-align: left; margin-bottom: 20px; }
.input-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border: 1px solid #e2e8f0; }
.output-card { background: white; padding: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border: 1px solid #e2e8f0; }
.run-btn { background-color: #1a365d !important; color: white !important; }
"""

theme = gr.themes.Soft(
    primary_hue="slate",
    secondary_hue="blue",
).set(
    button_primary_background_fill="#1a365d",
    block_title_text_weight="700"
)

with gr.Blocks(theme=theme, css=custom_css, title="Judiciary AI") as demo:

    # --- Header ---
    with gr.Row(elem_classes="header-area"):
        with gr.Column():
            gr.HTML("""
            <div style="display: flex; align-items: center; gap: 15px;">
                <span style="font-size: 40px;">⚖️</span>
                <div>
                    <h1 style="margin: 0; color: #2d3748; font-size: 24px;">Judiciary AI</h1>
                    <p style="margin: 0; color: #718096; font-size: 14px;">Advanced Case Analysis & Legal Fact Extraction System</p>
                </div>
            </div>
            """)
    
    # --- Main Dashboard Layout ---
    with gr.Row():
        
        # 🟢 LEFT COLUMN: INPUTS (35% width)
        with gr.Column(scale=4):
            with gr.Group(elem_classes="input-card"):
                gr.Markdown("### 📝 Case Query Parameters\nFill in known details to find the judgment.")
                
                case_no = gr.Textbox(label="Case Number", placeholder="e.g. W.P.(C) No.-000819")
                
                with gr.Row():
                    pet = gr.Textbox(label="Petitioner", placeholder="Petitioner Name")
                    res = gr.Textbox(label="Respondent", placeholder="Respondent Name")
                
                # Accordion for optional details (matches screenshot)
                with gr.Accordion("👥 Advocate & Bench Details (Optional)", open=False):
                    pet_adv = gr.Textbox(label="Petitioner Advocate")
                    res_adv = gr.Textbox(label="Respondent Advocate")
                    bench = gr.Textbox(label="Bench Composition")
                    judgement_by = gr.Textbox(label="Judgment Delivered By")
                
                btn = gr.Button("🔍 Run Deep Analysis", elem_classes="run-btn")

        # 🔵 RIGHT COLUMN: ANALYSIS RESULTS (65% width)
        with gr.Column(scale=6):
            
            # Status & Confidence Row
            with gr.Row():
                with gr.Column(scale=1):
                    status_output = gr.Textbox(label="Analysis Status", value="WAITING...", interactive=False, text_align="center")
                with gr.Column(scale=1):
                    # gr.Label creates the "Bar" visualization for confidence
                    conf_output = gr.Label(label="AI Confidence Match", num_top_classes=1)

            # Tabbed Results
            with gr.Tabs(elem_classes="output-card"):
                with gr.TabItem("📜 Legal Facts Extraction"):
                    # Using HTML component to render the styled facts and summary box
                    facts_output = gr.HTML()
                
                with gr.TabItem("📄 Source Document"):
                    gr.Markdown("### Download Original Judgment")
                    dl_output = gr.File(label="PDF File", interactive=False)

    # --- Event Listener ---
    btn.click(
        fn=match_case_ui,
        inputs=[case_no, pet, res, pet_adv, res_adv, bench, judgement_by],
        outputs=[status_output, conf_output, dl_output, facts_output]
    )

if __name__ == "__main__":
    demo.launch()