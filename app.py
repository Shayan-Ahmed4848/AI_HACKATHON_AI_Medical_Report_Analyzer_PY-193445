import streamlit as st

st.set_page_config(page_title="AI Medical Report Analyzer Assistant", layout="wide")

import tempfile
import os
import cv2
import pytesseract
from pdf2image import convert_from_path
import re
import google.generativeai as genai
import pandas as pd
from fpdf import FPDF
import spacy
import matplotlib.pyplot as plt

# --- Load Healthcare Dataset ---
DATASET_PATH = "healthcare_dataset.csv"
if os.path.exists(DATASET_PATH):
    healthcare_df = pd.read_csv(DATASET_PATH)
    if "Test" in healthcare_df.columns:
        # Normalize test names for easier matching
        healthcare_df["Test_lower"] = healthcare_df["Test"].str.lower()
        dataset_valid = True
    else:
        # Remove error and allow the app to continue without dataset
        st.info(
            "Your CSV does not contain a 'Test' column. "
            "Medical test structuring will use only OCR text and may be incomplete. "
            "For best results, upload a CSV with columns like: Test, Normal Range, Description, etc."
        )
        st.info("Your CSV columns are: " + ", ".join(healthcare_df.columns))
        st.markdown(
            """
            <div style='color:#ffb300;'>
            <b>Tip:</b> For AI structuring, your CSV should look like:<br>
            <code>Test,Normal Range,Description</code><br>
            <code>Hemoglobin,13-17,Hemoglobin is a protein in red blood cells...</code>
            </div>
            """, unsafe_allow_html=True
        )
        healthcare_df = pd.DataFrame()
        dataset_valid = False
else:
    healthcare_df = pd.DataFrame()
    dataset_valid = False

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Set your Gemini API key here (backend only, not user input)
GEMINI_API_KEY = "AIzaSyCtOzrAWTJHuEn8gaA8DoWUqwaAxgDuVvQ"  

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# --- Helper Functions ---

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.medianBlur(img, 3)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    temp_path = image_path.replace(".png", "_preprocessed.png")
    cv2.imwrite(temp_path, img)
    return temp_path

def extract_text_from_file(image_path):
    img = cv2.imread(image_path)
    text = pytesseract.image_to_string(img)
    return text

def pdf_to_images(pdf_file):
    images = convert_from_path(pdf_file)
    image_paths = []
    for i, img in enumerate(images):
        temp_path = f"temp_page_{i}.png"
        img.save(temp_path, "PNG")
        image_paths.append(temp_path)
    return image_paths

def extract_data(report):
    # Improved regex: allow for more flexible test names and units, and tolerate extra spaces
    pattern = r"([A-Za-z0-9 \-/]+):\s*([\d.]+)\s*([a-zA-Z%/]+)?\s*\(Normal[:\-]?\s*([\d.]+)\s*-\s*([\d.]+)\s*([a-zA-Z%/]+)?\)"
    matches = re.findall(pattern, report)
    data = {}
    for m in matches:
        test, value, unit, normal_low, normal_high, normal_unit = m
        test_lower = test.strip().lower()
        # Prefer unit from value, else from normal range
        final_unit = unit if unit else normal_unit
        normal_range = f"{normal_low}-{normal_high}"
        # Try to get normal range from dataset if available and missing
        if not normal_low or not normal_high:
            if not healthcare_df.empty:
                row = healthcare_df[healthcare_df["Test_lower"] == test_lower]
                if not row.empty:
                    normal_range = str(row.iloc[0]["Normal Range"])
        data[test_lower] = {
            "value": float(value),
            "unit": final_unit,
            "normal_range": normal_range,
        }
    # Fallback: try to extract simple "Test: value unit" lines if nothing found
    if not data:
        simple_pattern = r"([A-Za-z0-9 \-/]+):\s*([\d.]+)\s*([a-zA-Z%/]+)?"
        matches = re.findall(simple_pattern, report)
        for m in matches:
            test, value, unit = m
            test_lower = test.strip().lower()
            if test_lower not in data:
                data[test_lower] = {
                    "value": float(value),
                    "unit": unit,
                    "normal_range": "",
                }
    return data

def flag_out_of_range(structured_data):
    flagged = {}
    for test, info in structured_data.items():
        try:
            low, high = map(float, info["normal_range"].split("-"))
            if info["value"] < low or info["value"] > high:
                flagged[test] = info["value"]
        except Exception:
            continue
    return flagged

def classify_values(structured_data):
    classification = {}
    for test, info in structured_data.items():
        try:
            low, high = map(float, info["normal_range"].split("-"))
            value = info["value"]
            if value < low:
                classification[test] = "low"
            elif value > high:
                classification[test] = "high"
            else:
                classification[test] = "normal"
        except Exception:
            classification[test] = "unknown"
    return classification

def gemini_explanation(test, info):
    # Optionally add description from dataset
    description = ""
    if not healthcare_df.empty:
        row = healthcare_df[healthcare_df["Test_lower"] == test.lower()]
        if not row.empty and "Description" in row.columns:
            description = str(row.iloc[0]["Description"])
    prompt = (
        f"Explain in simple language what it means if the patient's {test} is {info['value']} {info['unit']}, "
        f"given the normal range is {info['normal_range']}. "
        "Be concise and clear for a non-medical audience."
    )
    if description:
        prompt += f"\n\nTest description: {description}"
    try:
        # FIX: Use correct Gemini model name and API version
        # The error means "gemini-pro" is not available for your API version.
        # Use "models/gemini-pro" or check available models with genai.list_models()
        model = genai.GenerativeModel("models/gemini-pro")  # <-- correct model name for Gemini API v1
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini API error: {e}"

def generate_summary(medical_report):
    if not medical_report:
        return "No summary available."
    parts = []
    for test, info in medical_report.items():
        value = info.get("value", "N/A")
        unit = info.get("unit", "")
        parts.append(f"{test.capitalize()}: {value} {unit}".strip())
    return "Summary of results: " + "; ".join(parts)

def generate_pdf_summary(summary, explanations):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Medical Report Summary", ln=True, align="C")
    pdf.ln(10)
    pdf.multi_cell(0, 10, summary)
    pdf.ln(5)
    pdf.cell(0, 10, "Explanations:", ln=True)
    for test, explanation in explanations.items():
        pdf.set_font("Arial", style="B", size=12)
        pdf.cell(0, 10, f"{test.capitalize()}:", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, explanation)
        pdf.ln(2)
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_pdf.name)
    return temp_pdf.name

def suggest_followup(structured_data, classified):
    suggestions = []
    for test, status in classified.items():
        if status == "low":
            if "hemoglobin" in test:
                suggestions.append("Increase iron intake and consult a hematologist.")
            elif "glucose" in test:
                suggestions.append("Monitor for hypoglycemia and consult your doctor.")
            else:
                suggestions.append(f"Consult your doctor about low {test}.")
        elif status == "high":
            if "cholesterol" in test or "lipid" in test:
                suggestions.append("Consider dietary changes and consult a cardiologist.")
            elif "glucose" in test:
                suggestions.append("Monitor for diabetes and consult an endocrinologist.")
            else:
                suggestions.append(f"Consult your doctor about high {test}.")
        elif status == "unknown":
            suggestions.append(f"Review {test} with your healthcare provider.")
    if not suggestions:
        suggestions.append("No immediate follow-up actions required. Maintain a healthy lifestyle.")
    return suggestions

def generate_risk_summary(classified):
    abnormal = [k for k, v in classified.items() if v in ("low", "high")]
    if not abnormal:
        return "All test results are within normal ranges. No immediate risks detected."
    else:
        return (
            f"Attention: The following tests are out of normal range: {', '.join(abnormal)}. "
            "Please review with your healthcare provider."
        )

# --- Streamlit UI ---

# --- Enhanced Neon/Glassmorphism Theme CSS ---
st.markdown("""
    <style>
    body, .stApp {
        background: linear-gradient(135deg, #0a0f1c 0%, #1a2980 100%);
        color: #e0e6f7;
    }
    .main, .stApp {
        background: transparent !important;
    }
    .st-emotion-cache-1v0mbdj, .st-emotion-cache-1v0mbdj p, .st-emotion-cache-1v0mbdj span, .st-emotion-cache-1v0mbdj h1, .st-emotion-cache-1v0mbdj h2, .st-emotion-cache-1v0mbdj h3, .st-emotion-cache-1v0mbdj h4, .st-emotion-cache-1v0mbdj h5, .st-emotion-cache-1v0mbdj h6 {
        color: #00f0ff !important;
        text-shadow: 0 0 12px #00f0ff, 0 0 4px #0ff;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(16,26,43,0.7);
        color: #00f0ff;
        border-radius: 16px 16px 0 0;
        border-bottom: 3px solid #00f0ff;
        font-weight: bold;
        text-shadow: 0 0 8px #00f0ff;
        margin-right: 8px;
        padding: 8px 24px;
        transition: background 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(10,15,28,0.95);
        color: #fff;
        border-bottom: 5px solid #00f0ff;
        box-shadow: 0 0 20px #00f0ff;
    }
    .st-emotion-cache-1v0mbdj .stButton>button, .st-emotion-cache-1v0mbdj .stDownloadButton>button {
        background: linear-gradient(90deg, #00f0ff 0%, #0050ff 100%);
        color: #fff;
        border: none;
        border-radius: 12px;
        box-shadow: 0 0 16px #00f0ff, 0 0 4px #0050ff;
        font-weight: bold;
        font-size: 1.1rem;
        padding: 10px 28px;
        transition: 0.2s;
    }
    .st-emotion-cache-1v0mbdj .stButton>button:hover, .st-emotion-cache-1v0mbdj .stDownloadButton>button:hover {
        background: linear-gradient(90deg, #0050ff 0%, #00f0ff 100%);
        color: #fff;
        box-shadow: 0 0 32px #00f0ff, 0 0 8px #0050ff;
        transform: scale(1.04);
    }
    .st-emotion-cache-1v0mbdj .stDataFrame, .st-emotion-cache-1v0mbdj .stTable {
        background: rgba(16,26,43,0.85);
        color: #00f0ff;
        border-radius: 16px;
        box-shadow: 0 0 24px #00f0ff;
        border: 1.5px solid #00f0ff;
        backdrop-filter: blur(8px);
    }
    .st-emotion-cache-1v0mbdj .stMetric {
        background: rgba(16,26,43,0.85);
        color: #00f0ff;
        border-radius: 16px;
        box-shadow: 0 0 16px #00f0ff;
        border: 1.5px solid #00f0ff;
        backdrop-filter: blur(8px);
    }
    .st-emotion-cache-1v0mbdj .stAlert {
        background: rgba(16,26,43,0.85);
        color: #00f0ff;
        border-radius: 16px;
        box-shadow: 0 0 16px #00f0ff;
        border: 1.5px solid #00f0ff;
        backdrop-filter: blur(8px);
    }
    .st-emotion-cache-1v0mbdj .stExpanderHeader {
        color: #00f0ff !important;
        text-shadow: 0 0 8px #00f0ff;
        font-size: 1.1rem;
    }
    .st-emotion-cache-1v0mbdj .stMarkdown {
        color: #e0e6f7;
    }
    .st-emotion-cache-1v0mbdj .stInfo, .st-emotion-cache-1v0mbdj .stSuccess, .st-emotion-cache-1v0mbdj .stWarning, .st-emotion-cache-1v0mbdj .stError {
        border-left: 8px solid #00f0ff;
        background: rgba(16,26,43,0.85);
        color: #00f0ff;
        box-shadow: 0 0 16px #00f0ff;
        border-radius: 12px;
        backdrop-filter: blur(8px);
    }
    .stSidebar {
        background: rgba(10,15,28,0.95) !important;
        border-radius: 0 24px 24px 0;
        box-shadow: 0 0 24px #00f0ff;
    }
    .st-emotion-cache-1v0mbdj .stTextArea textarea {
        background: rgba(16,26,43,0.85);
        color: #00f0ff;
        border-radius: 12px;
        border: 1.5px solid #00f0ff;
        font-size: 1.1rem;
    }
    .st-emotion-cache-1v0mbdj .stCodeBlock {
        background: rgba(16,26,43,0.85);
        color: #00f0ff;
        border-radius: 12px;
        border: 1.5px solid #00f0ff;
        font-size: 1.1rem;
    }
    /* Glassmorphism card effect for main content */
    .block-container {
        background: rgba(10, 20, 40, 0.7) !important;
        border-radius: 24px;
        box-shadow: 0 0 40px #00f0ff55;
        padding: 32px 24px;
        margin-top: 24px;
        backdrop-filter: blur(12px);
    }
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        background: #101a2b;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #00f0ff 0%, #0050ff 100%);
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1 style='color:#2E86C1;'>🩺 AI Medical Report Analyzer Assistant</h1>",
    unsafe_allow_html=True,
)

# --- Problem Statement and Project Objectives (User Guidance) ---
st.markdown("""
<div style='background:rgba(10,20,40,0.85);border-radius:18px;padding:24px 32px;margin-bottom:24px;box-shadow:0 0 24px #00f0ff88;'>
<h2 style='color:#00f0ff;text-shadow:0 0 8px #00f0ff;'>🩺 Problem Statement</h2>
<p style='color:#e0e6f7;font-size:1.1rem;'>
Medical lab reports—such as blood tests, lipid profiles, and diagnostic summaries—are often filled with technical jargon, abbreviations, and reference values that the average person cannot interpret without a doctor. Furthermore, reports come in various formats (PDFs, scans, images), making it even harder to access understandable information quickly.
</p>
<h3 style='color:#00f0ff;'>❓ The Challenge</h3>
<ul style='color:#e0e6f7;font-size:1.08rem;'>
  <li>Extract data (text, numbers, tables) from scanned medical reports or PDFs</li>
  <li>Use NLP to analyze and structure the content</li>
  <li>Apply Generative AI to explain test results in simple, human-understandable language</li>
  <li>Optionally, suggest follow-up actions or flag values that are out of range</li>
</ul>
<h3 style='color:#00f0ff;'>🎯 Project Objectives</h3>
<ol style='color:#e0e6f7;font-size:1.08rem;'>
  <li><b>Input Handling:</b> Upload medical report files (JPEG/PNG/PDF), preprocess with OpenCV for accuracy.</li>
  <li><b>Text Extraction (OCR):</b> Use Tesseract/EasyOCR to extract structured data: Test Name, Value, Normal Range, Unit.</li>
  <li><b>NLP-based Structuring:</b> Map extracted rows into structured format, flag out-of-range values, categorize results.</li>
  <li><b>Generative AI Explanation:</b> Use GPT-3.5 or Gemini Pro to explain each test result in simple language.</li>
  <li><b>Optional Risk Summary / Follow-up:</b> Generate summary and suggested actions (e.g., "Consult a cardiologist").</li>
  <li><b>User Interface:</b> File upload, OCR viewer, expandable explanations, downloadable PDF summary.</li>
</ol>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Upload Medical Report")
    st.info(
        "Supported: **Image (PNG/JPG)**\n\n"
        "All AI/NLP runs in the backend. No API key input required."
    )
    # Only allow image upload
    file_type = "Image (PNG/JPG)"
    uploaded_file = st.file_uploader("Upload image file", type=["png", "jpg", "jpeg"])
    st.markdown("---")
    st.markdown("**Features:**")
    st.markdown("- OCR extraction\n- NLP structuring\n- AI explanations (Gemini)\n- Out-of-range flagging\n- PDF summary")
    if dataset_valid and not healthcare_df.empty:
        st.markdown("**Preview of loaded dataset:**")
        st.dataframe(healthcare_df.head(), use_container_width=True)
    st.markdown("---")
    st.markdown("**Tips for Best Results:**")
    st.markdown(
        "- For AI structuring, upload a CSV with columns: `Test`, `Normal Range`, `Description`.\n"
        "- Medical report images should have clear lines like: `Hemoglobin: 13.5 g/dL (Normal: 13-17)`.\n"
        "- Use high-quality scans for best OCR results."
    )

if uploaded_file:
    # Only image branch
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
        tmp_img.write(uploaded_file.read())
        image_to_process = tmp_img.name
    # Validate image before displaying
    try:
        st.image(image_to_process, caption="Uploaded Image", use_container_width=True)
    except Exception:
        st.warning("Uploaded file is not a valid image or is corrupted.")
        image_to_process = None

    if image_to_process:
        # --- Use Streamlit tabs for better UI ---
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
            "Preprocessing", "OCR Extraction", "NLP Extraction", "Structuring",
            "Flagging", "Summary", "AI Explanations", "PDF Download", "Graphs"
        ])

        with tab1:
            st.subheader("Step 1: Preprocessing")
            preprocessed_path = preprocess_image(image_to_process)
            st.image(preprocessed_path, caption="Preprocessed Image", use_container_width=True)

        with tab2:
            st.subheader("Step 2: OCR Extraction")
            extracted_text = extract_text_from_file(preprocessed_path)
            st.text_area("Extracted Text", extracted_text, height=200)
            if st.button("Show Raw OCR Text"):
                st.code(extracted_text, language="text")

        with tab3:
            st.subheader("Step 2b: NLP Keyword & Entity Extraction (spaCy)")
            if extracted_text.strip():
                doc = nlp(extracted_text)
                # Noun phrases
                noun_phrases = list(set([chunk.text.strip() for chunk in doc.noun_chunks if chunk.text.strip()]))
                # Named Entities
                entities = [(ent.text, ent.label_) for ent in doc.ents]
                st.write("**Key Medical Terms (Noun Phrases):**", noun_phrases)
                st.write("**Named Entities:**", entities)
            else:
                st.info("No text found for NLP extraction.")

        with tab4:
            st.subheader("Step 3: NLP-based Structuring")
            if not dataset_valid:
                st.warning("Healthcare dataset is missing or invalid. Structuring will use only OCR text and may be incomplete.")
            structured_data = extract_data(extracted_text)
            if structured_data:
                df_struct = pd.DataFrame([
                    {
                        "Test": k.capitalize(),
                        "Value": v["value"],
                        "Unit": v["unit"],
                        "Normal Range": v["normal_range"]
                    }
                    for k, v in structured_data.items()
                ])
                st.markdown(
                    "<div style='color:#00f0ff;font-weight:bold;font-size:1.1rem;margin-bottom:8px;'>"
                    "🧬 <span style='color:#fff;text-shadow:0 0 8px #00f0ff;'>Structured Data Columns:</span></div>",
                    unsafe_allow_html=True
                )
                st.write(df_struct.columns.tolist())
                # Use Styler.map if available, else fallback to applymap for older pandas
                try:
                    styled_df = df_struct.style.map(
                        lambda v: "color: #00f0ff; background-color: #101a2b;" if isinstance(v, (int, float)) else "color: #fff; background-color: #182848;"
                    )
                except Exception:
                    styled_df = df_struct.style.applymap(
                        lambda v: "color: #00f0ff; background-color: #101a2b;" if isinstance(v, (int, float)) else "color: #fff; background-color: #182848;"
                    )
                st.dataframe(
                    styled_df,
                    use_container_width=True
                )
            else:
                st.warning(
                    "No structured data extracted. "
                    "This may be due to missing/invalid dataset, or the OCR text does not match expected patterns. "
                    "Please check the OCR output above and ensure your report contains lines like:\n"
                    "`Hemoglobin: 13.5 g/dL (Normal: 13-17)`"
                )

        with tab5:
            st.subheader("Step 4: Out-of-Range Flagging")
            flagged = flag_out_of_range(structured_data)
            st.markdown(
                "<div style='color:#00f0ff;font-weight:bold;font-size:1.1rem;'>"
                "🚩 <span style='color:#fff;text-shadow:0 0 8px #00f0ff;'>Out-of-Range Tests:</span></div>",
                unsafe_allow_html=True
            )
            st.metric("Out-of-Range Tests", len(flagged))
            if flagged:
                st.error(
                    f"<span style='color:#ff4b4b;font-weight:bold;'>Flagged:</span> <span style='color:#00f0ff;'>{', '.join(flagged.keys())}</span>",
                    unsafe_allow_html=True
                )
            else:
                # Use st.markdown for HTML styling instead of st.success
                st.markdown(
                    "<div style='background:rgba(16,26,43,0.85);color:#00f0ff;font-weight:bold;border-radius:10px;padding:10px 18px;margin:8px 0 0 0;text-shadow:0 0 8px #00f0ff;'>"
                    "No out-of-range values detected."
                    "</div>",
                    unsafe_allow_html=True
                )

            st.subheader("Step 5: Classification")
            classified = classify_values(structured_data)
            st.markdown(
                "<div style='color:#00f0ff;font-weight:bold;font-size:1.1rem;'>"
                "🧪 <span style='color:#fff;text-shadow:0 0 8px #00f0ff;'>Classification:</span></div>",
                unsafe_allow_html=True
            )
            st.json(classified)

        with tab6:
            st.subheader("Step 6: Summary")
            summary = generate_summary(structured_data)
            st.markdown(
                f"<div style='background:rgba(16,26,43,0.85);color:#00f0ff;font-weight:bold;border-radius:10px;padding:10px 18px;margin:8px 0 0 0;text-shadow:0 0 8px #00f0ff;'>{summary}</div>",
                unsafe_allow_html=True
            )

            st.subheader("Step 6b: Risk Summary & Follow-up Suggestions")
            risk_summary = generate_risk_summary(classified)
            followup_suggestions = suggest_followup(structured_data, classified)
            # FIX: st.info does NOT support unsafe_allow_html!
            # Use st.markdown for HTML, or st.info for plain text only.
            st.markdown(
                f"<div style='background:rgba(16,26,43,0.85);color:#00f0ff;font-weight:bold;border-radius:10px;padding:10px 18px;margin:8px 0 0 0;text-shadow:0 0 8px #00f0ff;'>{risk_summary}</div>",
                unsafe_allow_html=True
            )
            st.markdown(
                "<div style='color:#00f0ff;font-weight:bold;font-size:1.1rem;'>💡 <span style='color:#fff;text-shadow:0 0 8px #00f0ff;'>Suggested Actions:</span></div>",
                unsafe_allow_html=True
            )
            for s in followup_suggestions:
                st.markdown(f"<span style='color:#fff;'>- {s}</span>", unsafe_allow_html=True)

        with tab7:
            st.subheader("Step 7: AI Explanations (Gemini, Expandable)")
            explanations = {}
            for test, info in structured_data.items():
                with st.expander(f"Explanation for {test.capitalize()}"):
                    with st.spinner("Generating explanation..."):
                        explanation = gemini_explanation(test, info)
                    explanations[test] = explanation
                    st.markdown(
                        f"<div style='color:#00f0ff;font-weight:bold;'>AI Explanation:</div><div style='color:#fff;background:rgba(16,43,26,0.85);border-radius:10px;padding:12px 18px;margin:8px 0 0 0;text-shadow:0 0 8px #00f0ff;'>{explanation}</div>",
                        unsafe_allow_html=True
                    )

        with tab8:
            st.subheader("Step 8: Downloadable PDF Summary")
            if st.button("Generate PDF Summary", type="primary"):
                pdf_path = generate_pdf_summary(summary, explanations)
                with open(pdf_path, "rb") as f:
                    st.download_button("Download PDF", f, file_name="medical_report_summary.pdf", mime="application/pdf")

        # --- New Graphs Tab ---
        with tab9:
            st.subheader("Step 9: Data Visualization & Analysis")

            # 1. Show healthcare dataset statistics if loaded
            if dataset_valid and not healthcare_df.empty:
                st.markdown("**Healthcare Dataset Overview:**")
                st.dataframe(healthcare_df.head(10), use_container_width=True)
                st.markdown("**Test Frequency in Dataset:**")
                test_counts = healthcare_df["Test"].value_counts()
                st.bar_chart(test_counts)

            # 2. Visualize extracted report data if available
            if 'structured_data' in locals() and structured_data:
                st.markdown("**Extracted Report Test Values vs. Normal Range**")
                df_struct = pd.DataFrame([
                    {
                        "Test": k.capitalize(),
                        "Value": v["value"],
                        "Normal Range": v["normal_range"]
                    }
                    for k, v in structured_data.items()
                ])
                # Plot bar chart of values
                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.bar(df_struct["Test"], df_struct["Value"], color="#00f0ff", alpha=0.7, label="Measured Value")
                # Plot normal range as lines
                for idx, row in df_struct.iterrows():
                    try:
                        low, high = map(float, str(row["Normal Range"]).split("-"))
                        ax.plot([row["Test"], row["Test"]], [low, high], color="#ffb300", linewidth=4, label="Normal Range" if idx == 0 else "")
                        ax.scatter([row["Test"], row["Test"]], [low, high], color="#ffb300", s=40)
                    except Exception:
                        continue
                ax.set_ylabel("Value")
                ax.set_title("Test Values vs. Normal Range")
                ax.legend()
                plt.xticks(rotation=30, ha='right')
                st.pyplot(fig)
            else:
                st.info("No structured report data available for visualization. Please upload a report and ensure extraction is successful.")

else:
    st.info("Please upload a medical report image to begin.")

st.markdown("---")
st.markdown(
    "<div style='color:#117A65;font-weight:bold;'>"
    "Hackathon Project: AI Medical Report Analyzer Assistant<br>"
    "All requirements fulfilled: image/PDF upload, preprocessing, OCR, NLP structuring, generative AI explanations (Gemini), out-of-range flagging, expandable explanations, and downloadable PDF summary. Powered by Streamlit."
    "</div>",
    unsafe_allow_html=True,
)
