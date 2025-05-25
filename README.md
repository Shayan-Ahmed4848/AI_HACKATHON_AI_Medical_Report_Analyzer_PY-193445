# AI Medical Report Analyzer Assistant (ID: 193445)

## Overview

**AI Medical Report Analyzer Assistant** is a Streamlit-based application that leverages OCR, NLP, and Generative AI to help users interpret medical lab reports (such as blood tests and lipid profiles) from images or scanned documents. The app extracts, structures, and explains test results in simple language, flags abnormal values, and provides downloadable summaries.

---

## Features

- **File Upload:** Supports uploading medical report images (PNG/JPG).
- **Preprocessing:** Uses OpenCV to enhance image quality for better OCR accuracy.
- **OCR Extraction:** Extracts text from images using Tesseract.
- **NLP Structuring:** Structures extracted text into test name, value, unit, and normal range.
- **Healthcare Dataset Integration:** Optionally uses a CSV dataset for richer test info and normal ranges.
- **Out-of-Range Flagging:** Flags test values outside the normal range.
- **Classification:** Categorizes results as low, normal, or high.
- **AI Explanations:** Uses Google Gemini Pro to explain each test result in simple terms.
- **Summary & Follow-up:** Generates a summary, risk assessment, and follow-up suggestions.
- **PDF Download:** Allows users to download a PDF summary of results and explanations.
- **Data Visualization:** Visualizes test values and normal ranges with interactive charts.

---
## 🎥 Demo Video

👉 [Click here to watch the demo](Demo.mp4)
## How It Works

1. **Upload** a medical report image (PNG/JPG).
2. **Preprocessing** improves image quality for OCR.
3. **OCR** extracts raw text from the image.
4. **NLP** parses and structures the extracted data.
5. **Flagging** highlights abnormal test results.
6. **AI Explanations** provide easy-to-understand summaries for each test.
7. **Download** a PDF summary or view interactive charts.

---

## Setup & Installation

1. **Clone the repository:**
   ```sh
   git clone <(https://github.com/Shayan-Ahmed4848/AI_HACKATHON_AI_Medical_Report_Analyzer_PY-193445)>
   cd AI-Hackathon-Python-Programming
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

   **Required packages include:**
   - streamlit
   - opencv-python
   - pytesseract
   - pdf2image
   - pandas
   - fpdf
   - spacy
   - matplotlib
   - google-generativeai

3. **Download spaCy model:**
   ```sh
   python -m spacy download en_core_web_sm
   ```

4. **(Optional) Add your healthcare dataset:**
   - Place a CSV file named `healthcare_dataset.csv` in the project root.
   - Columns: `Test`, `Normal Range`, `Description`

5. **Set up Google Gemini API key:**
   - The API key is set in `app.py` as `GEMINI_API_KEY`. Replace with your own for production use.

---

## Running the App

```sh
streamlit run app.py
```

Open the provided local URL in your browser to use the app.

---

## File Structure

- `app.py` — Main Streamlit application.
- `healthcare_dataset.csv` — (Optional) Dataset for test info and normal ranges.
- `requirements.txt` — Python dependencies.

---

## Example Usage

1. Upload a clear image of your medical report.
2. Review extracted and structured data.
3. Read AI-generated explanations for each test.
4. Download a PDF summary or view data visualizations.

---

## Notes

- All AI/NLP runs on the backend; no user API key input required.
- For best results, use high-quality scans and a well-formatted healthcare dataset.
- The app is for educational and informational purposes only. Always consult a healthcare professional for medical advice


---

## License

This project is for Ai-hackathon purposes. 
(ID: PY-193445) [MUHAMMAD SHAYAN AHMED] 
[Python Programming]

---
