# AI Medical Report Analyzer Assistant

## Hackathon Details
- **Event:** AI Medical Report Analyzer Assistant Hackathon
- **Dates:** May 24, 2025 - May 25, 2025 (1 day)
- **Host:** Zunaib
- **Category:** Artificial Intelligence and Data Science

## Description
The AI Medical Report Analyzer Assistant is an intelligent system designed to assist healthcare professionals and patients by automatically analyzing, summarizing, and interpreting medical reports. It leverages artificial intelligence and natural language processing (NLP) to extract key information from medical documents such as lab reports, radiology findings, pathology results, and discharge summaries, providing quick insights and enhancing clinical decision-making.

## Problem Statement
Medical lab reports—such as blood tests, lipid profiles, and diagnostic summaries—are often filled with technical jargon, abbreviations, and reference values that the average person cannot interpret without a doctor. Furthermore, reports come in various formats (PDFs, scans, images), making it even harder to access understandable information quickly.

## The Challenge
Design and develop an AI-powered assistant that can:
- Extract data (text, numbers, tables) from scanned medical reports or PDFs
- Use NLP to analyze and structure the content
- Apply Generative AI to explain test results in simple, human-understandable language
- Optionally, suggest follow-up actions or flag values that are out of range

## Project Objectives
1. **Input Handling**
   - Allow users to upload medical report files in image format (JPEG/PNG) or scanned PDFs.
   - Preprocess the input (denoising, binarization) using OpenCV to improve accuracy.
2. **Text Extraction (OCR)**
   - Use Tesseract or EasyOCR to extract content from reports.
   - Extract structured data like:
     - Test Name
     - Measured Value
     - Normal Range
     - Unit (mg/dL, etc.)
3. **NLP-based Structuring**
   - Use rule-based or ML-based logic to:
     - Map extracted rows into structured format (dictionary or table).
     - Identify values outside the normal reference range.
     - Categorize values (e.g., Critical, Borderline, Normal).
4. **Generative AI Explanation**
   - Use GPT-3.5 or Gemini Pro via API to explain each test result using a prompt like:
     > “Explain in simple language what it means if the patient’s Hemoglobin is 9.5 g/dL, given the normal range is 13–17 g/dL.”
   - Return explanations for each abnormal result or all if time allows.
5. **Optional Risk Summary / Follow-up Suggestion**
   - Based on extracted values and explanations, optionally generate:
     - A summary paragraph
     - A list of suggested actions like “Consult a cardiologist” or “Increase iron intake.”
6. **User Interface**
   - **Note:** This project uses a **Streamlit web interface** for all user interactions, including file upload, OCR result viewing, explanations, and downloadable summaries.
   - There is **no Flask or CLI interface**.

## Project Structure
```
ai-assistant-medical-reports
├── src
│   ├── main.py
│   ├── analysis
│   │   └── analyzer.py
│   ├── summarization
│   │   └── summarizer.py
│   ├── data
│   │   └── __init__.py
│   └── utils
│       └── helpers.py
├── requirements.txt
├── .gitignore
├── README.md
└── tests
    ├── test_analyzer.py
    └── test_summarizer.py
```

## Setup Instructions
1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd ai-assistant-medical-reports
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   # On Windows use:
   venv\Scripts\activate
   # On macOS/Linux use:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To run the application, execute the following command:
```bash
streamlit run src/main.py
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details."# AI_HACKATHON_AI_Medical_Report_Analyzer_PY-193445" 
