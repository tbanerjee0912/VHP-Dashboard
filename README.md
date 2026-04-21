# VHP Sterilization Analytics Platform
## Biocon Biologics · DP MSAT Devices Team

A local demo dashboard for analysing VHP sterilization cycle data from Prince PDF reports.

---

## Setup (one-time, ~3 minutes)

### 1. Prerequisites
- Python 3.9+ installed
- Tesseract OCR (for scanned PDF parsing):
  - **Windows:** Download installer from https://github.com/UB-Mannheim/tesseract/wiki
  - **Mac:** `brew install tesseract`
  - **Linux:** `sudo apt install tesseract-ocr`
- Poppler (for PDF-to-image):
  - **Windows:** Download from https://github.com/oschwartz10612/poppler-windows/releases → extract → add `bin/` to PATH
  - **Mac:** `brew install poppler`
  - **Linux:** `sudo apt install poppler-utils`

### 2. Install Python packages
```bash
pip install -r requirements.txt
```

### 3. Set your Anthropic API key
The app uses Claude Vision to OCR scanned PDFs.

**Windows (Command Prompt):**
```cmd
set ANTHROPIC_API_KEY=sk-ant-your-key-here
```

**Mac/Linux:**
```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

Get your key from: https://console.anthropic.com

### 4. Run the app
```bash
streamlit run app.py
```
The app opens in your browser at http://localhost:8501

---

## How to use

### Upload a run
1. Go to **Upload New Run**
2. Enter a run name (e.g. `VHP-F01-Feasibility`)
3. Upload the scanned PDF from Prince
4. Click **Extract & Save Run**
5. Claude Vision reads all pages, extracts all 11 parameters, and saves the run locally

### Analyse a run
- Go to **Analyse Run**
- Select a stored run
- See summary metrics (max/min/mean per parameter)
- Select which parameters to plot
- Toggle phase shading (Dehumidification / Conditioning / Sterilization / Aeration)
- Download raw data as CSV

### Compare runs
- Go to **Compare Runs**
- Select 2–5 runs
- Select parameters
- Overlay charts show all runs side by side
- Summary table shows max/min per run per parameter

### Manage runs
- Delete or inspect any stored run

---

## Data storage
All run data is stored locally in the `vhp_runs/` folder as JSON files.
No data is sent anywhere except to the Anthropic API during OCR extraction.
The Anthropic API call is only made during upload — after that, the JSON is stored locally.

---

## Parameters tracked
| Column | Unit |
|---|---|
| Time | HH:MM:SS |
| Status (Phase) | Text |
| Chamber Temp 1 | °C |
| Chamber Temp 2 | °C |
| Vaporizer Temp 1 | °C |
| Vaporizer Temp 2 | °C |
| Jacket Temp | °C |
| Pressure | Psi |
| H2O2 Concentration | ppmV |
| Relative Humidity | % |
| Saturation | % |

---

## Notes for IT handoff
- This is a local Streamlit app (single-user, laptop demo)
- For multi-user deployment: host on internal server with `streamlit run app.py --server.port 8501 --server.address 0.0.0.0`
- For production: replace JSON file storage with a database (PostgreSQL or SQLite)
- For SAP/D2 integration: add an export module that formats data per your ERP schema
- ANTHROPIC_API_KEY should be set as an environment variable or via a secrets manager in production
