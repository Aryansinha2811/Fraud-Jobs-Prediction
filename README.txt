# FraudScan — Setup & Run Guide (Windows)
# ==========================================

## FOLDER STRUCTURE
FraudJobsProject/
├── backend/
│   ├── app.py
│   └── requirements.txt
└── frontend/
    └── index.html

Outputs are auto-saved to: C:\Users\<YourName>\Downloads\FraudJobs\Outputs\

## STEP 1 — Install Python dependencies
Open Command Prompt in the backend/ folder and run:

    pip install -r requirements.txt

If you don't have pip, install Python from https://python.org first.

## STEP 2 — Start the backend
In the backend/ folder, run:

    python app.py

You should see:
    ✅ Backend running at http://localhost:5000
    📁 Outputs will be saved to: C:\Users\...\Downloads\FraudJobs\Outputs

Keep this terminal open while using the app.

## STEP 3 — Open the frontend
Just double-click frontend/index.html to open it in your browser.
(No npm, no node, no build step needed — it works directly!)

## STEP 4 — Run the analysis
1. Click "Drop your CSV here" and select fake_job_postings.csv
2. Click "▶ Run Analysis"
3. Watch the live progress bar (takes 1–3 minutes)
4. View your full results dashboard with all 6 charts

## OUTPUT FILES
After analysis, check Downloads/FraudJobs/Outputs/ for:
- fig1_class_distribution.png
- fig2_confusion_matrices.png
- fig3_model_comparison.png
- fig4_roc_curves.png
- fig5_feature_importance.png
- fig6_shap_importance.png
- results.json   ← all metrics in JSON format

## TROUBLESHOOTING
- "Cannot connect to backend" → Make sure app.py is running first
- CORS error → Use Chrome or Edge (not Firefox in some configs)
- Slow analysis → Normal — SHAP + SMOTE take time on large datasets
- Missing column error → Make sure your CSV is the EMSCAD dataset
