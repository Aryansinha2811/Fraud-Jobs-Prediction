"""
Fraud Job Detection - Flask Backend
Run: python app.py
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import os, json, base64, io, threading, time
from pathlib import Path

# ML
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, confusion_matrix, roc_auc_score, roc_curve)
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import scipy.sparse as sp
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# ── Paths ────────────────────────────────────────────────────────────────────
DOWNLOADS = Path.home() / "Downloads"
OUTPUT_DIR = DOWNLOADS / "FraudJobs" / "Outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Job state (in-memory progress tracking) ──────────────────────────────────
job_state = {
    "status": "idle",       # idle | running | done | error
    "progress": 0,
    "step": "",
    "results": None,
    "error": None
}

def update(status=None, progress=None, step=None):
    if status:  job_state["status"]   = status
    if progress is not None: job_state["progress"] = progress
    if step:    job_state["step"]     = step

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded

# ── Core ML pipeline ─────────────────────────────────────────────────────────
def run_pipeline(filepath):
    try:
        job_state["results"] = None
        job_state["error"]   = None
        sns.set_style("whitegrid")

        # STEP 1 – Load
        update("running", 5, "Loading dataset...")
        df = pd.read_csv(filepath)
        time.sleep(0.3)

        # STEP 2 – Preprocess
        update(progress=15, step="Preprocessing text data...")
        text_cols = ['title','company_profile','description','requirements','benefits']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].fillna('')
            else:
                df[col] = ''

        df['combined_text'] = (df['title'] + ' ' + df['company_profile'] + ' ' +
                               df['description'] + ' ' + df['requirements'] + ' ' + df['benefits'])

        # STEP 3 – Feature engineering
        update(progress=25, step="Engineering features...")
        df['has_salary']          = df['salary_range'].apply(lambda x: 0 if str(x).strip()=='' else 1) if 'salary_range' in df.columns else 0
        df['has_company_profile'] = df['company_profile'].apply(lambda x: 0 if str(x).strip()=='' else 1)
        df['has_requirements']    = df['requirements'].apply(lambda x: 0 if str(x).strip()=='' else 1)
        df['has_benefits']        = df['benefits'].apply(lambda x: 0 if str(x).strip()=='' else 1)
        df['desc_length']         = df['description'].apply(len)
        df['req_length']          = df['requirements'].apply(len)
        df['title_length']        = df['title'].apply(len)
        df['exclamation_count']   = df['description'].apply(lambda x: x.count('!'))
        df['caps_ratio']          = df['description'].apply(
            lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))
        suspicious_words = ['urgent','guaranteed','no experience','earn','easy money',
                            'work from home','unlimited','investment','apply now','limited']
        df['suspicious_keywords'] = df['combined_text'].apply(
            lambda x: sum(1 for w in suspicious_words if w in x.lower()))

        cat_cols = ['employment_type','required_experience','required_education']
        le = LabelEncoder()
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
                df[col+'_enc'] = le.fit_transform(df[col])
            else:
                df[col+'_enc'] = 0

        # Fix boolean columns — handles both 0/1 integers and 't'/'f' strings
        for bool_col in ['telecommuting', 'has_company_logo', 'has_questions']:
            if bool_col in df.columns:
                df[bool_col] = pd.to_numeric(df[bool_col], errors='coerce')
                df[bool_col] = df[bool_col].fillna(0).astype(int)
            else:
                df[bool_col] = 0

        structured_features = [
            'has_salary','has_company_profile','has_requirements','has_benefits',
            'telecommuting','has_company_logo','has_questions',
            'desc_length','req_length','title_length',
            'exclamation_count','caps_ratio','suspicious_keywords',
            'employment_type_enc','required_experience_enc','required_education_enc'
        ]
        # Ensure all features exist and are safely numeric
        for f in structured_features:
            if f not in df.columns:
                df[f] = 0
            df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

        # STEP 4 – TF-IDF
        update(progress=35, step="Extracting TF-IDF features...")
        tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2),
                                stop_words='english', sublinear_tf=True)
        X_tfidf = tfidf.fit_transform(df['combined_text'])
        X_struct = df[structured_features].values.astype(float)
        X_combined = sp.hstack([X_tfidf, sp.csr_matrix(X_struct)])
        # Fix target column — handles both 0/1 integers and 't'/'f' strings
        df['fraudulent'] = pd.to_numeric(df['fraudulent'], errors='coerce').fillna(0).astype(int)
        y = df['fraudulent'].values.astype(int)

        n_fraud = int(y.sum())
        n_legit = int((y==0).sum())

        # STEP 5 – Split + SMOTE
        update(progress=45, step="Balancing classes with SMOTE...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y, test_size=0.2, random_state=42, stratify=y)
        smote = SMOTE(random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

        # STEP 6 – Train RF
        update(progress=55, step="Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, max_depth=20,
                                    random_state=42, n_jobs=-1)
        rf.fit(X_train_bal, y_train_bal)
        rf_pred = rf.predict(X_test)
        rf_prob = rf.predict_proba(X_test)[:,1]

        # STEP 7 – Train GB
        update(progress=68, step="Training Gradient Boosting...")
        gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                         max_depth=4, random_state=42)
        gb.fit(X_train_bal, y_train_bal)
        gb_pred = gb.predict(X_test)
        gb_prob = gb.predict_proba(X_test)[:,1]

        def metrics(yt, yp, yprob):
            return {
                "accuracy":  round(accuracy_score(yt, yp),4),
                "precision": round(precision_score(yt, yp),4),
                "recall":    round(recall_score(yt, yp),4),
                "f1":        round(f1_score(yt, yp),4),
                "auc":       round(roc_auc_score(yt, yprob),4),
            }

        rf_metrics = metrics(y_test, rf_pred, rf_prob)
        gb_metrics = metrics(y_test, gb_pred, gb_prob)

        # STEP 8 – Generate charts
        update(progress=78, step="Generating visualizations...")

        charts = {}

        # Chart 1: Class distribution
        fig, ax = plt.subplots(figsize=(6,4))
        bars = ax.bar(['Legitimate','Fraudulent'], [n_legit, n_fraud],
                      color=['#22c55e','#ef4444'], edgecolor='black', width=0.5)
        for bar, val in zip(bars, [n_legit, n_fraud]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
                    str(val), ha='center', fontweight='bold', fontsize=11)
        ax.set_title('Class Distribution', fontsize=13, fontweight='bold')
        ax.set_ylabel('Count')
        charts['class_dist'] = fig_to_base64(fig)
        (OUTPUT_DIR / "fig1_class_distribution.png").write_bytes(
            base64.b64decode(charts['class_dist']))

        # Chart 2: Confusion matrices
        fig, axes = plt.subplots(1,2,figsize=(12,4))
        for ax, (name, pred) in zip(axes, [('Random Forest', rf_pred),
                                            ('Gradient Boosting', gb_pred)]):
            cm = confusion_matrix(y_test, pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Legit','Fraud'],
                        yticklabels=['Legit','Fraud'])
            ax.set_title(f'{name}', fontweight='bold')
            ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        plt.suptitle('Confusion Matrices', fontsize=13, fontweight='bold')
        charts['confusion'] = fig_to_base64(fig)
        (OUTPUT_DIR / "fig2_confusion_matrices.png").write_bytes(
            base64.b64decode(charts['confusion']))

        # Chart 3: Metric comparison
        metric_names = ['Accuracy','Precision','Recall','F1-Score','AUC-ROC']
        rf_vals  = [rf_metrics['accuracy'], rf_metrics['precision'],
                    rf_metrics['recall'], rf_metrics['f1'], rf_metrics['auc']]
        gb_vals  = [gb_metrics['accuracy'], gb_metrics['precision'],
                    gb_metrics['recall'], gb_metrics['f1'], gb_metrics['auc']]
        x = np.arange(len(metric_names)); w = 0.35
        fig, ax = plt.subplots(figsize=(10,5))
        b1 = ax.bar(x-w/2, rf_vals, w, label='Random Forest', color='#3b82f6', edgecolor='black')
        b2 = ax.bar(x+w/2, gb_vals, w, label='Gradient Boosting', color='#f97316', edgecolor='black')
        for bar in list(b1)+list(b2):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f'{bar.get_height():.3f}', ha='center', fontsize=8)
        ax.set_ylim(0,1.12); ax.set_xticks(x); ax.set_xticklabels(metric_names)
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
        ax.legend()
        charts['comparison'] = fig_to_base64(fig)
        (OUTPUT_DIR / "fig3_model_comparison.png").write_bytes(
            base64.b64decode(charts['comparison']))

        # Chart 4: ROC Curves
        fig, ax = plt.subplots(figsize=(7,5))
        for name, prob, col in [('Random Forest', rf_prob, '#3b82f6'),
                                  ('Gradient Boosting', gb_prob, '#f97316')]:
            fpr, tpr, _ = roc_curve(y_test, prob)
            auc_val = roc_auc_score(y_test, prob)
            ax.plot(fpr, tpr, label=f'{name} (AUC={auc_val:.4f})', color=col, lw=2)
        ax.plot([0,1],[0,1],'k--', lw=1.5, label='Random Classifier')
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves', fontsize=13, fontweight='bold')
        ax.legend(loc='lower right')
        charts['roc'] = fig_to_base64(fig)
        (OUTPUT_DIR / "fig4_roc_curves.png").write_bytes(
            base64.b64decode(charts['roc']))

        # Chart 5: Feature importance
        n_tfidf = X_tfidf.shape[1]
        struct_imp = rf.feature_importances_[n_tfidf:]
        fi_df = pd.DataFrame({'Feature': structured_features, 'Importance': struct_imp})
        fi_df = fi_df.sort_values('Importance', ascending=True).tail(12)
        fig, ax = plt.subplots(figsize=(8,5))
        ax.barh(fi_df['Feature'], fi_df['Importance'], color='#8b5cf6', edgecolor='black')
        ax.set_xlabel('Importance'); ax.set_title('Feature Importances (Random Forest)',
                                                    fontsize=12, fontweight='bold')
        charts['feature_imp'] = fig_to_base64(fig)
        (OUTPUT_DIR / "fig5_feature_importance.png").write_bytes(
            base64.b64decode(charts['feature_imp']))

        # STEP 9 – SHAP
        update(progress=88, step="Computing SHAP explainability...")
        try:
            X_tr_s = X_train_bal.toarray()[:, n_tfidf:]
            X_te_s = X_test.toarray()[:, n_tfidf:]
            rf_s = RandomForestClassifier(n_estimators=50, max_depth=8,
                                           random_state=42, n_jobs=-1)
            rf_s.fit(X_tr_s, y_train_bal)
            explainer = shap.TreeExplainer(rf_s)
            sv = explainer.shap_values(X_te_s[:300])
            if isinstance(sv, list):
                sv_pos = sv[1]
            elif hasattr(sv, 'ndim') and sv.ndim == 3:
                sv_pos = sv[:, :, 1]
            else:
                sv_pos = sv
            mean_shap = np.abs(sv_pos).mean(axis=0).flatten()
            shap_df = pd.DataFrame({'Feature': structured_features, 'SHAP': mean_shap})
            shap_df = shap_df.sort_values('SHAP', ascending=True)
            fig, ax = plt.subplots(figsize=(9,5))
            ax.barh(shap_df['Feature'], shap_df['SHAP'], color='#06b6d4', edgecolor='black')
            ax.set_xlabel('Mean |SHAP Value|')
            ax.set_title('SHAP Explainability', fontsize=12, fontweight='bold')
            charts['shap'] = fig_to_base64(fig)
            (OUTPUT_DIR / "fig6_shap_importance.png").write_bytes(
                base64.b64decode(charts['shap']))
        except Exception as shap_err:
            charts['shap'] = None

        # STEP 10 – Save results JSON
        update(progress=95, step="Saving results...")
        summary = {
            "dataset": {
                "total": len(df),
                "fraudulent": n_fraud,
                "legitimate": n_legit,
                "fraud_rate": round(n_fraud/len(df)*100, 2)
            },
            "random_forest": rf_metrics,
            "gradient_boosting": gb_metrics,
            "charts": charts
        }
        with open(OUTPUT_DIR / "results.json", "w") as f:
            json.dump({k: v for k, v in summary.items() if k != "charts"}, f, indent=2)

        job_state["results"] = summary
        update("done", 100, "Analysis complete!")

    except Exception as e:
        import traceback
        job_state["error"] = str(e) + "\n" + traceback.format_exc()
        update("error", 0, f"Error: {str(e)}")

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/api/upload', methods=['POST'])
def upload():
    if job_state["status"] == "running":
        return jsonify({"error": "Analysis already running"}), 400
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files['file']
    save_path = OUTPUT_DIR / f.filename
    f.save(str(save_path))
    # Run pipeline in background thread
    t = threading.Thread(target=run_pipeline, args=(str(save_path),))
    t.daemon = True
    t.start()
    return jsonify({"message": "Analysis started", "file": f.filename})

@app.route('/api/status')
def status():
    return jsonify({
        "status":   job_state["status"],
        "progress": job_state["progress"],
        "step":     job_state["step"],
        "error":    job_state["error"]
    })

@app.route('/api/results')
def results():
    if job_state["results"] is None:
        return jsonify({"error": "No results yet"}), 404
    return jsonify(job_state["results"])

@app.route('/api/reset', methods=['POST'])
def reset():
    job_state.update({"status":"idle","progress":0,"step":"","results":None,"error":None})
    return jsonify({"message": "Reset done"})

@app.route('/api/output-path')
def output_path():
    return jsonify({"path": str(OUTPUT_DIR)})

if __name__ == '__main__':
    print(f"\n✅ Backend running at http://localhost:5000")
    print(f"📁 Outputs will be saved to: {OUTPUT_DIR}\n")
    app.run(debug=True, port=5000)