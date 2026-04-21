import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import re
from datetime import datetime
import base64
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VHP Sterilization Analytics",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"] { background: #161b27; border-right: 1px solid #1e293b; }
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
.metric-card {
    background: #161b27; border: 1px solid #1e293b;
    border-radius: 10px; padding: 16px 20px; margin-bottom: 8px;
}
.metric-label {
    font-size: 11px; text-transform: uppercase; letter-spacing: 1px;
    color: #64748b; margin-bottom: 4px; font-family: monospace;
}
.metric-value { font-size: 22px; font-weight: 700; color: #f1f5f9; font-family: 'Courier New', monospace; }
.metric-unit { font-size: 12px; color: #64748b; margin-left: 4px; }
.phase-badge {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 600; letter-spacing: 0.5px;
    margin: 2px; font-family: monospace;
}
.header-title { font-size: 28px; font-weight: 800; color: #f1f5f9; letter-spacing: -0.5px; font-family: monospace; }
.header-sub { font-size: 13px; color: #475569; margin-top: 2px; font-family: monospace; }
.section-title {
    font-size: 13px; text-transform: uppercase; letter-spacing: 2px;
    color: #38bdf8; font-family: monospace; padding: 8px 0 4px;
    border-bottom: 1px solid #1e293b; margin-bottom: 12px;
}
.warn-box {
    background: #1a1200; border: 1px solid #854d0e; border-radius: 8px;
    padding: 12px 16px; color: #fbbf24; font-size: 13px; font-family: monospace;
}
.ok-box {
    background: #021a0e; border: 1px solid #166534; border-radius: 8px;
    padding: 12px 16px; color: #4ade80; font-size: 13px; font-family: monospace;
}
.stButton > button {
    background: #0ea5e9; color: #fff; border: none; border-radius: 8px;
    font-weight: 600; font-family: monospace; letter-spacing: 0.5px;
}
.stButton > button:hover { background: #0284c7; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR = Path("vhp_runs")
DATA_DIR.mkdir(exist_ok=True)

PLOT_PARAMS = [
    "Chamber Temp 1 (C)",
    "Chamber Temp 2 (C)",
    "Vaporizer Temp 1 (C)",
    "Vaporizer Temp 2 (C)",
    "Jacket Temp (C)",
    "Pressure (Psi)",
    "H2O2 Conc (ppmV)",
    "Relative Humidity (%)",
    "Saturation (%)",
]

PHASE_COLORS = {
    "dehumidification": "#f59e0b",
    "conditioning":     "#8b5cf6",
    "sterilization":    "#ef4444",
    "aeration":         "#22c55e",
    "idle":             "#64748b",
    "unknown":          "#94a3b8",
}

PARAM_COLORS = {
    "Chamber Temp 1 (C)":    "#38bdf8",
    "Chamber Temp 2 (C)":    "#7dd3fc",
    "Vaporizer Temp 1 (C)":  "#f97316",
    "Vaporizer Temp 2 (C)":  "#fdba74",
    "Jacket Temp (C)":       "#a78bfa",
    "Pressure (Psi)":        "#fb7185",
    "H2O2 Conc (ppmV)":      "#4ade80",
    "Relative Humidity (%)": "#facc15",
    "Saturation (%)":        "#34d399",
}

PARAM_UNITS = {
    "Chamber Temp 1 (C)":    "°C",
    "Chamber Temp 2 (C)":    "°C",
    "Vaporizer Temp 1 (C)":  "°C",
    "Vaporizer Temp 2 (C)":  "°C",
    "Jacket Temp (C)":       "°C",
    "Pressure (Psi)":        "Psi",
    "H2O2 Conc (ppmV)":      "ppmV",
    "Relative Humidity (%)": "%",
    "Saturation (%)":        "%",
}

# ── Gemini API key ─────────────────────────────────────────────────────────────
def get_gemini_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.environ.get("GEMINI_API_KEY", "")

# ── PDF → base64 images (PyMuPDF, no system deps) ────────────────────────────
def pdf_to_base64_images(pdf_bytes):
    import fitz
    import io
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    b64_list = []
    for page in doc:
        mat = fitz.Matrix(2.5, 2.5)   # ~250 DPI for good OCR
        pix = page.get_pixmap(matrix=mat)
        b64_list.append(base64.standard_b64encode(pix.tobytes("png")).decode())
    return b64_list

# ── Gemini OCR extraction ─────────────────────────────────────────────────────
def extract_with_gemini(b64_images, run_name):
    import google.generativeai as genai
    from PIL import Image
    import io

    api_key = get_gemini_key()
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in Streamlit secrets.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    prompt = """You are a precise scientific data extraction engine for pharmaceutical manufacturing.
These are scanned pages from a VHP (Vaporized Hydrogen Peroxide) sterilization cycle report from a Prince sterilizer.

Extract ALL tabular data rows. The table columns are:
1. Time (HH:MM:SS format)
2. Status / Phase (e.g. Dehumidification, Conditioning, Sterilization, Aeration, Idle)
3. Chamber Temp 1 (C)
4. Chamber Temp 2 (C)
5. Vaporizer Temp 1 (C)
6. Vaporizer Temp 2 (C)
7. Jacket Temp (C)
8. Pressure (Psi)
9. H2O2 Conc (ppmV)
10. Relative Humidity (%)
11. Saturation (%)

Return ONLY valid JSON, no explanation, no markdown fences:
{
  "cycle_id": "cycle ID or run number visible in the report, or null",
  "cycle_date": "date visible in the report, or null",
  "rows": [
    {
      "Time": "00:00:10",
      "Status": "Idle",
      "Chamber Temp 1 (C)": 25.3,
      "Chamber Temp 2 (C)": 25.1,
      "Vaporizer Temp 1 (C)": 40.2,
      "Vaporizer Temp 2 (C)": 40.1,
      "Jacket Temp (C)": 30.0,
      "Pressure (Psi)": 14.7,
      "H2O2 Conc (ppmV)": 0.0,
      "Relative Humidity (%)": 45.2,
      "Saturation (%)": 12.1
    }
  ]
}
If a cell is unreadable use null. Extract every visible row across all pages into one rows array."""

    # Convert base64 images to PIL images for Gemini
    content = [prompt]
    for b64 in b64_images:
        img_bytes = base64.b64decode(b64)
        img = Image.open(io.BytesIO(img_bytes))
        content.append(img)

    response = model.generate_content(content)
    raw = response.text.strip()

    # Strip markdown fences if Gemini adds them
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    parsed = json.loads(raw)
    parsed["run_name"] = run_name
    parsed["uploaded_at"] = datetime.now().isoformat()
    return parsed

# ── Data helpers ──────────────────────────────────────────────────────────────
def load_runs():
    runs = {}
    for f in sorted(DATA_DIR.glob("*.json")):
        try:
            with open(f) as fp:
                runs[f.stem] = json.load(fp)
        except Exception:
            pass
    return runs

def save_run(name, data):
    with open(DATA_DIR / f"{name}.json", "w") as fp:
        json.dump(data, fp)

def delete_run(name):
    p = DATA_DIR / f"{name}.json"
    if p.exists():
        p.unlink()

def df_from_run(run):
    df = pd.DataFrame(run["rows"])
    for col in PLOT_PARAMS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def phase_color(phase_str):
    if not phase_str:
        return PHASE_COLORS["unknown"]
    p = str(phase_str).lower()
    for k, v in PHASE_COLORS.items():
        if k in p:
            return v
    return PHASE_COLORS["unknown"]

def phase_segments(df):
    if "Status" not in df.columns:
        return []
    segs, current, start = [], None, 0
    for i, val in enumerate(df["Status"]):
        p = str(val).strip() if pd.notna(val) else "Unknown"
        if p != current:
            if current is not None:
                segs.append((start, i - 1, current))
            current, start = p, i
    if current:
        segs.append((start, len(df) - 1, current))
    return segs

# ── Chart builders ────────────────────────────────────────────────────────────
def build_single_figure(df, selected_params, show_phases=True):
    times = df["Time"] if "Time" in df.columns else list(range(len(df)))
    fig = go.Figure()

    if show_phases and "Status" in df.columns:
        for start, end, phase in phase_segments(df):
            col = phase_color(phase)
            if start < len(times) and end < len(times):
                fig.add_vrect(
                    x0=times.iloc[start] if hasattr(times, "iloc") else times[start],
                    x1=times.iloc[end]   if hasattr(times, "iloc") else times[end],
                    fillcolor=col, opacity=0.08, line_width=0,
                    annotation_text=phase, annotation_position="top left",
                    annotation_font_size=10, annotation_font_color=col,
                )

    for param in selected_params:
        if param in df.columns:
            fig.add_trace(go.Scatter(
                x=times, y=df[param], name=param, mode="lines",
                line=dict(color=PARAM_COLORS.get(param, "#94a3b8"), width=1.8),
                hovertemplate=f"<b>{param}</b><br>Time: %{{x}}<br>Value: %{{y:.2f}} {PARAM_UNITS.get(param,'')}<extra></extra>",
            ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
        font=dict(family="monospace", color="#94a3b8", size=11),
        legend=dict(bgcolor="rgba(15,17,23,0.8)", bordercolor="#1e293b", borderwidth=1, font=dict(size=10)),
        hovermode="x unified",
        xaxis=dict(title="Time", gridcolor="#1e293b", linecolor="#1e293b", tickfont=dict(size=10)),
        yaxis=dict(title="Value", gridcolor="#1e293b", linecolor="#1e293b", tickfont=dict(size=10)),
        margin=dict(l=50, r=20, t=30, b=50), height=440,
    )
    return fig

def build_comparison_figure(runs_dict, selected_params, selected_run_names):
    n = len(selected_params)
    if n == 0:
        return go.Figure()

    fig = make_subplots(
        rows=n, cols=1, shared_xaxes=False,
        subplot_titles=selected_params, vertical_spacing=0.06,
    )
    run_colors = ["#38bdf8", "#f97316", "#a78bfa", "#4ade80", "#fb7185", "#facc15"]

    for pi, param in enumerate(selected_params):
        for ri, rname in enumerate(selected_run_names):
            if rname not in runs_dict:
                continue
            df = df_from_run(runs_dict[rname])
            times = df["Time"] if "Time" in df.columns else list(range(len(df)))
            if param not in df.columns:
                continue
            fig.add_trace(
                go.Scatter(
                    x=times, y=df[param], name=rname, mode="lines",
                    line=dict(color=run_colors[ri % len(run_colors)], width=1.6),
                    legendgroup=rname, showlegend=(pi == 0),
                    hovertemplate=f"<b>{rname}</b><br>Time: %{{x}}<br>{param}: %{{y:.2f}}<extra></extra>",
                ),
                row=pi + 1, col=1,
            )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
        font=dict(family="monospace", color="#94a3b8", size=11),
        legend=dict(bgcolor="rgba(15,17,23,0.8)", bordercolor="#1e293b", borderwidth=1),
        hovermode="x unified", height=320 * n,
        margin=dict(l=50, r=20, t=40, b=40),
    )
    for i in range(1, n + 1):
        fig.update_xaxes(gridcolor="#1e293b", linecolor="#1e293b", row=i, col=1)
        fig.update_yaxes(gridcolor="#1e293b", linecolor="#1e293b", row=i, col=1)
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="header-title">VHP Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-sub">Biocon Biologics · DP MSAT Devices</div>', unsafe_allow_html=True)
    st.markdown("---")
    nav = st.radio(
        "Navigation",
        ["📤 Upload New Run", "📊 Analyse Run", "🔀 Compare Runs", "🗂 Manage Runs"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    runs = load_runs()
    st.markdown('<div class="metric-label">Stored runs</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{len(runs)}</div>', unsafe_allow_html=True)

    # API key status indicator
    st.markdown("---")
    key_set = bool(get_gemini_key())
    if key_set:
        st.markdown('<div class="ok-box">✓ Gemini API key set</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warn-box">⚠ Gemini API key missing</div>', unsafe_allow_html=True)

# ── Page: Upload ──────────────────────────────────────────────────────────────
if nav == "📤 Upload New Run":
    st.markdown('<div class="section-title">Upload VHP Cycle PDF</div>', unsafe_allow_html=True)
    st.markdown("Upload a scanned PDF from Prince sterilizer. Gemini Vision will OCR and extract all 11 parameters automatically.")

    if not get_gemini_key():
        st.markdown("""<div class="warn-box">
⚠ Gemini API key not configured.<br>
Go to your Streamlit app → ⋮ menu → Settings → Secrets → add:<br><br>
<code>GEMINI_API_KEY = "your-key-here"</code>
</div>""", unsafe_allow_html=True)
        st.stop()

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader("Select scanned PDF", type=["pdf"], label_visibility="collapsed")
    with col2:
        run_name = st.text_input("Run name", placeholder="e.g. VHP-F01-Feasibility")

    runs = load_runs()
    if not run_name:
        st.markdown('<div class="warn-box">⚠ Enter a run name before uploading.</div>', unsafe_allow_html=True)
    elif run_name in runs:
        st.markdown(f'<div class="warn-box">⚠ Run "{run_name}" already exists. Choose a different name or delete the existing one.</div>', unsafe_allow_html=True)
    elif uploaded:
        if st.button("🔍 Extract & Save Run"):
            with st.spinner("Converting PDF pages to images…"):
                try:
                    b64_images = pdf_to_base64_images(uploaded.read())
                except Exception as e:
                    st.error(f"PDF conversion failed: {e}. Make sure PyMuPDF is in requirements.txt")
                    st.stop()

            st.info(f"Extracted {len(b64_images)} page(s). Sending to Gemini Vision for OCR — this takes 20–40 seconds…")
            try:
                with st.spinner("Gemini is reading your cycle data…"):
                    extracted = extract_with_gemini(b64_images, run_name)
                save_run(run_name, extracted)
                nrows = len(extracted.get("rows", []))
                st.markdown(f'<div class="ok-box">✓ Extracted {nrows} rows. Run "{run_name}" saved successfully.</div>', unsafe_allow_html=True)
                df = df_from_run(extracted)
                st.markdown('<div class="section-title" style="margin-top:20px">Preview — first 20 rows</div>', unsafe_allow_html=True)
                st.dataframe(df.head(20), use_container_width=True)
            except json.JSONDecodeError:
                st.error("Gemini returned malformed JSON. This sometimes happens with very low quality scans. Try a higher resolution PDF scan.")
            except Exception as e:
                st.error(f"Extraction error: {e}")

# ── Page: Analyse ─────────────────────────────────────────────────────────────
elif nav == "📊 Analyse Run":
    st.markdown('<div class="section-title">Single Run Analysis</div>', unsafe_allow_html=True)
    runs = load_runs()

    if not runs:
        st.markdown('<div class="warn-box">No runs uploaded yet. Go to Upload New Run first.</div>', unsafe_allow_html=True)
    else:
        selected = st.selectbox("Select run", list(runs.keys()))
        run = runs[selected]
        df = df_from_run(run)

        # Metadata row
        mc = st.columns(4)
        mc[0].markdown(f'<div class="metric-card"><div class="metric-label">Run ID</div><div class="metric-value" style="font-size:15px">{run.get("cycle_id") or selected}</div></div>', unsafe_allow_html=True)
        mc[1].markdown(f'<div class="metric-card"><div class="metric-label">Cycle date</div><div class="metric-value" style="font-size:15px">{run.get("cycle_date") or "—"}</div></div>', unsafe_allow_html=True)
        mc[2].markdown(f'<div class="metric-card"><div class="metric-label">Data points</div><div class="metric-value">{len(df)}</div></div>', unsafe_allow_html=True)
        mc[3].markdown(f'<div class="metric-card"><div class="metric-label">Uploaded</div><div class="metric-value" style="font-size:13px">{run.get("uploaded_at","—")[:10]}</div></div>', unsafe_allow_html=True)

        # Phases
        if "Status" in df.columns:
            phases = df["Status"].dropna().unique()
            badges = "".join(
                f'<span class="phase-badge" style="background:{phase_color(ph)}22;color:{phase_color(ph)};border:1px solid {phase_color(ph)}44">{ph}</span>'
                for ph in phases
            )
            st.markdown(f"**Phases detected:** {badges}", unsafe_allow_html=True)

        st.markdown("---")

        # Summary cards
        st.markdown('<div class="section-title">Parameter Summary</div>', unsafe_allow_html=True)
        pcols = st.columns(3)
        for i, param in enumerate(PLOT_PARAMS):
            if param in df.columns:
                s = df[param].dropna()
                if not s.empty:
                    unit = PARAM_UNITS.get(param, "")
                    col_hex = PARAM_COLORS.get(param, "#f1f5f9")
                    pcols[i % 3].markdown(f"""<div class="metric-card">
<div class="metric-label">{param}</div>
<div style="display:flex;gap:16px;margin-top:6px">
  <div><div class="metric-label">Max</div><div class="metric-value" style="font-size:15px;color:{col_hex}">{s.max():.1f}<span class="metric-unit">{unit}</span></div></div>
  <div><div class="metric-label">Min</div><div class="metric-value" style="font-size:15px">{s.min():.1f}<span class="metric-unit">{unit}</span></div></div>
  <div><div class="metric-label">Mean</div><div class="metric-value" style="font-size:15px">{s.mean():.1f}<span class="metric-unit">{unit}</span></div></div>
</div></div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Chart
        st.markdown('<div class="section-title">Time-Series Chart</div>', unsafe_allow_html=True)
        sel_params = st.multiselect(
            "Parameters to plot", PLOT_PARAMS,
            default=["Chamber Temp 1 (C)", "Chamber Temp 2 (C)", "H2O2 Conc (ppmV)", "Relative Humidity (%)"],
        )
        show_phases = st.checkbox("Show phase shading", value=True)
        if sel_params:
            st.plotly_chart(build_single_figure(df, sel_params, show_phases), use_container_width=True)
        else:
            st.info("Select at least one parameter to plot.")

        # Raw data + download
        with st.expander("Raw data table"):
            st.dataframe(df, use_container_width=True)
            st.download_button("⬇ Download CSV", df.to_csv(index=False).encode(), f"{selected}.csv", "text/csv")

# ── Page: Compare ─────────────────────────────────────────────────────────────
elif nav == "🔀 Compare Runs":
    st.markdown('<div class="section-title">Multi-Run Comparison</div>', unsafe_allow_html=True)
    runs = load_runs()

    if len(runs) < 2:
        st.markdown('<div class="warn-box">Need at least 2 uploaded runs to compare. Upload more runs first.</div>', unsafe_allow_html=True)
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            compare_runs = st.multiselect("Select runs", list(runs.keys()), default=list(runs.keys())[:2])
        with col_b:
            compare_params = st.multiselect(
                "Parameters to compare", PLOT_PARAMS,
                default=["Chamber Temp 1 (C)", "H2O2 Conc (ppmV)", "Relative Humidity (%)"],
            )

        if compare_runs and compare_params:
            st.plotly_chart(build_comparison_figure(runs, compare_params, compare_runs), use_container_width=True)

            # Summary table
            st.markdown('<div class="section-title" style="margin-top:16px">Run comparison table</div>', unsafe_allow_html=True)
            rows = []
            for rname in compare_runs:
                df = df_from_run(runs[rname])
                row = {"Run": rname}
                for param in compare_params:
                    if param in df.columns:
                        s = df[param].dropna()
                        row[f"{param} max"] = round(s.max(), 2) if not s.empty else None
                        row[f"{param} min"] = round(s.min(), 2) if not s.empty else None
                        row[f"{param} mean"] = round(s.mean(), 2) if not s.empty else None
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("Select at least 2 runs and 1 parameter above.")

# ── Page: Manage ──────────────────────────────────────────────────────────────
elif nav == "🗂 Manage Runs":
    st.markdown('<div class="section-title">Manage Stored Runs</div>', unsafe_allow_html=True)
    runs = load_runs()

    if not runs:
        st.info("No runs stored yet.")
    else:
        for rname, rdata in runs.items():
            with st.expander(f"🔬 {rname}  ·  {len(rdata.get('rows', []))} rows  ·  {rdata.get('cycle_date') or '—'}"):
                c1, c2 = st.columns([3, 1])
                c1.write(f"**Cycle ID:** {rdata.get('cycle_id') or '—'}")
                c1.write(f"**Uploaded:** {rdata.get('uploaded_at','—')[:19]}")
                c1.write(f"**Total rows:** {len(rdata.get('rows',[]))}")
                if c2.button("🗑 Delete", key=f"del_{rname}"):
                    delete_run(rname)
                    st.rerun()
                st.dataframe(df_from_run(rdata).head(5), use_container_width=True)
