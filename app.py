import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import re
from datetime import datetime
import anthropic
import base64
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VHP Sterilization Analytics",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Base */
[data-testid="stAppViewContainer"] {
    background: #0f1117;
}
[data-testid="stSidebar"] {
    background: #161b27;
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/* Cards */
.metric-card {
    background: #161b27;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 8px;
}
.metric-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #64748b;
    margin-bottom: 4px;
    font-family: monospace;
}
.metric-value {
    font-size: 22px;
    font-weight: 700;
    color: #f1f5f9;
    font-family: 'Courier New', monospace;
}
.metric-unit {
    font-size: 12px;
    color: #64748b;
    margin-left: 4px;
}
.phase-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin: 2px;
    font-family: monospace;
}
.header-title {
    font-size: 28px;
    font-weight: 800;
    color: #f1f5f9;
    letter-spacing: -0.5px;
    font-family: monospace;
}
.header-sub {
    font-size: 13px;
    color: #475569;
    margin-top: 2px;
    font-family: monospace;
}
.section-title {
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #38bdf8;
    font-family: monospace;
    padding: 8px 0 4px;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 12px;
}
.warn-box {
    background: #1a1200;
    border: 1px solid #854d0e;
    border-radius: 8px;
    padding: 12px 16px;
    color: #fbbf24;
    font-size: 13px;
    font-family: monospace;
}
.ok-box {
    background: #021a0e;
    border: 1px solid #166534;
    border-radius: 8px;
    padding: 12px 16px;
    color: #4ade80;
    font-size: 13px;
    font-family: monospace;
}
.stButton > button {
    background: #0ea5e9;
    color: #fff;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-family: monospace;
    letter-spacing: 0.5px;
}
.stButton > button:hover { background: #0284c7; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR = Path("vhp_runs")
DATA_DIR.mkdir(exist_ok=True)

PARAMS = [
    "Time",
    "Status",
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

PLOT_PARAMS = [p for p in PARAMS if p not in ("Time", "Status")]

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

# ── Helpers ───────────────────────────────────────────────────────────────────
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

def pdf_to_base64_images(pdf_bytes):
    """Convert PDF bytes to list of base64-encoded PNG images."""
    from pdf2image import convert_from_bytes
    images = convert_from_bytes(pdf_bytes, dpi=200)
    b64_list = []
    import io
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64_list.append(base64.standard_b64encode(buf.getvalue()).decode())
    return b64_list

def extract_with_claude(b64_images, run_name):
    """Send scanned PDF pages to Claude Vision and extract structured table data."""
    client = anthropic.Anthropic()

    system_prompt = """You are a precise scientific data extraction engine for pharmaceutical manufacturing.
You will receive scanned PDF pages from a VHP (Vaporized Hydrogen Peroxide) sterilization cycle report from Prince.
Your job: extract ALL tabular data rows exactly as they appear.

The table has these columns (in order):
1. Time (format HH:MM:SS or similar)
2. Status/Phase (text like Dehumidification, Conditioning, Sterilization, Aeration, Idle)
3. Chamber Temp 1 (C) — numeric
4. Chamber Temp 2 (C) — numeric
5. Vaporizer Temp 1 (C) — numeric
6. Vaporizer Temp 2 (C) — numeric
7. Jacket Temp (C) — numeric
8. Pressure (Psi) — numeric
9. H2O2 Conc (ppmV) — numeric
10. Relative Humidity (%) — numeric
11. Saturation (%) — numeric

Return ONLY valid JSON in this exact format, no commentary:
{
  "cycle_id": "extracted cycle ID or run number if visible",
  "cycle_date": "date if visible, else null",
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
If a value is not readable, use null. Extract every row visible on each page.
Combine rows from all pages into one "rows" array."""

    content = []
    for b64 in b64_images:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": b64,
            }
        })
    content.append({
        "type": "text",
        "text": f"Extract all tabular VHP cycle data from these {len(b64_images)} scanned PDF page(s). Return only JSON."
    })

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=8000,
        system=system_prompt,
        messages=[{"role": "user", "content": content}]
    )

    raw = response.content[0].text.strip()
    # Strip markdown fences if present
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    parsed = json.loads(raw)
    parsed["run_name"] = run_name
    parsed["uploaded_at"] = datetime.now().isoformat()
    return parsed

def phase_segments(df):
    """Return list of (start_idx, end_idx, phase_name) for background shading."""
    if "Status" not in df.columns:
        return []
    segs = []
    current = None
    start = 0
    for i, val in enumerate(df["Status"]):
        p = str(val).strip() if pd.notna(val) else "Unknown"
        if p != current:
            if current is not None:
                segs.append((start, i - 1, current))
            current = p
            start = i
    if current:
        segs.append((start, len(df) - 1, current))
    return segs

def build_multi_param_figure(df, selected_params, run_label="Run", show_phases=True):
    times = df["Time"] if "Time" in df.columns else list(range(len(df)))

    fig = go.Figure()

    # Phase background shading
    if show_phases and "Status" in df.columns:
        for start, end, phase in phase_segments(df):
            col = phase_color(phase)
            if start < len(times) and end < len(times):
                fig.add_vrect(
                    x0=times.iloc[start] if hasattr(times, "iloc") else times[start],
                    x1=times.iloc[end] if hasattr(times, "iloc") else times[end],
                    fillcolor=col,
                    opacity=0.08,
                    line_width=0,
                    annotation_text=phase,
                    annotation_position="top left",
                    annotation_font_size=10,
                    annotation_font_color=col,
                )

    for param in selected_params:
        if param in df.columns:
            fig.add_trace(go.Scatter(
                x=times,
                y=df[param],
                name=param,
                mode="lines",
                line=dict(color=PARAM_COLORS.get(param, "#94a3b8"), width=1.8),
                hovertemplate=f"<b>{param}</b><br>Time: %{{x}}<br>Value: %{{y:.2f}} {PARAM_UNITS.get(param,'')}<extra></extra>",
            ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0d1117",
        font=dict(family="monospace", color="#94a3b8", size=11),
        legend=dict(
            bgcolor="rgba(15,17,23,0.8)",
            bordercolor="#1e293b",
            borderwidth=1,
            font=dict(size=10),
        ),
        hovermode="x unified",
        xaxis=dict(
            title="Time",
            gridcolor="#1e293b",
            linecolor="#1e293b",
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            title="Value",
            gridcolor="#1e293b",
            linecolor="#1e293b",
            tickfont=dict(size=10),
        ),
        margin=dict(l=50, r=20, t=30, b=50),
        height=420,
    )
    return fig

def build_comparison_figure(runs_dict, selected_params, selected_run_names):
    """Overlay multiple runs on one chart for a given parameter, one subplot per param."""
    n = len(selected_params)
    if n == 0:
        return go.Figure()

    fig = make_subplots(
        rows=n, cols=1,
        shared_xaxes=False,
        subplot_titles=selected_params,
        vertical_spacing=0.06,
    )

    run_colors = ["#38bdf8", "#f97316", "#a78bfa", "#4ade80", "#fb7185"]

    for pi, param in enumerate(selected_params):
        for ri, rname in enumerate(selected_run_names):
            if rname not in runs_dict:
                continue
            df = df_from_run(runs_dict[rname])
            times = df["Time"] if "Time" in df.columns else list(range(len(df)))
            if param not in df.columns:
                continue
            color = run_colors[ri % len(run_colors)]
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=df[param],
                    name=rname,
                    mode="lines",
                    line=dict(color=color, width=1.6),
                    legendgroup=rname,
                    showlegend=(pi == 0),
                    hovertemplate=f"<b>{rname}</b><br>Time: %{{x}}<br>{param}: %{{y:.2f}}<extra></extra>",
                ),
                row=pi + 1, col=1,
            )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0d1117",
        font=dict(family="monospace", color="#94a3b8", size=11),
        legend=dict(
            bgcolor="rgba(15,17,23,0.8)",
            bordercolor="#1e293b",
            borderwidth=1,
        ),
        hovermode="x unified",
        height=300 * n,
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
    st.markdown(f'<div class="metric-label">Stored runs</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{len(runs)}</div>', unsafe_allow_html=True)

# ── Page: Upload ──────────────────────────────────────────────────────────────
if nav == "📤 Upload New Run":
    st.markdown('<div class="section-title">Upload VHP Cycle PDF</div>', unsafe_allow_html=True)
    st.markdown("Upload a scanned PDF from Prince sterilizer. Claude Vision will OCR and extract all 11 parameters.")

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader("Select scanned PDF", type=["pdf"], label_visibility="collapsed")
    with col2:
        run_name = st.text_input("Run name", placeholder="e.g. VHP-F01-Feasibility")

    if not run_name:
        st.markdown('<div class="warn-box">⚠ Enter a run name before uploading.</div>', unsafe_allow_html=True)
    elif run_name in runs:
        st.markdown(f'<div class="warn-box">⚠ Run "{run_name}" already exists. Choose a different name.</div>', unsafe_allow_html=True)
    elif uploaded:
        if st.button("🔍 Extract & Save Run"):
            with st.spinner("Converting PDF pages to images…"):
                b64_images = pdf_to_base64_images(uploaded.read())
            st.info(f"Extracted {len(b64_images)} page(s). Sending to Claude Vision for OCR…")
            progress = st.progress(0, text="Extracting data…")
            try:
                extracted = extract_with_claude(b64_images, run_name)
                progress.progress(100, text="Done!")
                save_run(run_name, extracted)
                nrows = len(extracted.get("rows", []))
                st.markdown(f'<div class="ok-box">✓ Extracted {nrows} rows. Run "{run_name}" saved.</div>', unsafe_allow_html=True)

                # Quick preview
                df = df_from_run(extracted)
                st.markdown('<div class="section-title" style="margin-top:20px">Preview (first 20 rows)</div>', unsafe_allow_html=True)
                st.dataframe(df.head(20), use_container_width=True)
            except json.JSONDecodeError as e:
                st.error(f"JSON parse error: {e}. Try re-uploading or check PDF quality.")
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

        # Run metadata
        meta_cols = st.columns(4)
        meta_cols[0].markdown(f'<div class="metric-card"><div class="metric-label">Run ID</div><div class="metric-value" style="font-size:16px">{run.get("cycle_id", selected)}</div></div>', unsafe_allow_html=True)
        meta_cols[1].markdown(f'<div class="metric-card"><div class="metric-label">Date</div><div class="metric-value" style="font-size:16px">{run.get("cycle_date", "—")}</div></div>', unsafe_allow_html=True)
        meta_cols[2].markdown(f'<div class="metric-card"><div class="metric-label">Data Points</div><div class="metric-value">{len(df)}</div></div>', unsafe_allow_html=True)
        meta_cols[3].markdown(f'<div class="metric-card"><div class="metric-label">Uploaded</div><div class="metric-value" style="font-size:14px">{run.get("uploaded_at","—")[:10]}</div></div>', unsafe_allow_html=True)

        # Phase summary
        if "Status" in df.columns:
            phases = df["Status"].dropna().unique()
            st.markdown("**Phases detected:**")
            badges = ""
            for ph in phases:
                col = phase_color(ph)
                badges += f'<span class="phase-badge" style="background:{col}22;color:{col};border:1px solid {col}44">{ph}</span>'
            st.markdown(badges, unsafe_allow_html=True)

        st.markdown("---")

        # Parameter summary cards
        st.markdown('<div class="section-title">Parameter Summary</div>', unsafe_allow_html=True)
        pcols = st.columns(3)
        for i, param in enumerate(PLOT_PARAMS):
            if param in df.columns:
                series = df[param].dropna()
                if not series.empty:
                    unit = PARAM_UNITS.get(param, "")
                    card_html = f"""<div class="metric-card">
<div class="metric-label">{param}</div>
<div style="display:flex;gap:12px;margin-top:4px">
  <div><div class="metric-label">Max</div><div class="metric-value" style="font-size:16px;color:{PARAM_COLORS.get(param,'#f1f5f9')}">{series.max():.1f}<span class="metric-unit">{unit}</span></div></div>
  <div><div class="metric-label">Min</div><div class="metric-value" style="font-size:16px">{series.min():.1f}<span class="metric-unit">{unit}</span></div></div>
  <div><div class="metric-label">Mean</div><div class="metric-value" style="font-size:16px">{series.mean():.1f}<span class="metric-unit">{unit}</span></div></div>
</div></div>"""
                    pcols[i % 3].markdown(card_html, unsafe_allow_html=True)

        st.markdown("---")

        # Chart
        st.markdown('<div class="section-title">Time-Series Chart</div>', unsafe_allow_html=True)
        selected_params = st.multiselect(
            "Parameters to plot",
            PLOT_PARAMS,
            default=["Chamber Temp 1 (C)", "Chamber Temp 2 (C)", "H2O2 Conc (ppmV)", "Relative Humidity (%)"],
        )
        show_phases = st.checkbox("Show phase shading", value=True)

        if selected_params:
            fig = build_multi_param_figure(df, selected_params, run_label=selected, show_phases=show_phases)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select at least one parameter to plot.")

        # Raw data
        with st.expander("Raw data table"):
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode()
            st.download_button("Download CSV", csv, f"{selected}.csv", "text/csv")

# ── Page: Compare ─────────────────────────────────────────────────────────────
elif nav == "🔀 Compare Runs":
    st.markdown('<div class="section-title">Multi-Run Comparison</div>', unsafe_allow_html=True)
    runs = load_runs()

    if len(runs) < 2:
        st.markdown('<div class="warn-box">Need at least 2 uploaded runs to compare.</div>', unsafe_allow_html=True)
    else:
        col_a, col_b = st.columns([1, 1])
        with col_a:
            compare_runs = st.multiselect("Select runs to compare", list(runs.keys()), default=list(runs.keys())[:2])
        with col_b:
            compare_params = st.multiselect(
                "Parameters to compare",
                PLOT_PARAMS,
                default=["Chamber Temp 1 (C)", "H2O2 Conc (ppmV)", "Relative Humidity (%)"],
            )

        if compare_runs and compare_params:
            fig = build_comparison_figure(runs, compare_params, compare_runs)
            st.plotly_chart(fig, use_container_width=True)

            # Delta table — max/min per run per param
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
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("Select runs and parameters above.")

# ── Page: Manage ──────────────────────────────────────────────────────────────
elif nav == "🗂 Manage Runs":
    st.markdown('<div class="section-title">Manage Stored Runs</div>', unsafe_allow_html=True)
    runs = load_runs()

    if not runs:
        st.info("No runs stored yet.")
    else:
        for rname, rdata in runs.items():
            with st.expander(f"🔬 {rname}  ·  {len(rdata.get('rows', []))} rows  ·  {rdata.get('cycle_date','—')}"):
                col1, col2 = st.columns([3, 1])
                col1.write(f"**Cycle ID:** {rdata.get('cycle_id', '—')}")
                col1.write(f"**Uploaded:** {rdata.get('uploaded_at', '—')[:19]}")
                col1.write(f"**Rows:** {len(rdata.get('rows', []))}")
                if col2.button("🗑 Delete", key=f"del_{rname}"):
                    delete_run(rname)
                    st.rerun()
                df = df_from_run(rdata)
                st.dataframe(df.head(5), use_container_width=True)
