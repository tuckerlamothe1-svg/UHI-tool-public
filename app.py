# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Neighborhood Heat Tool", layout="wide")

st.title("Neighborhood Heat Tool — Simple Intervention Simulator")
st.markdown("""
Enter baseline values and proposed interventions.  
The model outputs estimated surface & air temperature change (°F), component breakdown, and a simple heat score.
""")

# --- Model functions (same math we discussed) ---
def predict_temp_change(
    TC_base, TC_new,
    IMP_base, IMP_new,
    ALB_base, ALB_new,
    CP_base, CP_new,
    GR_base, GR_new,
    Ts_base,
    alpha=0.85,
    beta=0.6,
    coeffs=None
):
    # default coefficients (midpoints) if none provided
    if coeffs is None:
        coeffs = {
            "k_tree": -0.15,   # °F per 1% canopy
            "k_imp": 0.12,     # °F per 1% impervious
            "k_alb_0.1": -2.5, # °F per +0.1 albedo
            "k_cp": -0.25,     # °F per 1% cool pavement adoption
            "k_gr": -0.045     # °F per 1% green roof coverage
        }

    k_tree = coeffs["k_tree"]
    k_imp = coeffs["k_imp"]
    k_alb_0.1 = coeffs["k_alb_0.1"]
    k_cp = coeffs["k_cp"]
    k_gr = coeffs["k_gr"]

    dT_tree = k_tree * (TC_new - TC_base)
    dT_imp = k_imp * (IMP_new - IMP_base)
    dT_alb = k_alb_0.1 * ((ALB_new - ALB_base) / 0.1)
    dT_cp = k_cp * (CP_new - CP_base)
    dT_gr = k_gr * (GR_new - GR_base)

    dT_sum = dT_tree + dT_imp + dT_alb + dT_cp + dT_gr
    dT_total = alpha * dT_sum
    dT_air = beta * dT_total

    components = {
        "Tree canopy (°F)": dT_tree,
        "Impervious surface (°F)": dT_imp,
        "Albedo change (°F)": dT_alb,
        "Cool pavement (°F)": dT_cp,
        "Green roofs (°F)": dT_gr
    }

    return {
        "ΔT_surface": dT_total,
        "ΔT_air": dT_air,
        "Ts_new": Ts_base + dT_total,
        "components": components
    }

# --- Sidebar: inputs ---
st.sidebar.header("Input parameters")

st.sidebar.markdown("**Baseline / Proposed values**")
TC_base = st.sidebar.slider("Tree canopy — baseline (%)", 0.0, 80.0, 12.0, step=1.0)
TC_new = st.sidebar.slider("Tree canopy — proposed (%)", TC_base, 80.0, min(TC_base+5, TC_base), step=1.0)

IMP_base = st.sidebar.slider("Impervious surface — baseline (%)", 0.0, 100.0, 61.0, step=1.0)
IMP_new = st.sidebar.slider("Impervious surface — proposed (%)", 0.0, 100.0, IMP_base, step=1.0)

ALB_base = st.sidebar.slider("Albedo — baseline (0.0–1.0)", 0.0, 1.0, 0.12, step=0.01)
ALB_new = st.sidebar.slider("Albedo — proposed (0.0–1.0)", 0.0, 1.0, ALB_base, step=0.01)

CP_base = st.sidebar.slider("Cool pavement — baseline (%)", 0.0, 100.0, 0.0, step=1.0)
CP_new = st.sidebar.slider("Cool pavement — proposed (%)", 0.0, 100.0, CP_base, step=1.0)

GR_base = st.sidebar.slider("Green roofs — baseline (%)", 0.0, 100.0, 0.0, step=1.0)
GR_new = st.sidebar.slider("Green roofs — proposed (%)", 0.0, 100.0, GR_base, step=1.0)

Ts_base = st.sidebar.number_input("Baseline peak surface temp (°F)", value=95.0, step=0.1)

alpha = st.sidebar.slider("Interaction factor (α)", 0.5, 1.0, 0.85, step=0.01)
beta = st.sidebar.slider("Air temp scaling (β)", 0.1, 1.0, 0.6, step=0.01)

st.sidebar.markdown("---")
st.sidebar.markdown("**Coefficient overrides (optional)** — leave blank to use defaults.")
k_tree = st.sidebar.number_input("k_tree (°F per 1% canopy)", value=-0.15, step=0.01)
k_imp = st.sidebar.number_input("k_imp (°F per 1% impervious)", value=0.12, step=0.01)
k_alb = st.sidebar.number_input("k_albedo_0.1 (°F per 0.1 albedo)", value=-2.5, step=0.1)
k_cp = st.sidebar.number_input("k_cp (°F per 1% cool pavement)", value=-0.25, step=0.01)
k_gr = st.sidebar.number_input("k_gr (°F per 1% green roofs)", value=-0.045, step=0.01)

coeffs = {
    "k_tree": k_tree,
    "k_imp": k_imp,
    "k_alb_0.1": k_alb,
    "k_cp": k_cp,
    "k_gr": k_gr
}

# --- Run model ---
res = predict_temp_change(
    TC_base, TC_new, IMP_base, IMP_new,
    ALB_base, ALB_new, CP_base, CP_new,
    GR_base, GR_new, Ts_base, alpha, beta, coeffs
)

# --- Main layout: results ---
col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Results")
    st.write(f"**Estimated surface temperature change:** {res['ΔT_surface']:.2f} °F")
    st.write(f"**Estimated air temperature change (scaled):** {res['ΔT_air']:.2f} °F")
    st.write(f"**New estimated peak surface temp:** {res['Ts_new']:.2f} °F")

    st.markdown("**Component contributions (°F)**")
    comp_df = pd.DataFrame.from_dict(res['components'], orient='index', columns=['ΔT (°F)'])
    comp_df['abs(ΔT)'] = comp_df['ΔT (°F)'].abs()
    comp_df = comp_df.sort_values('abs(ΔT)', ascending=False).drop(columns=['abs(ΔT)'])
    st.table(comp_df)

with col2:
    st.subheader("Quick visualization")
    labels = list(res['components'].keys())
    values = [res['components'][k] for k in labels]
    fig, ax = plt.subplots(figsize=(4,3))
    bars = ax.barh(labels, values)
    ax.set_xlabel("ΔT (°F)")
    ax.axvline(0, color='black', linewidth=0.8)
    st.pyplot(fig)

st.markdown("---")
st.subheader("Heat Score (0–100, simple normalization)")
Ts_min = st.number_input("Score min baseline temp (°F) (for normalization)", value=75.0)
Ts_max = st.number_input("Score max baseline temp (°F) (for normalization)", value=115.0)

def compute_heat_score(Ts, Ts_min, Ts_max):
    raw = 100 * (Ts - Ts_min) / (Ts_max - Ts_min)
    raw = max(0, min(100, raw))
    return raw

score_old = compute_heat_score(Ts_base, Ts_min, Ts_max)
score_new = compute_heat_score(res['Ts_new'], Ts_min, Ts_max)
st.write(f"Old Heat Score: **{score_old:.1f}**")
st.write(f"New Heat Score: **{score_new:.1f}**")
st.write(f"Score change: **{score_new - score_old:.1f}**")

# --- Optional: download CSV of results ---
download_df = pd.DataFrame({
    "metric": ["ΔT_surface", "ΔT_air", "Ts_old", "Ts_new", "heat_score_old", "heat_score_new"],
    "value": [res['ΔT_surface'], res['ΔT_air'], Ts_base, res['Ts_new'], score_old, score_new]
})
csv = download_df.to_csv(index=False).encode('utf-8')
st.download_button("Download results CSV", data=csv, file_name='heat_results.csv', mime='text/csv')
