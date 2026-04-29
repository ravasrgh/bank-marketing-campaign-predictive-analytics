"""
BSM Bank Marketing Campaign – Predictive Analytics Dashboard
==============================================================
Professional Streamlit application for Bank BSM Telemarketing Team.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils import (
    load_model, load_data, prepare_input, predict_with_threshold,
    get_custom_css, CATEGORY_OPTIONS, FEATURE_LABELS, ALL_FEATURES,
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, COLORS,
)

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="BSM Bank – Campaign Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(get_custom_css(), unsafe_allow_html=True)

# ── Load resources ──
model, metadata = load_model()
df_raw = load_data()

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## BSM Analytics")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["Project Context", "Interactive Dashboard", "Customer Predictor"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#94A3B8;font-size:0.78rem;'>"
        "LightGBM + ROS Model v1<br>© 2026 Group Beta</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════
#  PAGE 1 – PROJECT CONTEXT
# ══════════════════════════════════════════════════════════════
def page_context():
    # Hero
    st.markdown(
        '<div class="hero-container">'
        '<div class="hero-title">Bank Marketing Campaign Predictor</div>'
        '<div class="hero-subtitle">Intelligent Decision Support for BSM Telemarketing Team</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    # KPI row
    if metadata:
        bi = metadata["business_impact"]
        mt = metadata["metrics_test"]
        cols = st.columns(4)
        kpis = [
            (f"Rp {bi['net_savings_test_rp']/1e6:.0f} Jt", "Net Savings"),
            (f"{bi['roi_pct']}%", "ROI"),
            (f"{bi['calls_reduction_pct']}%", "Calls Reduced"),
            (f"{mt['f2_score']:.4f}", "F2-Score"),
        ]
        for col, (val, lbl) in zip(cols, kpis):
            col.markdown(
                f'<div class="metric-card"><div class="metric-value">{val}</div>'
                f'<div class="metric-label">{lbl}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("")

    # Context cards
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Business Objective</div>', unsafe_allow_html=True)
        st.markdown(
            """
            PT Bank Sentral Merakyat (BSM) runs telemarketing campaigns to sell
            **term deposit** products. The current approach — calling customers
            randomly — wastes agent time and misses high-potential leads.

            This application uses a **LightGBM machine-learning model** to score
            every customer's probability of subscribing, so the team can
            **prioritize calls** and maximize conversion while cutting costs.
            """
        )
        st.markdown('<div class="section-header">Target Users</div>', unsafe_allow_html=True)
        st.markdown(
            """
            | Role | How They Use This App |
            |---|---|
            | **Campaign Manager** | Filter & rank the contact list before each campaign wave |
            | **Telemarketing Agent** | Focus calls on *Priority* customers first |
            | **Marketing Analyst** | Explore conversion drivers via the dashboard |
            """
        )

    with c2:
        st.markdown('<div class="section-header">How It Works</div>', unsafe_allow_html=True)
        st.markdown(
            """
            1. **Upload** your customer list (CSV) or enter a single customer profile.
            2. The model calculates a **subscription probability** (0–100 %).
            3. Customers above the optimal threshold are flagged **Priority** (call them);
               those below are **Non-Priority** (skip or deprioritise).
            """
        )
        if metadata:
            st.markdown(
                f'<div style="background:rgba(37,99,235,0.15);border:1px solid rgba(59,130,246,0.35);'
                f'border-radius:12px;padding:1rem 1.25rem;margin-top:0.75rem;">'
                f'<p style="color:#E2E8F0;margin:0;"><strong>Optimal F2 threshold:</strong> {metadata["threshold_f2"]:.2f}<br>'
                f'<strong>Profit-optimal threshold:</strong> {metadata["threshold_profit"]:.2f}<br>'
                f'The app uses the <strong>F2 threshold</strong> by default to maximise recall.</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-header">Cost of Errors</div>', unsafe_allow_html=True)
        st.markdown(
            """
            | Error | Cost | Risk |
            |---|---|---|
            | **False Negative** – miss a willing customer | Rp 500,000 lost margin | Critical |
            | **False Positive** – call an uninterested one | Rp 50,000 wasted call | Moderate |

            > *FN : FP cost ratio = **10 : 1** → minimising False Negatives is priority #1.*
            """
        )


# ══════════════════════════════════════════════════════════════
#  PAGE 2 – INTERACTIVE DASHBOARD
# ══════════════════════════════════════════════════════════════
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#CBD5E1"),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)


def page_dashboard():
    st.markdown(
        '<div class="hero-container">'
        '<div class="hero-title">Interactive Dashboard</div>'
        '<div class="hero-subtitle">Explore conversion drivers in the BSM customer dataset</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    if df_raw is None:
        st.error("Dataset not available.")
        return

    df = df_raw.copy()
    df["Subscribed"] = df["target"].map({1: "Yes", 0: "No"})

    # ── Row 1: Overview KPIs ──
    total = len(df)
    subs = df["target"].sum()
    rate = subs / total * 100
    k1, k2, k3, k4 = st.columns(4)
    for col, (v, l) in zip(
        [k1, k2, k3, k4],
        [
            (f"{total:,}", "Total Customers"),
            (f"{subs:,}", "Subscribers"),
            (f"{total - subs:,}", "Non-Subscribers"),
            (f"{rate:.1f}%", "Conversion Rate"),
        ],
    ):
        col.markdown(
            f'<div class="metric-card"><div class="metric-value">{v}</div>'
            f'<div class="metric-label">{l}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── Chart 1: Conversion Rate by Job ──
    st.markdown('<div class="section-header">Conversion Rate by Job Type</div>', unsafe_allow_html=True)
    job_stats = (
        df.groupby("job")["target"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "conversion_rate", "count": "n_customers"})
        .sort_values("conversion_rate", ascending=True)
    )
    job_stats["conversion_pct"] = job_stats["conversion_rate"] * 100

    fig_job = px.bar(
        job_stats, y="job", x="conversion_pct",
        orientation="h",
        color="conversion_pct",
        color_continuous_scale=["#1E3A5F", "#2563EB", "#06B6D4", "#10B981"],
        labels={"conversion_pct": "Conversion Rate (%)", "job": "Job Type"},
        title="",
        hover_data={"n_customers": True},
    )
    fig_job.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False, height=420)
    fig_job.update_traces(
        hovertemplate="<b>%{y}</b><br>Rate: %{x:.1f}%<br>Customers: %{customdata[0]:,}<extra></extra>"
    )
    st.plotly_chart(fig_job, use_container_width=True)

    # ── Chart 2 & 3 side by side ──
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-header">Age Distribution vs Conversion</div>', unsafe_allow_html=True)
        fig_age = px.histogram(
            df, x="age", color="Subscribed", barmode="overlay",
            nbins=40, opacity=0.75,
            color_discrete_map={"Yes": "#10B981", "No": "#3B82F6"},
            labels={"age": "Customer Age", "count": "Count"},
        )
        age_layout = {**PLOT_LAYOUT, "legend": {**PLOT_LAYOUT.get("legend", {}), "x": 0.75, "y": 0.95}}
        fig_age.update_layout(**age_layout, height=400, bargap=0.05)
        st.plotly_chart(fig_age, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">Contact Type & Conversion</div>', unsafe_allow_html=True)
        contact_stats = (
            df.groupby("contact")["target"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "rate", "count": "n"})
        )
        contact_stats["pct"] = contact_stats["rate"] * 100
        fig_contact = px.bar(
            contact_stats, x="contact", y="pct",
            color="contact",
            color_discrete_map={"cellular": "#10B981", "telephone": "#3B82F6"},
            text=contact_stats["pct"].apply(lambda x: f"{x:.1f}%"),
            labels={"pct": "Conversion Rate (%)", "contact": "Contact Type"},
        )
        fig_contact.update_traces(textposition="outside")
        fig_contact.update_layout(**PLOT_LAYOUT, height=400, showlegend=False)
        st.plotly_chart(fig_contact, use_container_width=True)

    # ── Chart 4: Previous Outcome ──
    st.markdown('<div class="section-header">Previous Campaign Outcome vs Conversion</div>', unsafe_allow_html=True)
    pout_stats = (
        df.groupby("poutcome")["target"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "rate", "count": "n"})
    )
    pout_stats["pct"] = pout_stats["rate"] * 100
    fig_pout = px.bar(
        pout_stats, x="poutcome", y="pct",
        color="pct",
        color_continuous_scale=["#1E3A5F", "#2563EB", "#10B981"],
        text=pout_stats["pct"].apply(lambda x: f"{x:.1f}%"),
        labels={"pct": "Conversion Rate (%)", "poutcome": "Previous Outcome"},
    )
    fig_pout.update_traces(textposition="outside")
    fig_pout.update_layout(**PLOT_LAYOUT, height=380, coloraxis_showscale=False)
    st.plotly_chart(fig_pout, use_container_width=True)

    # ── Chart 5: Macro indicator ──
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-header">Euribor 3M Rate vs Subscription</div>', unsafe_allow_html=True)
        fig_eur = px.box(
            df, x="Subscribed", y="euribor3m",
            color="Subscribed",
            color_discrete_map={"Yes": "#10B981", "No": "#3B82F6"},
            labels={"euribor3m": "Euribor 3-Month Rate"},
        )
        fig_eur.update_layout(**PLOT_LAYOUT, height=400, showlegend=False)
        st.plotly_chart(fig_eur, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Monthly Conversion Rates</div>', unsafe_allow_html=True)
        month_order = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
        month_stats = (
            df.groupby("month")["target"].agg(["mean","count"]).reset_index()
            .rename(columns={"mean":"rate","count":"n"})
        )
        month_stats["month"] = pd.Categorical(month_stats["month"], categories=month_order, ordered=True)
        month_stats = month_stats.sort_values("month")
        month_stats["pct"] = month_stats["rate"] * 100

        fig_month = px.bar(
            month_stats, x="month", y="pct",
            color="pct",
            color_continuous_scale=["#1E3A5F", "#2563EB", "#06B6D4", "#10B981"],
            text=month_stats["pct"].apply(lambda x: f"{x:.0f}%"),
            labels={"pct": "Conversion Rate (%)", "month": "Month"},
        )
        fig_month.update_traces(textposition="outside")
        fig_month.update_layout(**PLOT_LAYOUT, height=400, coloraxis_showscale=False)
        st.plotly_chart(fig_month, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 3 – CUSTOMER PREDICTOR
# ══════════════════════════════════════════════════════════════
def page_predictor():
    st.markdown(
        '<div class="hero-container">'
        '<div class="hero-title">Customer Predictor</div>'
        '<div class="hero-subtitle">Classify customers as Priority or Non-Priority for your campaign</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    if model is None or metadata is None:
        st.error("Model could not be loaded. Please check the .pkl and .json files.")
        return

    threshold = metadata["threshold_f2"]

    tab_single, tab_batch = st.tabs(["Single Prediction", "Batch Prediction"])

    # ────────────────────────────────────────────
    # TAB 1 – Single Prediction
    # ────────────────────────────────────────────
    with tab_single:
        st.markdown('<div class="section-header">Customer Profile</div>', unsafe_allow_html=True)

        with st.form("single_pred_form"):
            st.markdown("##### Demographics")
            d1, d2, d3, d4 = st.columns(4)
            age = d1.number_input("Age", 17, 100, 35, key="age_input")
            job = d2.selectbox("Job", CATEGORY_OPTIONS["job"], key="job_input")
            marital = d3.selectbox("Marital Status", CATEGORY_OPTIONS["marital"], key="marital_input")
            education = d4.selectbox("Education", CATEGORY_OPTIONS["education"], index=6, key="edu_input")

            st.markdown("##### Financial Status")
            f1, f2, f3 = st.columns(3)
            default = f1.selectbox("Credit Default?", CATEGORY_OPTIONS["default"], key="def_input")
            housing = f2.selectbox("Housing Loan?", CATEGORY_OPTIONS["housing"], key="house_input")
            loan = f3.selectbox("Personal Loan?", CATEGORY_OPTIONS["loan"], key="loan_input")

            st.markdown("##### Campaign Info")
            c1, c2, c3, c4 = st.columns(4)
            contact = c1.selectbox("Contact Type", CATEGORY_OPTIONS["contact"], key="contact_input")
            month = c2.selectbox("Month", CATEGORY_OPTIONS["month"], index=4, key="month_input")
            day_of_week = c3.selectbox("Day of Week", CATEGORY_OPTIONS["day_of_week"], key="dow_input")
            poutcome = c4.selectbox("Prev. Outcome", CATEGORY_OPTIONS["poutcome"], index=1, key="pout_input")

            st.markdown("##### History & Macro Indicators")
            m1, m2, m3, m4 = st.columns(4)
            previous = m1.number_input("Previous Contacts", 0, 50, 0, key="prev_input")
            was_contacted = m2.selectbox("Contacted Before?", [0, 1], format_func=lambda x: "Yes" if x else "No", key="wc_input")
            emp_var = m3.number_input("Emp. Var. Rate", -4.0, 2.0, 1.1, step=0.1, key="emp_input")
            cons_price = m4.number_input("Consumer Price Idx", 92.0, 95.0, 93.994, step=0.01, format="%.3f", key="cpi_input")

            m5, m6, m7 = st.columns(3)
            cons_conf = m5.number_input("Consumer Conf. Idx", -51.0, -26.0, -36.4, step=0.1, key="cci_input")
            euribor = m6.number_input("Euribor 3M", 0.5, 5.1, 4.857, step=0.01, key="eur_input")
            nr_emp = m7.number_input("Nr. Employed", 4960.0, 5230.0, 5191.0, step=0.1, format="%.1f", key="nre_input")

            submitted = st.form_submit_button("Predict", use_container_width=True)

        if submitted:
            input_df = pd.DataFrame([{
                "education": education, "job": job, "marital": marital,
                "default": default, "housing": housing, "loan": loan,
                "contact": contact, "month": month, "day_of_week": day_of_week,
                "poutcome": poutcome, "age": age, "previous": previous,
                "emp.var.rate": emp_var, "cons.price.idx": cons_price,
                "cons.conf.idx": cons_conf, "euribor3m": euribor,
                "nr.employed": nr_emp, "was_contacted_before": was_contacted,
            }])

            try:
                probs, preds = predict_with_threshold(model, input_df[ALL_FEATURES], threshold)
                prob = probs[0]
                pred = preds[0]
                is_priority = pred == 1

                st.markdown("")
                r1, r2 = st.columns([1, 1])
                with r1:
                    cls = "pred-priority" if is_priority else "pred-nonpriority"
                    label = "PRIORITY — Call This Customer" if is_priority else "NON-PRIORITY — Skip / Deprioritise"
                    color = "#10B981" if is_priority else "#EF4444"
                    st.markdown(
                        f'<div class="pred-card {cls}">'
                        f'<div class="pred-label" style="color:{color}">{label}</div>'
                        f'<div class="pred-prob">Subscription Probability: <b>{prob:.1%}</b></div>'
                        f'<div class="pred-prob">Threshold: {threshold:.2f}</div>'
                        "</div>",
                        unsafe_allow_html=True,
                    )

                with r2:
                    # Probability gauge
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        number={"suffix": "%", "font": {"size": 42, "color": "#E2E8F0"}},
                        gauge=dict(
                            axis=dict(range=[0, 100], tickcolor="#64748B"),
                            bar=dict(color="#3B82F6"),
                            bgcolor="rgba(30,41,59,0.5)",
                            steps=[
                                dict(range=[0, threshold * 100], color="rgba(239,68,68,0.15)"),
                                dict(range=[threshold * 100, 100], color="rgba(16,185,129,0.15)"),
                            ],
                            threshold=dict(
                                line=dict(color="#F59E0B", width=3),
                                thickness=0.8,
                                value=threshold * 100,
                            ),
                        ),
                        title=dict(text="Subscription Probability", font=dict(size=16, color="#94A3B8")),
                    ))
                    fig_gauge.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        height=280,
                        margin=dict(l=30, r=30, t=60, b=10),
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")

    # ────────────────────────────────────────────
    # TAB 2 – Batch Prediction
    # ────────────────────────────────────────────
    with tab_batch:
        st.markdown('<div class="section-header">Upload Customer CSV</div>', unsafe_allow_html=True)
        st.markdown(
            "Upload a CSV file containing customer data. Required columns: "
            f"`{'`, `'.join(CATEGORICAL_FEATURES + [c for c in NUMERIC_FEATURES if c != 'was_contacted_before'])}`. "
            "The column `was_contacted_before` will be auto-generated from `pdays` if missing."
        )

        uploaded = st.file_uploader("Choose CSV file", type=["csv"], key="batch_upload")

        # Threshold selector
        th_col1, th_col2 = st.columns(2)
        with th_col1:
            th_choice = st.radio(
                "Threshold Strategy",
                ["F2-Optimized", "Profit-Optimized"],
                horizontal=True,
                key="th_choice",
            )
        batch_threshold = (
            metadata["threshold_f2"] if th_choice == "F2-Optimized" else metadata["threshold_profit"]
        )
        with th_col2:
            st.metric("Active Threshold", f"{batch_threshold:.2f}")

        if uploaded is not None:
            try:
                df_upload = pd.read_csv(uploaded)
                st.success(f"Loaded **{len(df_upload):,}** rows x **{len(df_upload.columns)}** columns")

                with st.expander("Preview uploaded data", expanded=False):
                    st.dataframe(df_upload.head(10), use_container_width=True)

                # Prepare & predict
                df_pred = prepare_input(df_upload)
                probs, preds = predict_with_threshold(model, df_pred, batch_threshold)

                # Build result table
                result = df_upload.copy()
                result["probability"] = probs
                result["prediction"] = preds
                result["recommendation"] = result["prediction"].map(
                    {1: "Priority", 0: "Non-Priority"}
                )
                result = result.sort_values("probability", ascending=False)

                # Summary metrics
                n_priority = (preds == 1).sum()
                n_total = len(preds)
                st.markdown('<div class="section-header">Results Summary</div>', unsafe_allow_html=True)
                s1, s2, s3, s4 = st.columns(4)
                for col, (v, l) in zip(
                    [s1, s2, s3, s4],
                    [
                        (f"{n_total:,}", "Total Customers"),
                        (f"{n_priority:,}", "Priority (Call)"),
                        (f"{n_total - n_priority:,}", "Non-Priority (Skip)"),
                        (f"{n_priority / n_total:.1%}", "Call Rate"),
                    ],
                ):
                    col.markdown(
                        f'<div class="metric-card"><div class="metric-value">{v}</div>'
                        f'<div class="metric-label">{l}</div></div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("")

                # Probability distribution
                fig_dist = px.histogram(
                    result, x="probability", color="recommendation",
                    nbins=40, barmode="overlay", opacity=0.7,
                    color_discrete_map={"Priority": "#10B981", "Non-Priority": "#3B82F6"},
                    labels={"probability": "Subscription Probability", "recommendation": ""},
                )
                fig_dist.add_vline(
                    x=batch_threshold, line_dash="dash", line_color="#F59E0B",
                    annotation_text=f"Threshold = {batch_threshold:.2f}",
                    annotation_font_color="#F59E0B",
                )
                fig_dist.update_layout(**PLOT_LAYOUT, height=350)
                st.plotly_chart(fig_dist, use_container_width=True)

                # Results table
                st.markdown('<div class="section-header">Detailed Results</div>', unsafe_allow_html=True)
                display_cols = [c for c in result.columns if c not in ("target", "y")]
                st.dataframe(
                    result[display_cols].style.format({"probability": "{:.2%}"}),
                    use_container_width=True,
                    height=400,
                )

                # Download
                csv_out = result[display_cols].to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Prediction Results",
                    data=csv_out,
                    file_name="bsm_prediction_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            except ValueError as ve:
                st.error(f"Data validation error: {ve}")
            except Exception as e:
                st.error(f"Processing error: {e}")


# ══════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════
if page == "Project Context":
    page_context()
elif page == "Interactive Dashboard":
    page_dashboard()
else:
    page_predictor()
