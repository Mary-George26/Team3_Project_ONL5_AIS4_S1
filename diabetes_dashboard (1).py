"""DIABETES DASHBOARD | Exploration + classification model on diabetes_cleaned.csv

Install: pip install dash dash-bootstrap-components plotly pandas numpy scikit-learn
Run:     python diabetes_dashboard.py      (keep diabetes_cleaned.csv in the same folder)
Open:    http://127.0.0.1:8050

Structure: imports -> data -> model -> layout -> callbacks.
Tabs: 1) explore the patients  2) evaluate the model  3) check one patient
"""

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, dcc, html
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# 2. LOAD THE DATA
# ============================================================
# One row = one patient. Target: diabetes (0 = no, 1 = yes).
DATA_FILE = Path(__file__).with_name("diabetes_cleaned.csv")
if not DATA_FILE.exists():
    raise FileNotFoundError(f"Put diabetes_cleaned.csv next to this script: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)
df["result"] = df["diabetes"].map({0: "No diabetes", 1: "Diabetes"})
df["hypertension_txt"] = df["hypertension"].map({0: "No", 1: "Yes"})
df["heart_disease_txt"] = df["heart_disease"].map({0: "No", 1: "Yes"})

OVERALL_RATE = df["diabetes"].mean()

# Fixed category order so charts always read from low to high.
ORDERS = {
    "age_group": ["<18", "18-30", "31-45", "46-60", "60+"],
    "bmi_category": ["Underweight", "Normal", "Overweight", "Obese"],
    "smoking_history": ["never", "former", "current", "unknown"],
    "gender": ["Female", "Male", "Other"],
    "hypertension_txt": ["No", "Yes"],
    "heart_disease_txt": ["No", "Yes"],
}
BREAKDOWNS = {
    "age_group": "Age group",
    "bmi_category": "BMI category",
    "smoking_history": "Smoking history",
    "gender": "Gender",
    "hypertension_txt": "Hypertension",
    "heart_disease_txt": "Heart disease",
}


def dropdown_options(column, all_label):
    """'All' plus the categories of a column, in the fixed order above."""
    return [{"label": all_label, "value": "All"}] + [
        {"label": c, "value": c} for c in ORDERS[column] if c in set(df[column])
    ]


# ============================================================
# 3. TRAIN A BASIC CLASSIFICATION MODEL
# ============================================================
# risk_score and the two *_x_* columns are built from the other columns,
# so the model uses only the original clinical measurements.
FEATURES = ["age", "bmi_capped", "HbA1c_level", "blood_glucose_level",
            "hypertension", "heart_disease"]
FEATURE_LABELS = {
    "age": "Age", "bmi_capped": "BMI", "HbA1c_level": "HbA1c",
    "blood_glucose_level": "Blood glucose", "hypertension": "Hypertension",
    "heart_disease": "Heart disease",
}
X = df[FEATURES]
y = df["diabetes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# IMPORTANT: fit preprocessing ONLY on training data.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)
test_probability = model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, test_probability)
auc = roc_auc_score(y_test, test_probability)

# Always predicting "no diabetes" is right this often. Accuracy alone is misleading.
BASELINE_ACCURACY = 1 - y_test.mean()


# ============================================================
# 4. DASH APP + STYLING
# ============================================================
FONT_URL = "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap"
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY, FONT_URL])
app.title = "Diabetes risk dashboard"

INK = "#12303a"         # text, headings, header band
ACCENT = "#0f6b75"      # interface accent (tabs, sliders, ROC line)
HEALTHY = "#5fa8bd"     # no diabetes
DIABETIC = "#e0522d"    # diabetes
AMBER = "#e9a93a"       # secondary highlight for cards
MUTED = "#62767f"
LINE = "#dbe4e7"
PAGE_BG = "#eef3f4"
RESULT_COLORS = {"No diabetes": HEALTHY, "Diabetes": DIABETIC}

PAGE = {"background": PAGE_BG, "minHeight": "100vh", "color": INK,
        "fontFamily": "'IBM Plex Sans', 'Segoe UI', Arial, sans-serif",
        "padding": "28px 12px 40px"}
PANEL = {"background": "white", "border": f"1px solid {LINE}",
         "borderRadius": "10px", "padding": "18px 20px", "height": "100%"}
HEADER = {"background": INK, "borderRadius": "10px", "padding": "28px 32px",
          "marginBottom": "20px"}

app.index_string = app.index_string.replace(
    "</head>",
    f"""<style>
      body {{ background: {PAGE_BG}; }}
      .dashboard-tabs {{ border-bottom: 1px solid {LINE}; margin-bottom: 20px; }}
      .dashboard-tab {{ border: 0 !important; background: transparent !important;
                        padding: 14px 22px !important; font-weight: 500;
                        color: {MUTED} !important; }}
      .dashboard-tab--selected {{ border-bottom: 3px solid {ACCENT} !important;
                                   color: {ACCENT} !important; font-weight: 600; }}
      .Select-control {{ border-radius: 8px; border-color: {LINE}; min-height: 40px; }}
      .dash-dropdown {{ border-radius: 8px; }}
      .rc-slider-track {{ background-color: {ACCENT}; }}
      .rc-slider-handle {{ border-color: {ACCENT}; }}
      input[type=number] {{ width: 100%; padding: 7px 10px; border-radius: 6px;
                            border: 1px solid {LINE}; }}
    </style></head>""",
)


def metric_card(title, value_id, color=ACCENT, hint=None):
    """Number card. The colored left edge matches the meaning of the number."""
    children = [
        html.Div(title, style={"fontSize": "13px", "color": MUTED, "marginBottom": "6px"}),
        html.Div(id=value_id, children="-", style={
            "fontWeight": "600", "fontSize": "30px", "lineHeight": "1.1", "color": INK}),
    ]
    if hint:
        children.append(html.Div(hint, style={"fontSize": "12px", "color": MUTED, "marginTop": "6px"}))
    return html.Div(children, style={**PANEL, "borderLeft": f"5px solid {color}"})


def chart_panel(title, subtitle, graph_id, height=320):
    """Reusable chart container with a heading and a one-line description."""
    return html.Div([
        html.H5(title, style={"fontWeight": "600", "fontSize": "16px", "marginBottom": "2px"}),
        html.P(subtitle, style={"color": MUTED, "fontSize": "13px", "marginBottom": "8px"}),
        dcc.Graph(id=graph_id, style={"height": f"{height}px"},
                  config={"displayModeBar": False, "responsive": True}),
    ], style=PANEL)


def format_chart(fig):
    """One visual style shared by all Plotly Express figures."""
    fig.update_layout(
        template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
        font={"family": "IBM Plex Sans, Arial, sans-serif", "color": INK, "size": 12},
        margin={"l": 25, "r": 20, "t": 14, "b": 25},
        legend={"orientation": "h", "y": -0.22, "x": 0, "title": None},
        colorway=[HEALTHY, DIABETIC, ACCENT, "#f2b84b"],
    )
    fig.update_xaxes(showgrid=False, linecolor=LINE)
    fig.update_yaxes(gridcolor="#eaf0f2", zeroline=False)
    return fig


def empty_figure(message="No patients match these filters"):
    fig = px.scatter()
    fig.add_annotation(text=message, showarrow=False, font={"size": 14, "color": MUTED})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return format_chart(fig)


# ============================================================
# 5. DASHBOARD LAYOUT (THREE TABS)
# ============================================================
def header_stat(value, label):
    """Small fact in the header band."""
    return html.Div([
        html.Div(value, style={"fontSize": "24px", "fontWeight": "600", "color": "white"}),
        html.Div(label, style={"fontSize": "12px", "color": "#a9c3cb"}),
    ], style={"paddingLeft": "14px", "borderLeft": "1px solid rgba(255,255,255,0.3)"})


def field(label, control, md=4):
    return dbc.Col([html.Label(label, style={"fontSize": "13px", "fontWeight": "600", "color": MUTED, "marginBottom": "6px"}), control], md=md, className="mb-3")


explore_tab = html.Div([
    html.Div(dbc.Row([
        field("Gender", dcc.Dropdown(id="gender-filter", options=dropdown_options("gender", "All genders"),
                                     value="All", clearable=False)),
        field("Age group", dcc.Dropdown(id="age-filter", options=dropdown_options("age_group", "All ages"),
                                        value="All", clearable=False)),
        field("BMI category", dcc.Dropdown(id="bmi-filter", options=dropdown_options("bmi_category", "All BMI categories"),
                                           value="All", clearable=False)),
    ]), style={**PANEL, "height": "auto", "marginBottom": "16px", "paddingBottom": "6px"}),
    dbc.Row([
        dbc.Col(metric_card("Patients in view", "patients-kpi", ACCENT, "After the filters above"), lg=3, sm=6, className="mb-3"),
        dbc.Col(metric_card("Diabetes rate", "rate-kpi", DIABETIC, f"All patients: {OVERALL_RATE:.1%}"), lg=3, sm=6, className="mb-3"),
        dbc.Col(metric_card("Average HbA1c (%)", "hba1c-kpi", AMBER, "Diabetes cutoff: 6.5%"), lg=3, sm=6, className="mb-3"),
        dbc.Col(metric_card("Average blood glucose (mg/dL)", "glucose-kpi", HEALTHY, "Average across the view"), lg=3, sm=6, className="mb-3"),
    ]),
    dbc.Row([
        dbc.Col(chart_panel("Outcome balance", "Share of patients with and without diabetes", "class-pie"),
                lg=4, className="mb-3"),
        dbc.Col(chart_panel("HbA1c by outcome", "Percent of each group at every HbA1c level", "hba1c-hist"),
                lg=8, className="mb-3"),
    ]),
    dbc.Row([
        dbc.Col(chart_panel("Blood glucose vs. HbA1c", "A random sample of up to 3,000 patients from the view",
                            "glucose-scatter", height=360), lg=7, className="mb-3"),
        dbc.Col(html.Div([
            html.H5("Diabetes rate by group", style={"fontWeight": "600", "fontSize": "16px", "marginBottom": "2px"}),
            html.P("Dashed line is the rate across all patients", style={"color": MUTED, "fontSize": "13px", "marginBottom": "8px"}),
            dcc.Dropdown(id="breakdown-select",
                         options=[{"label": v, "value": k} for k, v in BREAKDOWNS.items()],
                         value="age_group", clearable=False, className="mb-2"),
            dcc.Graph(id="breakdown-bar", style={"height": "285px"},
                      config={"displayModeBar": False, "responsive": True}),
        ], style=PANEL), lg=5, className="mb-3"),
    ]),
])

model_tab = html.Div([
    html.Div(dbc.Row([
        dbc.Col([
            html.H5("Decision threshold", className="fw-bold mb-1"),
            html.P("Predict diabetes when the model's probability is at least this value. "
                   "Lower it to catch more patients; raise it to reduce false alarms.",
                   style={"fontSize": "13px", "color": MUTED, "marginBottom": 0}),
        ], lg=5),
        dbc.Col(dcc.Slider(
            id="threshold-slider", min=0.05, max=0.95, step=0.05, value=0.3,
            marks={0.05: "0.05", 0.3: "0.3", 0.5: "0.5", 0.7: "0.7", 0.95: "0.95"},
            tooltip={"placement": "bottom", "always_visible": True},
        ), lg=7, className="pt-3"),
    ], align="center"), style={**PANEL, "height": "auto", "marginBottom": "16px"}),
    dbc.Row([
        dbc.Col(metric_card("Accuracy", "accuracy-kpi", ACCENT, f"Always saying no diabetes: {BASELINE_ACCURACY:.1%}"), lg=3, sm=6, className="mb-3"),
        dbc.Col(metric_card("Precision", "precision-kpi", HEALTHY, "Predicted cases that are real"), lg=3, sm=6, className="mb-3"),
        dbc.Col(metric_card("Recall", "recall-kpi", DIABETIC, "Diabetic patients found"), lg=3, sm=6, className="mb-3"),
        dbc.Col(metric_card("F1 score", "f1-kpi", AMBER, "Balance of precision and recall"), lg=3, sm=6, className="mb-3"),
    ]),
    html.P([
        f"Metrics use the held-out test set ({len(y_test):,} patients). ROC-AUC is {auc:.3f} and does not "
        f"depend on the threshold. Only {y.mean():.1%} of patients have diabetes, so always predicting "
        f"'no diabetes' would already reach {BASELINE_ACCURACY:.1%} accuracy. Judge the model by recall "
        f"(diabetic patients found) and precision (predicted cases that are real)."],
        style={"color": MUTED, "fontSize": "13px", "maxWidth": "900px"}),
    dbc.Row([
        dbc.Col(chart_panel("Confusion matrix", "Actual versus predicted class", "confusion-heatmap"),
                lg=5, className="mb-3"),
        dbc.Col(chart_panel("ROC curve", "The dot marks the selected threshold", "roc-line"),
                lg=7, className="mb-3"),
    ]),
    dbc.Row([
        dbc.Col(chart_panel("Predicted probabilities", "Percent of each actual group at every probability",
                            "prob-hist"), lg=7, className="mb-3"),
        dbc.Col(chart_panel("What drives the prediction", "Standardized coefficients; larger means stronger push toward diabetes",
                            "importance-bar"), lg=5, className="mb-3"),
    ]),
])

check_tab = dbc.Row([
    dbc.Col(html.Div([
        html.H5("Patient details", style={"fontWeight": "600", "fontSize": "16px", "marginBottom": "14px"}),
        dbc.Row([
            field("Age (years)", dcc.Input(id="calc-age", type="number", value=45, min=0, max=100, step=1), md=6),
            field("BMI", dcc.Input(id="calc-bmi", type="number", value=27, min=10, max=60, step=0.1), md=6),
            field("HbA1c (%)", dcc.Input(id="calc-hba1c", type="number", value=5.5, min=3, max=10, step=0.1), md=6),
            field("Blood glucose (mg/dL)", dcc.Input(id="calc-glucose", type="number", value=140, min=50, max=400, step=1), md=6),
            field("Hypertension", dcc.Dropdown(id="calc-hyp", options=[{"label": "No", "value": 0}, {"label": "Yes", "value": 1}],
                                                value=0, clearable=False), md=6),
            field("Heart disease", dcc.Dropdown(id="calc-heart", options=[{"label": "No", "value": 0}, {"label": "Yes", "value": 1}],
                                                 value=0, clearable=False), md=6),
        ]),
    ], style=PANEL), lg=5, className="mb-3"),
    dbc.Col(html.Div([
        html.H5("Estimated chance of diabetes", style={"fontWeight": "600", "fontSize": "16px", "marginBottom": "10px"}),
        html.Div(id="calc-prob", style={"fontSize": "56px", "fontWeight": "600", "lineHeight": "1.1"}),
        dbc.Progress(id="calc-progress", value=0, style={"height": "12px", "margin": "14px 0"}),
        html.P(id="calc-msg", style={"fontSize": "15px"}),
        html.P("This is a classroom model trained on one dataset. It is not medical advice and cannot diagnose anyone.",
               style={"color": MUTED, "fontSize": "13px", "marginTop": "24px", "marginBottom": 0}),
    ], style=PANEL), lg=7, className="mb-3"),
])

app.layout = html.Div([
    dbc.Container([
        html.Div(dbc.Row([
            dbc.Col([
                html.H1("Diabetes risk dashboard", style={
                    "fontWeight": "600", "fontSize": "clamp(26px, 4vw, 36px)", "margin": "0 0 8px", "color": "white"}),
                html.P("Explore the patient records, then see how well a logistic regression model "
                       "estimates diabetes risk from age, BMI, HbA1c, blood glucose, hypertension and heart disease.",
                       style={"color": "#c4d6db", "fontSize": "15px", "maxWidth": "560px", "margin": 0}),
            ], lg=7),
            dbc.Col(dbc.Row([
                dbc.Col(header_stat(f"{len(df):,}", "patient records"), xs=4),
                dbc.Col(header_stat(f"{OVERALL_RATE:.1%}", "have diabetes"), xs=4),
                dbc.Col(header_stat(f"{auc:.2f}", "model ROC-AUC"), xs=4),
            ]), lg=5, className="mt-4 mt-lg-0"),
        ], align="center"), style=HEADER),
        dcc.Tabs(
            id="dashboard-tabs", value="explore", className="dashboard-tabs", children=[
                dcc.Tab(label="Explore patients", value="explore", children=explore_tab,
                        className="dashboard-tab", selected_className="dashboard-tab--selected"),
                dcc.Tab(label="Evaluate the model", value="evaluate", children=model_tab,
                        className="dashboard-tab", selected_className="dashboard-tab--selected"),
                dcc.Tab(label="Check a patient", value="check", children=check_tab,
                        className="dashboard-tab", selected_className="dashboard-tab--selected"),
            ],
        ),
    ], style={"maxWidth": "1260px"}, fluid=True),
], style=PAGE)


# ============================================================
# 6. CALLBACK 1: UPDATE THE EXPLORE TAB
# ============================================================
@app.callback(
    Output("patients-kpi", "children"),
    Output("rate-kpi", "children"),
    Output("hba1c-kpi", "children"),
    Output("glucose-kpi", "children"),
    Output("class-pie", "figure"),
    Output("hba1c-hist", "figure"),
    Output("glucose-scatter", "figure"),
    Output("breakdown-bar", "figure"),
    Input("gender-filter", "value"),
    Input("age-filter", "value"),
    Input("bmi-filter", "value"),
    Input("breakdown-select", "value"),
)
def update_explore(gender, age_group, bmi_category, breakdown):
    filtered = df
    if gender != "All":
        filtered = filtered[filtered["gender"] == gender]
    if age_group != "All":
        filtered = filtered[filtered["age_group"] == age_group]
    if bmi_category != "All":
        filtered = filtered[filtered["bmi_category"] == bmi_category]

    if filtered.empty:
        blank = empty_figure()
        return "0", "-", "-", "-", blank, blank, blank, blank

    patients = f"{len(filtered):,}"
    rate = f"{filtered['diabetes'].mean():.1%}"
    hba1c = f"{filtered['HbA1c_level'].mean():.2f}"
    glucose = f"{filtered['blood_glucose_level'].mean():.0f}"

    pie_fig = px.pie(
        filtered, names="result", hole=0.58, color="result",
        color_discrete_map=RESULT_COLORS,
        category_orders={"result": ["No diabetes", "Diabetes"]},
    )
    pie_fig.update_traces(textinfo="percent+label", textposition="outside",
                          marker={"line": {"color": "white", "width": 2}})
    pie_fig.update_layout(showlegend=False)

    hist_fig = px.histogram(
        filtered, x="HbA1c_level", color="result", nbins=30, barmode="overlay",
        histnorm="percent", opacity=0.75, color_discrete_map=RESULT_COLORS,
        category_orders={"result": ["No diabetes", "Diabetes"]},
        labels={"HbA1c_level": "HbA1c level (%)", "result": "Outcome"},
    )
    hist_fig.update_yaxes(title="Percent of group")

    # 96,000 dots would be slow and unreadable, so plot a sample (diabetes drawn last).
    sample = filtered.sample(min(len(filtered), 3000), random_state=1).sort_values("diabetes")
    scatter_fig = px.scatter(
        sample, x="HbA1c_level", y="blood_glucose_level", color="result",
        color_discrete_map=RESULT_COLORS, opacity=0.55,
        hover_data=["age", "bmi"],
        labels={"HbA1c_level": "HbA1c level (%)", "blood_glucose_level": "Blood glucose (mg/dL)",
                "result": "Outcome"},
    )

    groups = (filtered.groupby(breakdown, observed=True)["diabetes"]
              .agg(rate="mean", patients="size")
              .reindex([c for c in ORDERS[breakdown]])
              .dropna().reset_index())
    groups["rate"] = groups["rate"] * 100
    bar_fig = px.bar(
        groups, x=breakdown, y="rate", text=groups["rate"].map("{:.1f}%".format),
        hover_data={"patients": True, "rate": ":.1f"},
        color="rate", color_continuous_scale=[[0, "#f5cdbd"], [1, DIABETIC]],
        labels={breakdown: BREAKDOWNS[breakdown], "rate": "Diabetes rate (%)"},
    )
    bar_fig.update_traces(textposition="outside", cliponaxis=False)
    bar_fig.update_layout(coloraxis_showscale=False)
    bar_fig.add_hline(y=OVERALL_RATE * 100, line_dash="dash", line_color=INK,
                      annotation_text="All patients", annotation_position="top left",
                      annotation_font_color=MUTED)
    bar_fig.update_yaxes(title="Diabetes rate (%)", rangemode="tozero")

    return (patients, rate, hba1c, glucose,
            format_chart(pie_fig), format_chart(hist_fig),
            format_chart(scatter_fig), format_chart(bar_fig))


# ============================================================
# 7. CALLBACK 2: UPDATE THE MODEL TAB
# ============================================================
@app.callback(
    Output("accuracy-kpi", "children"),
    Output("precision-kpi", "children"),
    Output("recall-kpi", "children"),
    Output("f1-kpi", "children"),
    Output("confusion-heatmap", "figure"),
    Output("roc-line", "figure"),
    Output("prob-hist", "figure"),
    Output("importance-bar", "figure"),
    Input("threshold-slider", "value"),
)
def update_model(threshold):
    predictions = (test_probability >= threshold).astype(int)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    labels = ["No diabetes", "Diabetes"]
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    confusion_fig = px.imshow(
        matrix, x=labels, y=labels, text_auto=",", aspect="auto",
        color_continuous_scale=[[0, "#eef5f6"], [1, ACCENT]],
        labels={"x": "Predicted", "y": "Actual", "color": "Patients"},
    )
    confusion_fig.update_traces(textfont={"size": 20})
    confusion_fig.update_layout(coloraxis_showscale=False)

    # Point on the ROC curve that matches the chosen threshold.
    tn, fp, fn, tp = matrix.ravel()
    point_fpr = fp / (fp + tn) if (fp + tn) else 0
    point_tpr = tp / (tp + fn) if (tp + fn) else 0

    roc_fig = px.line(x=fpr, y=tpr, labels={"x": "False positive rate", "y": "True positive rate"})
    roc_fig.update_traces(line={"color": ACCENT, "width": 3}, name=f"ROC-AUC {auc:.3f}", showlegend=True)
    roc_fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                        line={"color": "#b3c0c7", "dash": "dash"})
    roc_fig.add_scatter(x=[point_fpr], y=[point_tpr], mode="markers",
                        name=f"Threshold {threshold:.2f}",
                        marker={"color": DIABETIC, "size": 12})
    roc_fig.update_xaxes(range=[0, 1])
    roc_fig.update_yaxes(range=[0, 1.03])

    # Histogram by ACTUAL label; percent per group keeps the small group visible.
    test_df = pd.DataFrame({
        "Predicted probability": test_probability,
        "Actual outcome": y_test.map({0: "No diabetes", 1: "Diabetes"}).to_numpy(),
    })
    prob_fig = px.histogram(
        test_df, x="Predicted probability", color="Actual outcome", barmode="overlay",
        histnorm="percent", opacity=0.75, nbins=25, color_discrete_map=RESULT_COLORS,
        category_orders={"Actual outcome": ["No diabetes", "Diabetes"]},
    )
    prob_fig.update_yaxes(title="Percent of group")
    prob_fig.add_vline(x=threshold, line_dash="dash", line_color=INK,
                       annotation_text=f"Threshold {threshold:.2f}", annotation_position="top")

    coefs = pd.DataFrame({
        "Feature": [FEATURE_LABELS[f] for f in FEATURES],
        "Coefficient": model.coef_[0],
    }).sort_values("Coefficient")
    importance_fig = px.bar(coefs, x="Coefficient", y="Feature", orientation="h",
                            color_discrete_sequence=[ACCENT])
    importance_fig.update_yaxes(title=None)

    return (f"{accuracy:.1%}", f"{precision:.1%}", f"{recall:.1%}", f"{f1:.1%}",
            format_chart(confusion_fig), format_chart(roc_fig),
            format_chart(prob_fig), format_chart(importance_fig))


# ============================================================
# 8. CALLBACK 3: CHECK ONE PATIENT
# ============================================================
@app.callback(
    Output("calc-prob", "children"),
    Output("calc-progress", "value"),
    Output("calc-progress", "color"),
    Output("calc-msg", "children"),
    Input("calc-age", "value"),
    Input("calc-bmi", "value"),
    Input("calc-hba1c", "value"),
    Input("calc-glucose", "value"),
    Input("calc-hyp", "value"),
    Input("calc-heart", "value"),
)
def check_patient(age, bmi, hba1c, glucose, hypertension, heart_disease):
    values = [age, bmi, hba1c, glucose, hypertension, heart_disease]
    if any(v is None for v in values):
        return "-", 0, "secondary", "Fill in every field to see the estimate."

    # Keep BMI inside the range the model was trained on (the data uses a capped BMI).
    bmi = float(np.clip(bmi, df["bmi_capped"].min(), df["bmi_capped"].max()))
    patient = pd.DataFrame([[age, bmi, hba1c, glucose, hypertension, heart_disease]], columns=FEATURES)
    probability = float(model.predict_proba(scaler.transform(patient))[0, 1])

    ratio = probability / OVERALL_RATE
    if ratio >= 1.05:
        comparison = f"{ratio:.1f} times the average of {OVERALL_RATE:.1%} across all patients."
    elif ratio <= 0.95:
        comparison = f"Lower than the average of {OVERALL_RATE:.1%} across all patients."
    else:
        comparison = f"About the same as the average of {OVERALL_RATE:.1%} across all patients."

    color = "danger" if probability >= 0.3 else "warning" if probability >= 0.1 else "success"
    return f"{probability:.1%}", round(probability * 100, 1), color, comparison


# ============================================================
# 9. RUN THE APP
# ============================================================
if __name__ == "__main__":
    app.run(debug=True, port=8050)
