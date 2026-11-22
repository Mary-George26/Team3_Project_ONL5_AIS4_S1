"""
HEALTHCARE PREDICTIVE ANALYTICS DASHBOARD
Multi-Disease Healthcare Analytics | Heart Disease, Diabetes & Chronic Kidney Disease

Open:
    http://127.0.0.1:8050/

Project Overview:
    An interactive healthcare analytics dashboard that brings
    together data exploration and visualization for multiple
    diseases, including Cardiovascular Disease, Diabetes, and
    Chronic Kidney Disease.

    The dashboard provides disease-specific insights through
    interactive filters, statistical summaries, and visualizations.
    Separate classification models are implemented for selected
    diseases to support predictive analysis.

Technologies:
    Python, Pandas, NumPy, Scikit-learn,
    Dash, Dash Bootstrap Components, and Plotly.

The application is organized into sections:
    Imports -> Data Loading -> Preprocessing -> Model Training
    -> App Setup -> Layout -> Callbacks -> Run App.

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
# 2. LOAD DATA
# ============================================================

# --- Dataset 1: Heart Disease ---
HEART_FILE = Path(__file__).with_name("cardio_cleaned.csv")
if not HEART_FILE.exists():
    raise FileNotFoundError(f"Put cardio_cleaned.csv next to this script: {HEART_FILE}")

df_heart = pd.read_csv(HEART_FILE)

# --- Dataset 2: Diabetes ---
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

# --- Dataset 3: Chronic Kidney Disease ---
KIDNEY_FILE = Path(__file__).with_name("kidney_cleaned.csv")
if not KIDNEY_FILE.exists():
    raise FileNotFoundError(f"Put kidney_cleaned.csv next to this script: {KIDNEY_FILE}")

df_kidney = pd.read_csv(KIDNEY_FILE)

# ============================================================
# 3. TRAIN A BASIC CLASSIFICATION MODEL
# ============================================================
# --- Dataset 1: Heart Disease ---
HEART_FEATURES = ["age_years", "gender", "cholesterol", "ap_hi", "ap_lo"]
X_heart = df_heart[HEART_FEATURES]
y_heart = df_heart["cardio"]

X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_heart, y_heart, test_size=0.25, random_state=42, stratify=y_heart
)

scaler_heart = StandardScaler()
X_train_h_scaled = scaler_heart.fit_transform(X_train_h)
X_test_h_scaled = scaler_heart.transform(X_test_h)

model_heart = LogisticRegression(max_iter=1000, random_state=42)
model_heart.fit(X_train_h_scaled, y_train_h)

heart_prob = model_heart.predict_proba(X_test_h_scaled)[:, 1]
fpr_h, tpr_h, _ = roc_curve(y_test_h, heart_prob)
auc_heart = roc_auc_score(y_test_h, heart_prob)

# --- Dataset 2: Diabetes ---
FEATURES_DIAB = ["age", "bmi_capped", "HbA1c_level", "blood_glucose_level",
                 "hypertension", "heart_disease"]

FEATURE_LABELS = {
    "age": "Age", 
    "bmi_capped": "BMI", 
    "HbA1c_level": "HbA1c",
    "blood_glucose_level": "Blood glucose", 
    "hypertension": "Hypertension",
    "heart_disease": "Heart disease",
}

X_diab = df[FEATURES_DIAB]
y_diab = df["diabetes"]

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_diab, y_diab, test_size=0.25, random_state=42, stratify=y_diab
)

# IMPORTANT: fit preprocessing ONLY on training data.
scaler_diab = StandardScaler()
X_train_d_scaled = scaler_diab.fit_transform(X_train_d)
X_test_d_scaled = scaler_diab.transform(X_test_d)

model_diab = LogisticRegression(max_iter=1000, random_state=42)
model_diab.fit(X_train_d_scaled, y_train_d)

test_probability_diab = model_diab.predict_proba(X_test_d_scaled)[:, 1]
fpr_diab, tpr_diab, _ = roc_curve(y_test_d, test_probability_diab)
auc_diab = roc_auc_score(y_test_d, test_probability_diab)

# Always predicting "no diabetes" is right this often. Accuracy alone is misleading.
BASELINE_ACCURACY = 1 - y_test_d.mean()

# --- Dataset 3: Chronic Kidney Disease ---
# Patient_ID and the target label are not used as input features.
FEATURES_KIDNEY = [
    "Age",
    "Blood_Pressure",
    "Blood_Urea",
    "Serum_Creatinine",
    "Hemoglobin"
]

X_kidney = df_kidney[FEATURES_KIDNEY]

# Convert target labels into binary values.
# no = 0 (No CKD), si = 1 (CKD)
y_kidney = df_kidney["Chronic_Kidney_Disease"].map({
    "no": 0,
    "si": 1
})

if y_kidney.isna().any():
    raise ValueError("Unexpected or missing CKD target values.")

X_train_k, X_test_k, y_train_k, y_test_k = train_test_split(
    X_kidney,
    y_kidney,
    test_size=0.25,
    random_state=42,
    stratify=y_kidney
)
# Fit preprocessing ONLY on training data.
scaler_kidney = StandardScaler()

X_train_k_scaled = scaler_kidney.fit_transform(X_train_k)
X_test_k_scaled = scaler_kidney.transform(X_test_k)

# Train Logistic Regression model.
model_kidney = LogisticRegression(
    max_iter=1000,
    random_state=42
)
model_kidney.fit(X_train_k_scaled, y_train_k)
# Predict probabilities for the positive class (CKD).
test_probability_kidney = model_kidney.predict_proba(
    X_test_k_scaled
)[:, 1]
# ROC Curve and AUC.
fpr_k, tpr_k, roc_thresholds_k = roc_curve(
    y_test_k,
    test_probability_kidney
)
auc_kidney = roc_auc_score(
    y_test_k,
    test_probability_kidney
)

# ============================================================
# 4. DASH APP + UNIFIED PREMIUM DESIGN SYSTEM
# ============================================================

FONT_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=IBM+Plex+Sans:wght@400;500;600;700;800&display=swap"
)
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY, FONT_URL]
)
app.title = "Health Intelligence Dashboard"
BG = "#0B1220"
CARD_BG = "#151F32"
CARD_BG_HOVER = "#1B2940"
BORDER = "#26354B"
TEXT = "#F1F5F9"
MUTED = "#94A3B8"
GRID = "#26354B"
WHITE = "#FFFFFF"
# Main accent colors
BLUE = "#60A5FA"
TEAL = "#34D399"
RED = "#FB7185"
PURPLE = "#A78BFA"
CYAN = "#22D3EE"
AMBER = "#FBBF24"
# Compatibility aliases for existing code
DARK = TEXT
PAPER = CARD_BG
LINE = BORDER
PAGE_BG = BG
INK = TEXT
ACCENT = TEAL
HEALTHY = CYAN
DIABETIC = RED
MUTED_DIAB = MUTED
MUTED_HEART = MUTED
KIDNEY_HEALTHY = TEAL
KIDNEY_DISEASE = RED
KIDNEY_ACCENT = PURPLE
KIDNEY_MUTED = MUTED
# Heart Disease
RESULT_COLORS_HEART = {
    "No Cardiovascular Disease": TEAL,
    "Cardiovascular Disease": RED
}
HEART_COLORWAY = [
    TEAL, RED, BLUE, PURPLE, AMBER, CYAN
]
# Diabetes
RESULT_COLORS_DIAB = {
    "No diabetes": CYAN,
    "Diabetes": RED
}
DIAB_COLORWAY = [
    CYAN, RED, TEAL, AMBER, PURPLE, BLUE
]
# Chronic Kidney Disease
RESULT_COLORS_KIDNEY = {
    "Healthy": TEAL,
    "CKD": RED
}
KIDNEY_COLORWAY = [
    PURPLE, TEAL, RED, CYAN, AMBER, BLUE
]
PANEL_STYLE = {
    "background": CARD_BG,
    "border": f"1px solid {BORDER}",
    "borderRadius": "16px",
    "padding": "20px",
    "height": "100%",
    "boxShadow": "0 8px 24px rgba(0,0,0,0.18)",
    "transition": "all 0.25s ease"
}
PANEL_HEART = PANEL_STYLE
PANEL_DIAB = PANEL_STYLE
def metric_card_base(title, value_id, accent=TEAL, hint=None):
    children = [
        html.Div(
            style={
                "width": "38px",
                "height": "4px",
                "background": accent,
                "borderRadius": "10px",
                "marginBottom": "16px",
                "boxShadow": f"0 0 12px {accent}55"}),
        html.Div(
            title,
            style={
                "fontSize": "11px",
                "fontWeight": "700",
                "textTransform": "uppercase",
                "letterSpacing": "1.1px",
                "color": MUTED,
                "marginBottom": "8px"}),
        html.H3(
            id=value_id,
            children="—",
            style={
                "fontWeight": "800",
                "fontSize": "30px",
                "color": TEXT,
                "margin": "0",
                "letterSpacing": "-0.7px"})]
    if hint:
        children.append(
            html.Div(
                hint,
                style={
                    "fontSize": "12px",
                    "color": MUTED,
                    "marginTop": "8px"}))
    return html.Div(children, style=PANEL_STYLE)
# Keep existing function names for compatibility
def metric_card_heart(title, value_id, accent=BLUE):
    return metric_card_base(title, value_id, accent)
def metric_card_diab(title, value_id, color=ACCENT, hint=None):
    return metric_card_base(title, value_id, color, hint)
def metric_card_kidney(title, component_id, color):
    return metric_card_base(title, component_id, color)
def chart_panel_base(title, subtitle, graph_id, height=340):
    return html.Div([
        html.Div(
            title,
            style={
                "fontWeight": "750",
                "fontSize": "16px",
                "color": TEXT,
                "marginBottom": "5px"}),
        html.P(
            subtitle,
            style={
                "color": MUTED,
                "fontSize": "12px",
                "lineHeight": "1.6",
                "marginBottom": "12px"}),
        dcc.Graph(
            id=graph_id,
            style={"height": f"{height}px"},
            config={
                "displayModeBar": True,
                "displaylogo": False,
                "responsive": True,
                "modeBarButtonsToRemove": [
                    "lasso2d",
                    "select2d"]})], style=PANEL_STYLE)
# Keep existing function names and signatures
def chart_panel_heart(title, subtitle, graph_id, height=340):
    return chart_panel_base(title, subtitle, graph_id, height)
def chart_panel_diab(title, subtitle, graph_id, height=340):
    return chart_panel_base(title, subtitle, graph_id, height)
def chart_panel_kidney(title, subtitle, graph_id):
    return chart_panel_base(title, subtitle, graph_id, 340)
def format_chart_base(fig, colorway=None):
    if colorway is None:
        colorway = [TEAL, BLUE, RED, PURPLE, CYAN, AMBER]
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font={
            "family": "IBM Plex Sans, Arial, sans-serif",
            "color": TEXT,
            "size": 13,  
        },
        margin={"l": 50, "r": 30, "t": 40, "b": 55},  
        colorway=colorway,
        legend={
            "orientation": "h",
            "y": -0.22,
            "x": 0,
            "title": None,
            "font": {"size": 12, "color": TEXT},  
            "bgcolor": "rgba(0,0,0,0)",
        },
        hoverlabel={
            "bgcolor": "#0F172A",
            "bordercolor": TEAL,
            "font": {
                "family": "IBM Plex Sans, Arial, sans-serif",
                "color": WHITE,
                "size": 13,
            },
        },
        hovermode="closest",)
    fig.update_xaxes(
        showgrid=False,
        showline=True,
        linecolor=BORDER,
        linewidth=1.2,
        tickfont={"color": TEXT, "size": 12},  
        title_font={"color": TEXT, "size": 13},
        zeroline=False,)
    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID,
        gridwidth=0.8,
        tickfont={"color": TEXT, "size": 12},
        title_font={"color": TEXT, "size": 13},
        zeroline=False,)
    fig.update_traces(
        selector=dict(type="pie"),
        textfont={"size": 13, "color": "#0B1220"},
        textinfo="percent+label",)
    return fig
# Dataset-specific wrappers
def format_chart_heart(fig):
    return format_chart_base(fig, HEART_COLORWAY)
def format_chart_diab(fig):
    return format_chart_base(fig, DIAB_COLORWAY)
def format_chart_kidney(fig):
    return format_chart_base(fig, KIDNEY_COLORWAY)
def empty_figure_diab(message="No patients match these filters"):
    fig = px.scatter()
    fig.add_annotation(
        text=message,
        showarrow=False,
        font={
            "size": 14,
            "color": MUTED})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return format_chart_diab(fig)
app.index_string = app.index_string.replace(
    "</head>",
    f"""
    <style>
        /* Main Page */
        body {{
            background: {BG};
            color: {TEXT};
            font-family: 'IBM Plex Sans', Arial, sans-serif;}}
        .container-fluid {{
            background: {BG};}}
        /* Dashboard Tabs */
        .dashboard-tabs {{
            border-bottom: 1px solid {BORDER};
            margin-bottom: 24px;
            gap: 8px;}}
        .dashboard-tab {{
            border: 0 !important;
            background: transparent !important;
            padding: 15px 22px !important;
            font-weight: 600;
            color: {MUTED} !important;
            transition: all 0.2s ease;}}
        .dashboard-tab:hover {{
            color: {TEXT} !important;}}
        .dashboard-tab--selected {{
            border-bottom: 3px solid {TEAL} !important;
            color: {TEAL} !important;
            font-weight: 700;}}
        /* Labels */
        label, .form-label {{
            color: {TEXT} !important;
            font-weight: 600;
            font-size: 13px;
            margin-bottom: 6px;}}
        .dash-dropdown,
        .dash-dropdown *,
        [class*="dash-dropdown"],
        [class*="dash-dropdown"] * {{
            color: #0F172A !important;}}
        .dash-dropdown [class*="control"],
        .dash-dropdown [class*="menu"],
        .dash-dropdown [class*="option"],
        .dash-dropdown input {{
            background-color: #FFFFFF !important;
            color: #0F172A !important;}}
        .dash-dropdown [class*="option"]:hover {{
            background-color: #E2E8F0 !important;
            color: #0F172A !important;}}
        /* Sliders & RangeSlider */
        .rc-slider-track {{
            background-color: {TEAL} !important;
        }}
        .rc-slider-handle,
        .rc-slider-dot-active {{
            border-color: {TEAL} !important;
            background-color: {CARD_BG} !important;
        }}
        [class*="slider"] [class*="mark"],
        [class*="slider"] [class*="Mark"],
        [class*="Slider"] [class*="mark"],
        [class*="Slider"] [class*="Mark"],
        .rc-slider-mark-text,
        .rc-slider-mark span,
        [class*="rc-slider"] span {{
            color: #F1F5F9 !important;
            font-size: 12px !important;
            font-weight: 600 !important;
            opacity: 1 !important;
        }}
        .rc-slider-mark-text-active,
        [class*="mark-active"],
        [class*="Mark"][class*="active"] {{
            color: #FFFFFF !important;
            font-weight: 700 !important;
        }}
        /* Tooltips*/
        [class*="slider"] [class*="tooltip"],
        [class*="slider"] [class*="Tooltip"],
        [class*="Slider"] [class*="tooltip"],
        [class*="Slider"] [class*="Tooltip"],
        .rc-slider-tooltip,
        .rc-slider-tooltip-inner {{
            background-color: #0F172A !important;
            background: #0F172A !important;
            color: {WHITE} !important;
            border: 1px solid {TEAL} !important;
            font-weight: 800 !important;
            font-size: 12px !important;
            padding: 4px 8px !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.45) !important;
        }}
        [class*="slider"] [class*="tooltip"] *,
        [class*="slider"] [class*="Tooltip"] *,
        .rc-slider-tooltip *,
        [class*="rc-slider"] [class*="tooltip"] *,
        [class*="rc-slider"] [class*="Tooltip"] * {{
            color: {WHITE} !important;
            background-color: transparent !important;
        }}
        /* Number Inputs */
        input[type=number] {{
            width: 100%;
            padding: 9px 12px;
            border-radius: 8px;
            border: 1px solid {BORDER};
            background: {CARD_BG};
            color: {TEXT};
        }}
        input[type=number]:focus {{
            outline: none;
            border-color: {TEAL};
        }}
        /* Scrollbar */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        ::-webkit-scrollbar-track {{
            background: {BG};
        }}
        ::-webkit-scrollbar-thumb {{
            background: {BORDER};
            border-radius: 10px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: {MUTED};
        }}
        .dash-slider-mark,
        .dash-slider-mark-outside-selection,
        .dash-slider-mark *,
        .dash-slider-mark-outside-selection * {{
            color: #F1F5F9 !important;
            font-size: 12px !important;
            font-weight: 600 !important;
            opacity: 1 !important;
        }}
        .dash-slider-mark:not(.dash-slider-mark-outside-selection) {{
            color: {TEAL} !important;
            font-weight: 700 !important;
        }}
    </style>
    </head>
    """
)

# ============================================================
# 5. DASHBOARD LAYOUT
# ============================================================
# --- Dataset 1: Heart Disease ---
overview_tab_heart = html.Div([
    html.Div([
        dbc.Row([
            dbc.Col([
                html.Label("Gender", style={"fontWeight": "600", "marginBottom": "8px"}),
                dcc.Dropdown(
                    id="heart-gender-filter",
                    options=[
                        {"label": "All genders", "value": "All"},
                        {"label": "Female", "value": 1},
                        {"label": "Male", "value": 2},
                    ],
                    value="All",
                    clearable=False,
                ),
            ], lg=5),
            dbc.Col([
                html.Label("Age Range", style={"fontWeight": "600", "marginBottom": "8px"}),
                dcc.RangeSlider(
                    id="heart-age-slider",
                    min=int(df_heart["age_years"].min()),
                    max=int(df_heart["age_years"].max()),
                    value=[
                        int(df_heart["age_years"].min()),
                        int(df_heart["age_years"].max())
                    ],
                    step=1,
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ], lg=7),
        ], align="center"),
    ], style={**PANEL_HEART, "marginBottom": "20px"}),
    dbc.Row([
        dbc.Col(
            metric_card_heart("Total Patients", "heart-patients-kpi", BLUE),
            lg=3, sm=6, className="mb-3"
        ),
        dbc.Col(
            metric_card_heart("Disease Rate", "heart-disease-kpi", RED),
            lg=3, sm=6, className="mb-3"
        ),
        dbc.Col(
            metric_card_heart("Average Age", "heart-age-kpi", TEAL),
            lg=3, sm=6, className="mb-3"
        ),
        dbc.Col(
            metric_card_heart("Average Systolic BP", "heart-bp-kpi", "#f6a34d"),
            lg=3, sm=6, className="mb-3"
        ),
    ]),
    dbc.Row([
        dbc.Col(
            chart_panel_heart(
                "Cardiovascular Disease Distribution",
                "Proportion of patients with and without cardiovascular disease",
                "heart-cardio-pie"
            ),
            lg=5, className="mb-3"
        ),
        dbc.Col(
            chart_panel_heart(
                "Age Distribution",
                "Distribution of patients across different ages",
                "heart-age-hist"
            ),
            lg=7, className="mb-3"
        ),
    ]),
    dbc.Row([
        dbc.Col(
            chart_panel_heart(
                "Blood Pressure vs. Disease Status",
                "Comparison of systolic blood pressure between the two groups",
                "heart-bp-box"
            ),
            lg=7, className="mb-3"
        ),
        dbc.Col(
            chart_panel_heart(
                "Cholesterol Levels",
                "Cholesterol categories by cardiovascular disease status",
                "heart-cholesterol-count"
            ),
            lg=5, className="mb-3"
        ),
    ]),
    dbc.Row([
        dbc.Col(
            chart_panel_heart(
                "BMI Distribution",
                "BMI distribution among patients by disease status",
                "heart-bmi-hist"
            ),
            lg=6, className="mb-3"
        ),
        dbc.Col(
            chart_panel_heart(
                "Age vs. Systolic Blood Pressure",
                "Relationship between age and systolic blood pressure",
                "heart-age-bp-scatter"
            ),
            lg=6, className="mb-3"
        ),
    ]),
    dbc.Row([
        dbc.Col(
            chart_panel_heart(
                "Disease Rate by Age Group",
                "Percentage of patients with cardiovascular disease in each age group",
                "heart-age-disease-rate"
            ),
            lg=6, className="mb-3"
        ),
        dbc.Col(
            chart_panel_heart(
                "BMI by Disease Status",
                "Comparison of BMI distribution using violin plots",
                "heart-bmi-violin"
            ),
            lg=6, className="mb-3"
        ),
    ]),
    dbc.Row([
        dbc.Col(
            chart_panel_heart(
                "Health Indicators Correlation",
                "Correlation between numerical health indicators and cardiovascular disease",
                "heart-correlation-heatmap",
                height=450
            ),
            lg=12, className="mb-3"
        ),
    ]),

])
model_tab_heart = html.Div([
    # Decision threshold
    html.Div([
        dbc.Row([
            dbc.Col([
                html.H5("Decision Threshold", className="fw-bold mb-1"),
                html.P("Adjust the threshold used to classify patients based on predicted probabilities.", style={"fontSize": "13px", "color": MUTED_HEART, "marginBottom": 0}),
            ], lg=5),
            dbc.Col(
                dcc.Slider(
                    id="heart-threshold-slider",
                    min=0.1, max=0.9, step=0.05, value=0.5,
                    marks={0.1: "0.1", 0.3: "0.3", 0.5: "0.5", 0.7: "0.7", 0.9: "0.9"},
                    tooltip={"placement": "bottom", "always_visible": True},
                ), lg=7, className="pt-3"
            ),
        ], align="center"),
    ], style={**PANEL_HEART, "marginBottom": "18px"}),

    # Model metrics
    dbc.Row([
        dbc.Col(metric_card_heart("Accuracy", "heart-accuracy-kpi", BLUE), lg=3, sm=6, className="mb-3"),
        dbc.Col(metric_card_heart("Precision", "heart-precision-kpi", TEAL), lg=3, sm=6, className="mb-3"),
        dbc.Col(metric_card_heart("Recall", "heart-recall-kpi", "#f6a34d"), lg=3, sm=6, className="mb-3"),
        dbc.Col(metric_card_heart("F1 Score", "heart-f1-kpi", RED), lg=3, sm=6, className="mb-3"),
    ]),

    html.P(
        f"Metrics are calculated on the held-out test set. ROC-AUC = {auc_heart:.3f}. ROC-AUC does not depend on the selected threshold.",
        style={"color": MUTED_HEART, "fontSize": "13px", "marginTop": "3px"}
    ),

    # Evaluation charts
    dbc.Row([
        dbc.Col(chart_panel_heart("Confusion Matrix", "Actual versus predicted cardiovascular disease classes", "heart-confusion-heatmap"), lg=5, className="mb-3"),
        dbc.Col(chart_panel_heart("ROC Curve", "True Positive Rate vs. False Positive Rate", "heart-roc-line"), lg=7, className="mb-3"),
    ]),
    dbc.Row([
        dbc.Col(chart_panel_heart("Predicted Probabilities", "Predicted probabilities grouped by actual outcome", "heart-prob-hist"), className="mb-3"),
    ]),
])
heart_full_tab = html.Div([
    dcc.Tabs(
        id="heart-sub-tabs",
        value="heart-explore",
        className="dashboard-tabs",
        children=[
            dcc.Tab(
                label="01 Explore the Data",
                value="heart-explore",
                children=overview_tab_heart,
                className="dashboard-tab",
                selected_className="dashboard-tab--selected"
            ),
            dcc.Tab(
                label="02 Evaluate the Model",
                value="heart-evaluate",
                children=model_tab_heart,
                className="dashboard-tab",
                selected_className="dashboard-tab--selected"
            ),
        ]
    )
])

# --- Dataset 2: Diabetes ---
def field_diab(label, control, md=4):
    return dbc.Col([html.Label(label, style={"fontSize": "13px", "fontWeight": "600", "color": MUTED_DIAB, "marginBottom": "6px"}), control], md=md, className="mb-3")
explore_tab_diab = html.Div([
    html.Div(
        dbc.Row([
            field_diab(
                "Gender",
                dcc.Dropdown(
                    id="diab-gender-filter",
                    options=dropdown_options("gender", "All genders"),
                    value="All",
                    clearable=False
                )
            ),
            field_diab(
                "Age Group",
                dcc.Dropdown(
                    id="diab-age-filter",
                    options=dropdown_options("age_group", "All ages"),
                    value="All",
                    clearable=False
                )
            ),
            field_diab(
                "BMI Category",
                dcc.Dropdown(
                    id="diab-bmi-filter",
                    options=dropdown_options("bmi_category", "All BMI categories"),
                    value="All",
                    clearable=False
                )
            ),
        ]),
        style={**PANEL_DIAB, "height": "auto", "marginBottom": "18px"}
    ),
    dbc.Row([
        dbc.Col(
            metric_card_diab(
                "Patients in View",
                "diab-patients-kpi",
                ACCENT,
                "After applying filters"
            ),
            lg=3, sm=6, className="mb-3"
        ),
        dbc.Col(
            metric_card_diab(
                "Diabetes Rate",
                "diab-rate-kpi",
                DIABETIC,
                f"Overall rate: {OVERALL_RATE:.1%}"
            ),
            lg=3, sm=6, className="mb-3"
        ),
        dbc.Col(
            metric_card_diab(
                "Average HbA1c",
                "diab-hba1c-kpi",
                AMBER,
                "Measured in %"
            ),
            lg=3, sm=6, className="mb-3"
        ),
        dbc.Col(
            metric_card_diab(
                "Average Blood Glucose",
                "diab-glucose-kpi",
                HEALTHY,
                "Measured in mg/dL"
            ),
            lg=3, sm=6, className="mb-3"
        ),
    ]),
    dbc.Row([
        dbc.Col(
            chart_panel_diab(
                "Diabetes Distribution",
                "Proportion of patients with and without diabetes",
                "diab-class-pie"
            ),
            lg=4, className="mb-3"
        ),
        dbc.Col(
            chart_panel_diab(
                "HbA1c Distribution by Outcome",
                "Comparison of HbA1c levels between both groups",
                "diab-hba1c-hist"
            ),
            lg=8, className="mb-3"
        ),
    ]),
    dbc.Row([
        dbc.Col(
            chart_panel_diab(
                "Blood Glucose vs. HbA1c",
                "Relationship between two important diabetes indicators",
                "diab-glucose-scatter",
                height=370
            ),
            lg=7, className="mb-3"
        ),

        dbc.Col(
            html.Div([
                html.H5(
                    "Diabetes Rate by Group",
                    style={"fontWeight": "600", "fontSize": "16px"}
                ),
                html.P(
                    "Compare diabetes prevalence across patient groups",
                    style={
                        "color": MUTED_DIAB,
                        "fontSize": "13px"
                    }
                ),
                dcc.Dropdown(
                    id="diab-breakdown-select",
                    options=[
                        {"label": v, "value": k}
                        for k, v in BREAKDOWNS.items()
                    ],
                    value="age_group",
                    clearable=False,
                    className="mb-2"
                ),
                dcc.Graph(
                    id="diab-breakdown-bar",
                    style={"height": "285px"},
                    config={"displayModeBar": False, "responsive": True}
                )
            ], style=PANEL_DIAB),
            lg=5, className="mb-3"
        ),
    ]),
    dbc.Row([
        dbc.Col(
            chart_panel_diab(
                "BMI Distribution by Diabetes Status",
                "Violin plot showing BMI distribution and density",
                "diab-bmi-violin"
            ),
            lg=6, className="mb-3"
        ),
        dbc.Col(
            chart_panel_diab(
                "Risk Score by Outcome",
                "Comparison of calculated risk scores between both groups",
                "diab-risk-score-box"
            ),
            lg=6, className="mb-3"
        ),
    ]),

    dbc.Row([
        dbc.Col(
            chart_panel_diab(
                "Diabetes Rate by Smoking History",
                "Comparison of diabetes prevalence across smoking groups",
                "diab-smoking-rate"
            ),lg=6, className="mb-3"),
        dbc.Col(chart_panel_diab(
                "Diabetes Rate by Health Conditions",
                "How hypertension and heart disease coexist with diabetes",
                "diab-comorbidity-rate"),lg=6, className="mb-3"),]),
    dbc.Row([dbc.Col(chart_panel_diab("Health Indicators Correlation",
                "Strength and direction of relationships between numerical variables",
                "diab-correlation-heatmap",height=450),lg=12, className="mb-3"),]),])

model_tab_diab = html.Div([
    html.Div(dbc.Row([
        dbc.Col([
            html.H5("Decision threshold", className="fw-bold mb-1"),
            html.P("Predict diabetes when the model's probability is at least this value.", style={"fontSize": "13px", "color": MUTED_DIAB, "marginBottom": 0}),
        ], lg=5),
        dbc.Col(dcc.Slider(
            id="diab-threshold-slider", min=0.05, max=0.95, step=0.05, value=0.3,
            marks={0.05: "0.05", 0.3: "0.3", 0.5: "0.5", 0.7: "0.7", 0.95: "0.95"},
            tooltip={"placement": "bottom", "always_visible": True},
        ), lg=7, className="pt-3"),
    ], align="center"), style={**PANEL_DIAB, "height": "auto", "marginBottom": "16px"}),
    dbc.Row([
        dbc.Col(metric_card_diab("Accuracy", "diab-accuracy-kpi", ACCENT, f"Baseline: {BASELINE_ACCURACY:.1%}"), lg=3, sm=6, className="mb-3"),
        dbc.Col(metric_card_diab("Precision", "diab-precision-kpi", HEALTHY, "Predicted cases that are real"), lg=3, sm=6, className="mb-3"),
        dbc.Col(metric_card_diab("Recall", "diab-recall-kpi", DIABETIC, "Diabetic patients found"), lg=3, sm=6, className="mb-3"),
        dbc.Col(metric_card_diab("F1 score", "diab-f1-kpi", AMBER, "Balance"), lg=3, sm=6, className="mb-3"),
    ]),
    dbc.Row([
        dbc.Col(chart_panel_diab("Confusion matrix", "Actual versus predicted class", "diab-confusion-heatmap"), lg=5, className="mb-3"),
        dbc.Col(chart_panel_diab("ROC curve", "The dot marks the selected threshold", "diab-roc-line"), lg=7, className="mb-3"),
    ]),
    dbc.Row([
        dbc.Col(chart_panel_diab("Predicted probabilities", "Percent of each actual group", "diab-prob-hist"), lg=7, className="mb-3"),
        dbc.Col(chart_panel_diab("What drives the prediction", "Standardized coefficients", "diab-importance-bar"), lg=5, className="mb-3"),
    ]),
])

check_tab_diab = dbc.Row([
    dbc.Col(html.Div([
        html.H5("Patient details", style={"fontWeight": "600", "fontSize": "16px", "marginBottom": "14px"}),
        dbc.Row([
            field_diab("Age (years)", dcc.Input(id="diab-calc-age", type="number", value=45, min=0, max=100, step=1), md=6),
            field_diab("BMI", dcc.Input(id="diab-calc-bmi", type="number", value=27, min=10, max=60, step=0.1), md=6),
            field_diab("HbA1c (%)", dcc.Input(id="diab-calc-hba1c", type="number", value=5.5, min=3, max=10, step=0.1), md=6),
            field_diab("Blood glucose (mg/dL)", dcc.Input(id="diab-calc-glucose", type="number", value=140, min=50, max=400, step=1), md=6),
            field_diab("Hypertension", dcc.Dropdown(id="diab-calc-hyp", options=[{"label": "No", "value": 0}, {"label": "Yes", "value": 1}], value=0, clearable=False), md=6),
            field_diab("Heart disease", dcc.Dropdown(id="diab-calc-heart", options=[{"label": "No", "value": 0}, {"label": "Yes", "value": 1}], value=0, clearable=False), md=6),
        ]),
    ], style=PANEL_DIAB), lg=5, className="mb-3"),
    dbc.Col(html.Div([
        html.H5("Estimated chance of diabetes", style={"fontWeight": "600", "fontSize": "16px", "marginBottom": "10px"}),
        html.Div(id="diab-calc-prob", style={"fontSize": "56px", "fontWeight": "600", "lineHeight": "1.1"}),
        dbc.Progress(id="diab-calc-progress", value=0, style={"height": "12px", "margin": "14px 0"}),
        html.P(id="diab-calc-msg", style={"fontSize": "15px"}),
        html.P("This is a classroom model...", style={"color": MUTED_DIAB, "fontSize": "13px", "marginTop": "24px", "marginBottom": 0}),
    ], style=PANEL_DIAB), lg=7, className="mb-3"),
])

diab_full_tab = html.Div([
    dcc.Tabs(
        id="diab-sub-tabs",
        value="diab-explore",
        className="dashboard-tabs",
        children=[
            dcc.Tab(label="Explore patients", value="diab-explore", children=explore_tab_diab, className="dashboard-tab", selected_className="dashboard-tab--selected"),
            dcc.Tab(label="Evaluate the model", value="diab-evaluate", children=model_tab_diab, className="dashboard-tab", selected_className="dashboard-tab--selected"),
            dcc.Tab(label="Check a patient", value="diab-check", children=check_tab_diab, className="dashboard-tab", selected_className="dashboard-tab--selected"),
        ]
    )
])

# --- Dataset 3: Chronic Kidney Disease ---
explore_tab_kidney = html.Div([
    html.Div([
        dbc.Row([
            dbc.Col([
                html.Label("Age Group", className="fw-bold mb-2"),
                dcc.Dropdown(
                    id="kidney-age-filter",
                    options=[
                        {"label": "All age groups", "value": "All"}
                    ] + [
                        {
                            "label": group.replace("_", " ").title(),
                            "value": group
                        }
                        for group in [
                            "child", "young_adult",
                            "middle_aged", "senior"
                        ]
                        if group in df_kidney["Age_Group"].unique()
                    ],
                    value="All",
                    clearable=False,
                ),
            ], md=5, className="mb-3"),

            dbc.Col([
                html.Label("Minimum Age", className="fw-bold mb-2"),
                dcc.Slider(
                    id="kidney-age-slider",
                    min=int(df_kidney["Age"].min()),
                    max=int(df_kidney["Age"].max()),
                    step=1,
                    value=int(df_kidney["Age"].min()),
                    marks={
                        i: str(i)
                        for i in range(
                            int(df_kidney["Age"].min()),
                            int(df_kidney["Age"].max()) + 1,
                            10
                        )
                    },
                    tooltip={
                        "placement": "bottom",
                        "always_visible": False
                    },
                ),
            ], md=7, className="mb-3"),

        ], align="center")
    ], style={**PANEL_DIAB, "marginBottom": "18px"}),
    dbc.Row([
        dbc.Col(
            metric_card_diab(
                "Patients in View",
                "kidney-patients-kpi",
                ACCENT
            ),
            lg=3, sm=6, className="mb-3"
        ),
        dbc.Col(
            metric_card_diab(
                "CKD Rate",
                "kidney-rate-kpi",
                DIABETIC
            ),
            lg=3, sm=6, className="mb-3"
        ),
        dbc.Col(
            metric_card_diab(
                "Average Age",
                "kidney-age-kpi",
                AMBER
            ),
            lg=3, sm=6, className="mb-3"
        ),
        dbc.Col(
            metric_card_diab(
                "Average Serum Creatinine",
                "kidney-creatinine-kpi",
                HEALTHY
            ),
            lg=3, sm=6, className="mb-3"
        ),
    ]),
    dbc.Row([
        dbc.Col(
            chart_panel_diab(
                "CKD Outcomes",
                "Distribution of patients with and without kidney disease",
                "kidney-class-pie"
            ),
            lg=5, className="mb-3"
        ),
        dbc.Col(
            chart_panel_diab(
                "Age Distribution",
                "Patient age distribution by CKD outcome",
                "kidney-age-hist"
            ),
            lg=7, className="mb-3"
        ),
    ]),
    dbc.Row([
        dbc.Col(
            chart_panel_diab(
                "Age vs. Serum Creatinine",
                "Relationship between patient age and creatinine level",
                "kidney-age-creatinine-scatter"
            ),
            lg=7, className="mb-3"
        ),
        dbc.Col(
            chart_panel_diab(
                "Patients by Age Group",
                "Number of CKD and non-CKD patients in each age group",
                "kidney-age-group-count"
            ),
            lg=5, className="mb-3"
        ),
    ]),
    dbc.Row([
        dbc.Col(
            chart_panel_diab(
                "Blood Urea by CKD Outcome",
                "Comparison of blood urea levels between patient groups",
                "kidney-urea-box"
            ),
            lg=6, className="mb-3"
        ),
        dbc.Col(
            chart_panel_diab(
                "Hemoglobin by CKD Outcome",
                "Comparison of hemoglobin levels between patient groups",
                "kidney-hemoglobin-box"
            ),
            lg=6, className="mb-3"
        ),
    ]),
    dbc.Row([
        dbc.Col(
            chart_panel_diab(
                "Albumin Distribution",
                "Albumin levels across CKD outcome groups",
                "kidney-albumin-hist"
            ),
            lg=6, className="mb-3"
        ),
        dbc.Col(
            chart_panel_diab(
                "CKD Rate by Blood Pressure Category",
                "Percentage of CKD patients in each blood pressure group",
                "kidney-bp-rate"
            ),
            lg=6, className="mb-3"
        ),
    ]),
    dbc.Row([
        dbc.Col(
            chart_panel_diab(
                "Blood Glucose vs. Serum Creatinine",
                "Relationship between two important laboratory measurements",
                "kidney-glucose-creatinine"
            ),
            lg=6, className="mb-3"
        ),
        dbc.Col(
            chart_panel_diab(
                "Clinical Features Correlation",
                "Correlation between numerical clinical measurements and CKD",
                "kidney-correlation-heatmap"
            ),
            lg=6, className="mb-3"
        ),
    ]),

])
kidney_model_tab = html.Div([
    # Threshold Filter
    html.Div([
        html.Label(
            "Classification Threshold",
            className="fw-bold mb-2"
        ),
        dcc.Slider(
            id="kidney-threshold-slider",
            min=0.1,
            max=0.9,
            step=0.05,
            value=0.5,
            marks={
                0.1: "0.1",
                0.3: "0.3",
                0.5: "0.5",
                0.7: "0.7",
                0.9: "0.9"
            },
            tooltip={
                "placement": "bottom",
                "always_visible": False
            }
        ),
    ], style={
        "background": "white",
        "padding": "22px",
        "borderRadius": "16px",
        "marginBottom": "18px"
    }),
    # KPI Cards
    dbc.Row([
        dbc.Col(
            metric_card_kidney(
                "Accuracy",
                "kidney-accuracy-kpi",
                KIDNEY_ACCENT
            ),
            lg=3, sm=6, className="mb-3"
        ),
        dbc.Col(
            metric_card_kidney(
                "Precision",
                "kidney-precision-kpi",
                KIDNEY_HEALTHY
            ),
            lg=3, sm=6, className="mb-3"
        ),
        dbc.Col(
            metric_card_kidney(
                "Recall",
                "kidney-recall-kpi",
                KIDNEY_DISEASE
            ),
            lg=3, sm=6, className="mb-3"
        ),
        dbc.Col(
            metric_card_kidney(
                "F1 Score",
                "kidney-f1-kpi",
                "#f6a34d"
            ),
            lg=3, sm=6, className="mb-3"
        ),
    ]),
    # Evaluation Charts
    dbc.Row([
        dbc.Col(
            chart_panel_kidney(
                "Confusion Matrix",
                "Actual vs. predicted CKD classifications",
                "kidney-confusion-heatmap"
            ),
            lg=6, className="mb-3"
        ),
        dbc.Col(
            chart_panel_kidney(
                "ROC Curve",
                f"Model discrimination performance | AUC = {auc_kidney:.3f}",
                "kidney-roc-line"
            ),
            lg=6, className="mb-3"
        ),
    ]),

])
kidney_full_tab = html.Div([
    dcc.Tabs(
        id="kidney-sub-tabs",
        value="kidney-explore",
        className="dashboard-tabs",
        children=[
            dcc.Tab(
                label="Explore patients",
                value="kidney-explore",
                children=explore_tab_kidney,
                className="dashboard-tab",
                selected_className="dashboard-tab--selected"
            ),
            dcc.Tab(
                label="Evaluate the model",
                value="kidney-evaluate",
                children=kidney_model_tab,
                className="dashboard-tab",
                selected_className="dashboard-tab--selected"
            ),
        ]
    )
])

# --- Main App Layout ---
app.layout = html.Div([
    dbc.Container([
        html.Div([
            html.Div("HEALTHCARE ANALYTICS / MULTI-MODEL SYSTEM", style={"fontSize": "12px", "letterSpacing": "2px", "fontWeight": "750", "color": "#bcd0ff", "marginBottom": "13px"}),
            html.H1("Unified Health Intelligence Dashboard", style={"fontWeight": "800", "fontSize": "clamp(27px, 4vw, 40px)", "margin": "0 0 8px", "color": "white"}),
            html.P("Explore Cardiovascular Disease, Diabetes, and Chronic Kidney Disease analytics seamlessly in one place.", style={"color": "#d6e1f5", "fontSize": "15px", "margin": "0 0 19px"}),
        ], style={
            "background": "linear-gradient(115deg, #101f3a 0%, #1d3976 70%, #2755a1 100%)",
            "borderRadius": "22px", "padding": "35px 35px", "color": "white",
            "boxShadow": "0 14px 35px rgba(16,31,58,0.12)", "marginBottom": "22px"
        }),
        dcc.Tabs(
            id="main-dataset-tabs", value="tab-heart", className="dashboard-tabs",
            children=[
                dcc.Tab(label="Cardiovascular Disease", value="tab-heart", children=heart_full_tab, className="dashboard-tab", selected_className="dashboard-tab--selected"),
                dcc.Tab(label="Diabetes Risk", value="tab-diabetes", children=diab_full_tab, className="dashboard-tab", selected_className="dashboard-tab--selected"),
                dcc.Tab(label="Chronic Kidney Disease", value="tab-kidney", children=kidney_full_tab, className="dashboard-tab", selected_className="dashboard-tab--selected"),
            ]
        ),
        html.P("Multi-Model Health Analytics Dashboard • Machine Learning Decision Support", style={"color": "#94a3b8", "textAlign": "center", "fontSize": "12px", "marginTop": "22px"}),
    ], style={"maxWidth": "1260px"}, fluid=True),
], style={"background": PAGE_BG, "minHeight": "100vh", "padding": "26px 12px 42px"})

# ============================================================
# 6. CALLBACK 1: UPDATE THE DATA EXPLORATION TAB
# ============================================================
# --- Dataset 1: Heart Disease ---
@app.callback(
    Output("heart-patients-kpi", "children"),
    Output("heart-disease-kpi", "children"),
    Output("heart-age-kpi", "children"),
    Output("heart-bp-kpi", "children"),
    Output("heart-cardio-pie", "figure"),
    Output("heart-age-hist", "figure"),
    Output("heart-bp-box", "figure"),
    Output("heart-cholesterol-count", "figure"),
    Output("heart-bmi-hist", "figure"),
    Output("heart-age-bp-scatter", "figure"),
    Output("heart-age-disease-rate", "figure"),
    Output("heart-bmi-violin", "figure"),
    Output("heart-correlation-heatmap", "figure"),
    Input("heart-gender-filter", "value"),
    Input("heart-age-slider", "value"),
)

def update_overview_heart(selected_gender, age_range):
    filtered = df_heart[
        (df_heart["age_years"] >= age_range[0]) &
        (df_heart["age_years"] <= age_range[1])
    ].copy()
    if selected_gender != "All":
        filtered = filtered[
            filtered["gender"] == int(selected_gender)
        ]
    if filtered.empty:
        empty_figs = []
        for _ in range(9):
            fig = px.scatter()
            fig.add_annotation(
                text="No patients match these filters",
                showarrow=False,
                font={"size": 14}
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            empty_figs.append(format_chart_heart(fig))
        return (
            "0",
            "—",
            "—",
            "—",
            *empty_figs
        )
    patients = f"{len(filtered):,}"
    disease_rate = f"{filtered['cardio'].mean():.1%}"
    avg_age = f"{filtered['age_years'].mean():.1f} years"
    avg_bp = f"{filtered['ap_hi'].mean():.1f} mmHg"
    filtered["cardio_txt"] = filtered["cardio"].map({
        0: "No Disease",
        1: "Cardiovascular Disease"
    })
    pie_fig = px.pie(
        filtered,
        names="cardio_txt",
        hole=0.62,
        color="cardio_txt",
        color_discrete_map={
            "No Disease": TEAL,
            "Cardiovascular Disease": RED
        }
    )
    pie_fig.update_traces(
        textinfo="percent+label",
        textposition="outside",
        textfont={"color": WHITE, "size": 12, "family": "IBM Plex Sans, Arial, sans-serif"},
        outsidetextfont={"color": WHITE, "size": 12, "family": "IBM Plex Sans, Arial, sans-serif"},
        automargin=True,
        marker={"line": {"color": "white", "width": 2}},
    )
    pie_fig.update_layout(showlegend=False)
    age_fig = px.histogram(
        filtered,
        x="age_years",
        nbins=25,
        color_discrete_sequence=[BLUE],
        labels={"age_years": "Age (years)"}
    )
    age_fig.update_yaxes(title="Number of patients")
    bp_fig = px.box(
        filtered,
        x="cardio",
        y="ap_hi",
        color="cardio",
        color_discrete_map={
            0: TEAL,
            1: RED
        },
        points="outliers",
        labels={
            "cardio": "Disease Status",
            "ap_hi": "Systolic Blood Pressure (mmHg)"
        }
    )
    bp_fig.update_xaxes(
        tickmode="array",
        tickvals=[0, 1],
        ticktext=["No Disease", "Disease"]
    )
    cholesterol_data = (
        filtered.groupby(["cholesterol", "cardio"])
        .size()
        .reset_index(name="Patients")
    )
    cholesterol_data["Cholesterol Category"] = (
        cholesterol_data["cholesterol"].map({
            1: "Normal",
            2: "Above Normal",
            3: "Well Above Normal"
        })
    )
    cholesterol_fig = px.bar(
        cholesterol_data,
        x="Cholesterol Category",
        y="Patients",
        color="cardio",
        barmode="group",
        color_discrete_map={
            0: TEAL,
            1: RED
        },
        category_orders={
            "Cholesterol Category": [
                "Normal",
                "Above Normal",
                "Well Above Normal"
            ]
        },
        labels={
            "cardio": "Disease Status",
            "Patients": "Number of patients"
        }
    )
    bmi_fig = px.histogram(
        filtered,
        x="bmi",
        nbins=25,
        color="cardio",
        color_discrete_map={
            0: TEAL,
            1: RED
        },
        barmode="overlay",
        opacity=0.60,
        labels={
            "bmi": "Body Mass Index (BMI)",
            "cardio": "Disease Status"
        }
    )
    bmi_fig.update_yaxes(title="Number of patients")
    scatter_fig = px.scatter(
        filtered,
        x="age_years",
        y="ap_hi",
        color="cardio",
        color_discrete_map={
            0: TEAL,
            1: RED
        },
        opacity=0.55,
        labels={
            "age_years": "Age (years)",
            "ap_hi": "Systolic Blood Pressure (mmHg)",
            "cardio": "Disease Status"
        }
    )
    age_data = filtered.copy()
    age_data["Age Group"] = pd.cut(
        age_data["age_years"],
        bins=[0, 40, 50, 60, 70, 200],
        labels=[
            "Under 40",
            "40–49",
            "50–59",
            "60–69",
            "70+"
        ],
        right=False
    )
    age_rate = (
        age_data.groupby("Age Group", observed=True)["cardio"]
        .mean()
        .reset_index()
    )
    age_rate["Disease Rate (%)"] = age_rate["cardio"] * 100
    age_rate_fig = px.bar(
        age_rate,
        x="Age Group",
        y="Disease Rate (%)",
        color="Disease Rate (%)",
        color_continuous_scale=["#93c5fd", BLUE],
        text="Disease Rate (%)",
        labels={
            "Age Group": "Age Group",
            "Disease Rate (%)": "Disease Rate"
        }
    )
    age_rate_fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )
    age_rate_fig.update_yaxes(
        title="Disease Rate (%)",
        range=[0, 100]
    )
    age_rate_fig.update_layout(coloraxis_showscale=False)
    bmi_violin_fig = px.violin(
        filtered,
        x="cardio",
        y="bmi",
        color="cardio",
        box=True,
        points=False,
        color_discrete_map={
            0: TEAL,
            1: RED
        },
        labels={
            "cardio": "Disease Status",
            "bmi": "Body Mass Index (BMI)"
        }
    )
    bmi_violin_fig.update_xaxes(
        tickmode="array",
        tickvals=[0, 1],
        ticktext=["No Disease", "Disease"]
    )
    numeric_data = filtered.select_dtypes(include="number").copy()
    if "id" in numeric_data.columns:
        numeric_data = numeric_data.drop(columns=["id"])

    correlation = numeric_data.corr()

    correlation_fig = px.imshow(
        correlation,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        labels={
            "x": "Health Indicators",
            "y": "Health Indicators",
            "color": "Correlation"
        }
    )
    correlation_fig.update_traces(
        hovertemplate=(
            "%{y} vs %{x}<br>"
            "Correlation: %{z:.2f}"
            "<extra></extra>"
        )
    )
    correlation_fig.update_layout(
        xaxis={"tickangle": -35}
    )
    return (
        patients,
        disease_rate,
        avg_age,
        avg_bp,
        format_chart_heart(pie_fig),
        format_chart_heart(age_fig),
        format_chart_heart(bp_fig),
        format_chart_heart(cholesterol_fig),
        format_chart_heart(bmi_fig),
        format_chart_heart(scatter_fig),
        format_chart_heart(age_rate_fig),
        format_chart_heart(bmi_violin_fig),
        format_chart_heart(correlation_fig),
    )

# --- Dataset 2: Diabetes ---
@app.callback(
    Output("diab-patients-kpi", "children"),
    Output("diab-rate-kpi", "children"),
    Output("diab-hba1c-kpi", "children"),
    Output("diab-glucose-kpi", "children"),
    Output("diab-class-pie", "figure"),
    Output("diab-hba1c-hist", "figure"),
    Output("diab-glucose-scatter", "figure"),
    Output("diab-breakdown-bar", "figure"),
    Output("diab-bmi-violin", "figure"),
    Output("diab-risk-score-box", "figure"),
    Output("diab-smoking-rate", "figure"),
    Output("diab-comorbidity-rate", "figure"),
    Output("diab-correlation-heatmap", "figure"),
    Input("diab-gender-filter", "value"),
    Input("diab-age-filter", "value"),
    Input("diab-bmi-filter", "value"),
    Input("diab-breakdown-select", "value"),
)

def update_explore_diab(gender, age_group, bmi_category, breakdown):
    filtered = df.copy()
    if gender != "All":
        filtered = filtered[filtered["gender"] == gender]
    if age_group != "All":
        filtered = filtered[filtered["age_group"] == age_group]
    if bmi_category != "All":
        filtered = filtered[filtered["bmi_category"] == bmi_category]
    if filtered.empty:
        blank_figures = []
        for _ in range(9):
            blank_figures.append(empty_figure_diab())
        return (
            "0",
            "-",
            "-",
            "-",
            *blank_figures
        )
    patients = f"{len(filtered):,}"
    rate = f"{filtered['diabetes'].mean():.1%}"
    hba1c = f"{filtered['HbA1c_level'].mean():.2f}"
    glucose = f"{filtered['blood_glucose_level'].mean():.0f}"
    pie_fig = px.pie(
        filtered,
        names="result",
        hole=0.62,
        color="result",
        color_discrete_map=RESULT_COLORS_DIAB,
        category_orders={
            "result": ["No diabetes", "Diabetes"]
        }
    )
    pie_fig.update_traces(
        textinfo="percent+label",
        textposition="outside",
        marker={"line": {"color": "white", "width": 2}}
    )
    pie_fig.update_layout(showlegend=False)
    hist_fig = px.histogram(
        filtered,
        x="HbA1c_level",
        color="result",
        nbins=30,
        barmode="overlay",
        histnorm="percent",
        opacity=0.75,
        color_discrete_map=RESULT_COLORS_DIAB,
        category_orders={
            "result": ["No diabetes", "Diabetes"]
        },
        labels={
            "HbA1c_level": "HbA1c Level (%)",
            "result": "Diabetes Status"
        }
    )
    hist_fig.update_yaxes(title="Percent of group")
    sample = filtered.sample(
        n=min(len(filtered), 3000),
        random_state=1
    ).sort_values("diabetes")
    scatter_fig = px.scatter(
        sample,
        x="HbA1c_level",
        y="blood_glucose_level",
        color="result",
        color_discrete_map=RESULT_COLORS_DIAB,
        opacity=0.55,
        hover_data=["age", "bmi", "risk_score"],
        labels={
            "HbA1c_level": "HbA1c Level (%)",
            "blood_glucose_level": "Blood Glucose (mg/dL)",
            "result": "Diabetes Status"
        }
    )
    groups = (
        filtered.groupby(breakdown, observed=True)["diabetes"]
        .agg(rate="mean", patients="size")
        .reindex(ORDERS[breakdown])
        .dropna()
        .reset_index()
    )
    groups["rate"] = groups["rate"] * 100
    bar_fig = px.bar(
        groups,
        x=breakdown,
        y="rate",
        text=groups["rate"].map("{:.1f}%".format),
        hover_data={
            "patients": True,
            "rate": ":.1f"
        },
        color="rate",
        color_continuous_scale=[
            [0, "#f5cdbd"],
            [1, DIABETIC]
        ],
        labels={
            breakdown: BREAKDOWNS[breakdown],
            "rate": "Diabetes Rate (%)"
        }
    )
    bar_fig.update_traces(
        textposition="outside",
        cliponaxis=False
    )
    bar_fig.update_layout(coloraxis_showscale=False)
    bar_fig.add_hline(
        y=OVERALL_RATE * 100,
        line_dash="dash",
        line_color=INK,
        annotation_text="Overall rate",
        annotation_position="top left",
        annotation_font_color=MUTED_DIAB
    )
    bar_fig.update_yaxes(
        title="Diabetes Rate (%)",
        rangemode="tozero"
    )
    bmi_fig = px.violin(
        filtered,
        x="result",
        y="bmi",
        color="result",
        box=True,
        points=False,
        color_discrete_map=RESULT_COLORS_DIAB,
        category_orders={
            "result": ["No diabetes", "Diabetes"]
        },
        labels={
            "result": "Diabetes Status",
            "bmi": "Body Mass Index (BMI)"
        }
    )
    risk_fig = px.box(
        filtered,
        x="result",
        y="risk_score",
        color="result",
        points="outliers",
        color_discrete_map=RESULT_COLORS_DIAB,
        category_orders={
            "result": ["No diabetes", "Diabetes"]
        },
        labels={
            "result": "Diabetes Status",
            "risk_score": "Risk Score"
        }
    )
    smoking_data = (
        filtered.groupby("smoking_history", observed=True)["diabetes"]
        .agg(rate="mean", patients="size")
        .reset_index()
    )
    smoking_data["rate_pct"] = smoking_data["rate"] * 100
    smoking_fig = px.bar(
        smoking_data,
        x="smoking_history",
        y="rate_pct",
        color="rate_pct",
        text=smoking_data["rate_pct"].map("{:.1f}%".format),
        hover_data={"patients": True},
        color_continuous_scale=[
            [0, "#f5cdbd"],
            [1, DIABETIC]
        ],
        labels={
            "smoking_history": "Smoking History",
            "rate_pct": "Diabetes Rate (%)"
        }
    )
    smoking_fig.update_traces(
        textposition="outside",
        cliponaxis=False
    )
    smoking_fig.update_layout(coloraxis_showscale=False)
    smoking_fig.update_yaxes(
        title="Diabetes Rate (%)",
        rangemode="tozero"
    )
    comorbidity_data = filtered.copy()
    comorbidity_data["comorbidity_group"] = "Neither condition"
    comorbidity_data.loc[
        (comorbidity_data["hypertension"] == 1) &
        (comorbidity_data["heart_disease"] == 0),
        "comorbidity_group"
    ] = "Hypertension only"
    comorbidity_data.loc[
        (comorbidity_data["hypertension"] == 0) &
        (comorbidity_data["heart_disease"] == 1),
        "comorbidity_group"
    ] = "Heart disease only"
    comorbidity_data.loc[
        (comorbidity_data["hypertension"] == 1) &
        (comorbidity_data["heart_disease"] == 1),
        "comorbidity_group"
    ] = "Both conditions"
    comorbidity_summary = (
        comorbidity_data.groupby("comorbidity_group")["diabetes"]
        .agg(rate="mean", patients="size")
        .reset_index()
    )
    comorbidity_summary["rate_pct"] = (
        comorbidity_summary["rate"] * 100
    )
    comorbidity_order = [
        "Neither condition",
        "Hypertension only",
        "Heart disease only",
        "Both conditions"
    ]
    comorbidity_fig = px.bar(
        comorbidity_summary,
        x="comorbidity_group",
        y="rate_pct",
        color="rate_pct",
        text=comorbidity_summary["rate_pct"].map("{:.1f}%".format),
        hover_data={"patients": True},
        category_orders={
            "comorbidity_group": comorbidity_order
        },
        color_continuous_scale=[
            [0, "#f5cdbd"],
            [1, DIABETIC]
        ],
        labels={
            "comorbidity_group": "Health Conditions",
            "rate_pct": "Diabetes Rate (%)"
        }
    )
    comorbidity_fig.update_traces(
        textposition="outside",
        cliponaxis=False
    )
    comorbidity_fig.update_layout(coloraxis_showscale=False)
    comorbidity_fig.update_xaxes(tickangle=-15)
    comorbidity_fig.update_yaxes(
        title="Diabetes Rate (%)",
        rangemode="tozero"
    )
    correlation_columns = [
        "age",
        "hypertension",
        "heart_disease",
        "bmi",
        "HbA1c_level",
        "blood_glucose_level",
        "diabetes",
        "risk_score"
    ]
    available_columns = [
        col for col in correlation_columns
        if col in filtered.columns
    ]
    correlation = filtered[available_columns].corr()
    correlation_fig = px.imshow(
        correlation,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        labels={
            "x": "Health Indicators",
            "y": "Health Indicators",
            "color": "Correlation"
        }
    )
    correlation_fig.update_layout(
        xaxis={"tickangle": -35}
    )
    return (
        patients,
        rate,
        hba1c,
        glucose,
        format_chart_diab(pie_fig),
        format_chart_diab(hist_fig),
        format_chart_diab(scatter_fig),
        format_chart_diab(bar_fig),
        format_chart_diab(bmi_fig),
        format_chart_diab(risk_fig),
        format_chart_diab(smoking_fig),
        format_chart_diab(comorbidity_fig),
        format_chart_diab(correlation_fig)
    )

# --- Dataset 3: Chronic Kidney Disease ---
@app.callback(
    Output("kidney-patients-kpi", "children"),
    Output("kidney-rate-kpi", "children"),
    Output("kidney-age-kpi", "children"),
    Output("kidney-creatinine-kpi", "children"),
    Output("kidney-class-pie", "figure"),
    Output("kidney-age-hist", "figure"),
    Output("kidney-age-creatinine-scatter", "figure"),
    Output("kidney-age-group-count", "figure"),
    Output("kidney-urea-box", "figure"),
    Output("kidney-hemoglobin-box", "figure"),
    Output("kidney-albumin-hist", "figure"),
    Output("kidney-bp-rate", "figure"),
    Output("kidney-glucose-creatinine", "figure"),
    Output("kidney-correlation-heatmap", "figure"),
    Input("kidney-age-filter", "value"),
    Input("kidney-age-slider", "value"),
)
def update_kidney_overview(selected_age_group, min_age):
    filtered = df_kidney[
        df_kidney["Age"] >= min_age
    ].copy()

    if selected_age_group != "All":
        filtered = filtered[
            filtered["Age_Group"] == selected_age_group
        ]
    if filtered.empty:
        empty_fig = format_chart_diab(
            px.scatter(
                title="No data available for the selected filters"
            )
        )
        return (
            "0", "N/A", "N/A", "N/A",
            *([empty_fig] * 10)
        )
    patients = f"{len(filtered):,}"
    ckd_rate = (
        filtered["Chronic_Kidney_Disease"]
        .eq("si")
        .mean()
    )
    ckd_rate = f"{ckd_rate:.1%}"
    average_age = f"{filtered['Age'].mean():.1f} years"
    average_creatinine = (
        f"{filtered['Serum_Creatinine'].mean():.2f}"
    )
    filtered["CKD_Result"] = (
        filtered["Chronic_Kidney_Disease"]
        .map({
            "no": "No CKD",
            "si": "CKD"
        })
    )
    COLORS = {
        "No CKD": HEALTHY,
        "CKD": DIABETIC
    }
    ORDER = ["No CKD", "CKD"]
    pie_fig = px.pie(
        filtered,
        names="CKD_Result",
        hole=0.58,
        color="CKD_Result",
        color_discrete_map=COLORS,
        category_orders={
            "CKD_Result": ORDER
        },
    )
    pie_fig.update_traces(
        textinfo="percent+label",
        textposition="outside",
        marker={
            "line": {
                "color": "white",
                "width": 2
            }
        }
    )
    pie_fig.update_layout(showlegend=False)
    hist_fig = px.histogram(
        filtered,
        x="Age",
        nbins=20,
        color="CKD_Result",
        barmode="overlay",
        opacity=0.75,
        color_discrete_map=COLORS,
        category_orders={
            "CKD_Result": ORDER
        },
        labels={
            "Age": "Age (years)",
            "CKD_Result": "CKD Outcome"
        },
    )
    hist_fig.update_yaxes(title="Number of patients")
    scatter_fig = px.scatter(
        filtered,
        x="Age",
        y="Serum_Creatinine",
        color="CKD_Result",
        color_discrete_map=COLORS,
        category_orders={
            "CKD_Result": ORDER
        },
        opacity=0.7,
        hover_data=[
            "Blood_Urea",
            "Hemoglobin",
            "Blood_Pressure"
        ],
        labels={
            "Age": "Age (years)",
            "Serum_Creatinine": "Serum Creatinine",
            "CKD_Result": "CKD Outcome"
        },
    )
    count_fig = px.histogram(
        filtered,
        x="Age_Group",
        color="CKD_Result",
        barmode="group",
        color_discrete_map=COLORS,
        category_orders={
            "Age_Group": [
                "child",
                "young_adult",
                "middle_aged",
                "senior"
            ],
            "CKD_Result": ORDER
        },
        labels={
            "Age_Group": "Age Group",
            "CKD_Result": "CKD Outcome"
        },
    )
    count_fig.update_yaxes(title="Number of patients")
    count_fig.update_xaxes(tickangle=-15)
    urea_fig = px.box(
        filtered,
        x="CKD_Result",
        y="Blood_Urea",
        color="CKD_Result",
        color_discrete_map=COLORS,
        category_orders={
            "CKD_Result": ORDER
        },
        points="outliers",
        labels={
            "CKD_Result": "CKD Outcome",
            "Blood_Urea": "Blood Urea"
        },
    )
    urea_fig.update_layout(showlegend=False)
    urea_fig.update_yaxes(title="Blood Urea")
    hemoglobin_fig = px.violin(
        filtered,
        x="CKD_Result",
        y="Hemoglobin",
        color="CKD_Result",
        color_discrete_map=COLORS,
        category_orders={
            "CKD_Result": ORDER
        },
        box=True,
        points=False,
        labels={
            "CKD_Result": "CKD Outcome",
            "Hemoglobin": "Hemoglobin"
        },
    )
    hemoglobin_fig.update_layout(showlegend=False)
    hemoglobin_fig.update_yaxes(title="Hemoglobin")
    albumin_fig = px.histogram(
        filtered,
        x="Albumin",
        color="CKD_Result",
        barmode="group",
        color_discrete_map=COLORS,
        category_orders={
            "CKD_Result": ORDER
        },
        labels={
            "Albumin": "Albumin Level",
            "CKD_Result": "CKD Outcome"
        },
    )

    albumin_fig.update_yaxes(title="Number of patients")
    bp_groups = (
        filtered.groupby(
            "Blood_Pressure_Category",
            observed=True
        )["Chronic_Kidney_Disease"]
        .apply(lambda x: x.eq("si").mean() * 100)
        .reset_index(name="CKD_Rate")
    )

    bp_fig = px.bar(
        bp_groups,
        x="Blood_Pressure_Category",
        y="CKD_Rate",
        text=bp_groups["CKD_Rate"].map(
            "{:.1f}%".format
        ),
        color="CKD_Rate",
        color_continuous_scale=[
            [0, "#b9d8e8"],
            [1, DIABETIC]
        ],
        labels={
            "Blood_Pressure_Category": "Blood Pressure Category",
            "CKD_Rate": "CKD Rate (%)"
        },
    )
    bp_fig.update_traces(
        textposition="outside",
        cliponaxis=False
    )
    bp_fig.update_layout(
        coloraxis_showscale=False
    )
    bp_fig.update_yaxes(
        title="CKD Rate (%)",
        rangemode="tozero"
    )
    glucose_fig = px.scatter(
        filtered,
        x="Blood_Glucose_Random",
        y="Serum_Creatinine",
        color="CKD_Result",
        color_discrete_map=COLORS,
        category_orders={
            "CKD_Result": ORDER
        },
        opacity=0.7,
        hover_data=[
            "Age",
            "Blood_Urea",
            "Blood_Pressure"
        ],
        labels={
            "Blood_Glucose_Random": "Random Blood Glucose",
            "Serum_Creatinine": "Serum Creatinine",
            "CKD_Result": "CKD Outcome"
        },
    )
    correlation_columns = [
        "Age",
        "Blood_Pressure",
        "Specific_Gravity",
        "Albumin",
        "Sugar",
        "Blood_Glucose_Random",
        "Blood_Urea",
        "Serum_Creatinine",
        "Sodium",
        "Potassium",
        "Hemoglobin",
        "Packed_Cell_Volume",
        "White_Blood_Cell_Count",
        "Red_Blood_Cell_Count"
    ]
    correlation_data = filtered[
        correlation_columns
    ].copy()
    correlation_data["CKD_Encoded"] = (
        filtered["Chronic_Kidney_Disease"]
        .eq("si")
        .astype(int)
    )
    correlation_matrix = correlation_data.corr()
    correlation_fig = px.imshow(
        correlation_matrix,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        labels={
            "x": "Clinical Features",
            "y": "Clinical Features",
            "color": "Correlation"
        },
    )
    correlation_fig.update_layout(
        coloraxis_colorbar={
            "title": "Correlation"
        }
    )
    return (
        patients,
        ckd_rate,
        average_age,
        average_creatinine,
        format_chart_diab(pie_fig),
        format_chart_diab(hist_fig),
        format_chart_diab(scatter_fig),
        format_chart_diab(count_fig),
        format_chart_diab(urea_fig),
        format_chart_diab(hemoglobin_fig),
        format_chart_diab(albumin_fig),
        format_chart_diab(bp_fig),
        format_chart_diab(glucose_fig),
        format_chart_diab(correlation_fig),
    )

# ============================================================
# 7. CALLBACK 2: UPDATE THE MODEL EVALUATION TAB
# ============================================================

# --- Dataset 1: Heart Disease ---
@app.callback(
    Output("heart-accuracy-kpi", "children"),
    Output("heart-precision-kpi", "children"),
    Output("heart-recall-kpi", "children"),
    Output("heart-f1-kpi", "children"),
    Output("heart-confusion-heatmap", "figure"),
    Output("heart-roc-line", "figure"),
    Output("heart-prob-hist", "figure"),
    Input("heart-threshold-slider", "value"),
)
def update_model_heart(threshold):

    # Convert probabilities into predictions using selected threshold (Heart Model)
    predictions = (heart_prob >= threshold).astype(int)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test_h, predictions)
    precision = precision_score(y_test_h, predictions, zero_division=0)
    recall = recall_score(y_test_h, predictions, zero_division=0)
    f1 = f1_score(y_test_h, predictions, zero_division=0)

    # Confusion Matrix
    matrix = confusion_matrix(y_test_h, predictions, labels=[0, 1])

    confusion_fig = px.imshow(
        matrix,
        x=["No Disease", "Disease"],
        y=["No Disease", "Disease"],
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        labels={
            "x": "Predicted Class",
            "y": "Actual Class",
            "color": "Number of Patients"
        },
    )

    confusion_fig.update_traces(textfont={"size": 22})
    confusion_fig.update_layout(coloraxis_showscale=False)

    # ROC Curve
    roc_fig = px.line(
        x=fpr_h,
        y=tpr_h,
        labels={
            "x": "False Positive Rate",
            "y": "True Positive Rate"
        },
    )

    roc_fig.update_traces(
        line={"color": BLUE, "width": 3},
        name=f"ROC-AUC = {auc_heart:.3f}"
    )

    roc_fig.add_scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random Baseline",
        line={"color": "#b3c0d1", "dash": "dash"}
    )

    roc_fig.update_xaxes(range=[0, 1])
    roc_fig.update_yaxes(range=[0, 1.03])

    # Predicted Probability Distribution
    test_df = pd.DataFrame({
        "Predicted Probability": heart_prob,
        "Actual Outcome": y_test_h.map({
            0: "No Cardiovascular Disease",
            1: "Cardiovascular Disease"
        }).to_numpy(),
    })

    prob_fig = px.histogram(
        test_df,
        x="Predicted Probability",
        color="Actual Outcome",
        barmode="overlay",
        opacity=0.70,
        nbins=25,
        color_discrete_map=RESULT_COLORS_HEART,
    )

    prob_fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color=DARK,
        annotation_text=f"Threshold = {threshold:.2f}",
        annotation_position="top"
    )

    return (
        f"{accuracy:.1%}",
        f"{precision:.1%}",
        f"{recall:.1%}",
        f"{f1:.1%}",
        format_chart_heart(confusion_fig),
        format_chart_heart(roc_fig),
        format_chart_heart(prob_fig),
    )

# --- Dataset 2: Diabetes ---
@app.callback(
    Output("diab-accuracy-kpi", "children"),
    Output("diab-precision-kpi", "children"),
    Output("diab-recall-kpi", "children"),
    Output("diab-f1-kpi", "children"),
    Output("diab-confusion-heatmap", "figure"),
    Output("diab-roc-line", "figure"),
    Output("diab-prob-hist", "figure"),
    Output("diab-importance-bar", "figure"),
    Input("diab-threshold-slider", "value"),
)
def update_model_diab(threshold):
    predictions = (test_probability_diab >= threshold).astype(int)
    accuracy = accuracy_score(y_test_d, predictions)
    precision = precision_score(y_test_d, predictions, zero_division=0)
    recall = recall_score(y_test_d, predictions, zero_division=0)
    f1 = f1_score(y_test_d, predictions, zero_division=0)

    labels = ["No diabetes", "Diabetes"]
    matrix = confusion_matrix(y_test_d, predictions, labels=[0, 1])
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

    roc_fig = px.line(x=fpr_diab, y=tpr_diab, labels={"x": "False positive rate", "y": "True positive rate"})
    roc_fig.update_traces(line={"color": ACCENT, "width": 3}, name=f"ROC-AUC {auc_diab:.3f}", showlegend=True)
    roc_fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                        line={"color": "#b3c0c7", "dash": "dash"})
    roc_fig.add_scatter(x=[point_fpr], y=[point_tpr], mode="markers",
                        name=f"Threshold {threshold:.2f}",
                        marker={"color": DIABETIC, "size": 12})
    roc_fig.update_xaxes(range=[0, 1])
    roc_fig.update_yaxes(range=[0, 1.03])

    # Histogram by ACTUAL label
    test_df = pd.DataFrame({
        "Predicted probability": test_probability_diab,
        "Actual outcome": y_test_d.map({0: "No diabetes", 1: "Diabetes"}).to_numpy(),
    })
    prob_fig = px.histogram(
        test_df, x="Predicted probability", color="Actual outcome", barmode="overlay",
        histnorm="percent", opacity=0.75, nbins=25, color_discrete_map=RESULT_COLORS_DIAB,
        category_orders={"Actual outcome": ["No diabetes", "Diabetes"]},
    )
    prob_fig.update_yaxes(title="Percent of group")
    prob_fig.add_vline(x=threshold, line_dash="dash", line_color=INK,
                       annotation_text=f"Threshold {threshold:.2f}", annotation_position="top")

    coefs = pd.DataFrame({
        "Feature": [FEATURE_LABELS[f] for f in FEATURES_DIAB],
        "Coefficient": model_diab.coef_[0],
    }).sort_values("Coefficient")
    importance_fig = px.bar(coefs, x="Coefficient", y="Feature", orientation="h",
                            color_discrete_sequence=[ACCENT])
    importance_fig.update_yaxes(title=None)

    return (f"{accuracy:.1%}", f"{precision:.1%}", f"{recall:.1%}", f"{f1:.1%}",
            format_chart_diab(confusion_fig), format_chart_diab(roc_fig),
            format_chart_diab(prob_fig), format_chart_diab(importance_fig))

# --- Dataset 3: Chronic Kidney Disease ---
@app.callback(
    Output("kidney-accuracy-kpi", "children"),
    Output("kidney-precision-kpi", "children"),
    Output("kidney-recall-kpi", "children"),
    Output("kidney-f1-kpi", "children"),
    Output("kidney-confusion-heatmap", "figure"),
    Output("kidney-roc-line", "figure"),
    Input("kidney-threshold-slider", "value"),
)
def update_model_kidney(threshold):
    # Predictions based on selected threshold.
    predictions = (
        test_probability_kidney >= threshold
    ).astype(int)
    # Evaluation metrics.
    accuracy = accuracy_score(y_test_k, predictions)
    precision = precision_score(
        y_test_k, predictions, zero_division=0
    )
    recall = recall_score(
        y_test_k, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test_k, predictions, zero_division=0
    )
    # Confusion Matrix.
    labels = ["No CKD", "CKD"]
    matrix = confusion_matrix(
        y_test_k,
        predictions,
        labels=[0, 1]
    )
    confusion_fig = px.imshow(
        matrix,
        x=labels,
        y=labels,
        text_auto=True,
        aspect="auto",
        color_continuous_scale=[
            [0, "#eef5f6"],
            [1, KIDNEY_ACCENT]
        ],
        labels={
            "x": "Predicted",
            "y": "Actual",
            "color": "Patients"
        },
    )
    confusion_fig.update_traces(
        textfont={"size": 20}
    )
    confusion_fig.update_layout(
        coloraxis_showscale=False
    )
    # Calculate the operating point for the selected threshold.
    tn, fp, fn, tp = matrix.ravel()
    point_fpr = fp / (fp + tn) if (fp + tn) else 0
    point_tpr = tp / (tp + fn) if (tp + fn) else 0
    # ROC Curve.
    roc_fig = px.line(
        x=fpr_k,
        y=tpr_k,
        labels={
            "x": "False Positive Rate",
            "y": "True Positive Rate"
        }
    )
    roc_fig.update_traces(
        line={
            "color": KIDNEY_ACCENT,
            "width": 3
        },
        name=f"ROC-AUC {auc_kidney:.3f}",
        showlegend=True
    )
    roc_fig.add_scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random Guess",
        line={
            "color": "#b3c0c7",
            "dash": "dash"
        }
    )
    roc_fig.add_scatter(
        x=[point_fpr],
        y=[point_tpr],
        mode="markers",
        name=f"Threshold {threshold:.2f}",
        marker={
            "color": KIDNEY_DISEASE,
            "size": 12
        }
    )
    roc_fig.update_xaxes(range=[0, 1])
    roc_fig.update_yaxes(range=[0, 1.03])
    return (
        f"{accuracy:.1%}",
        f"{precision:.1%}",
        f"{recall:.1%}",
        f"{f1:.1%}",
        format_chart_kidney(confusion_fig),
        format_chart_kidney(roc_fig)
    )

# ============================================================
# 8. RUN THE APP
# ============================================================
if __name__ == "__main__":
    app.run(debug=True, port=8050)