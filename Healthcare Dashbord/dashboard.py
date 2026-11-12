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
KIDNEY_FILE = Path(__file__).with_name("kidney_cleaning.csv")
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
# 4. DASH APP + SIMPLE STYLING
# ============================================================
FONT_URL = "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap"
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY, FONT_URL])
app.title = "Health Intelligence Dashboard"

# --- Dataset 1: Heart Disease ---
BLUE = "#2563eb"
DARK = "#17243a"
TEAL = "#16a6a0"
RED = "#ef7079"
MUTED_HEART = "#758399"
PAPER = "#ffffff"

RESULT_COLORS_HEART = {
    "No Cardiovascular Disease": TEAL,
    "Cardiovascular Disease": RED
}

PANEL_HEART = {
    "background": PAPER,
    "border": "1px solid #e7edf5",
    "borderRadius": "18px",
    "padding": "20px",
    "height": "100%",
    "boxShadow": "0 6px 20px rgba(27,53,91,0.04)",
}

def metric_card_heart(title, value_id, accent=BLUE):
    return html.Div([
        html.Div(style={"width": "31px", "height": "4px", "background": accent, "borderRadius": "8px", "marginBottom": "15px"}),
        html.Div(title, style={"fontSize": "12px", "fontWeight": "700", "textTransform": "uppercase", "letterSpacing": "1px", "color": MUTED_HEART}),
        html.H3(id=value_id, children="—", style={"fontWeight": "800", "marginTop": "8px", "marginBottom": "0", "fontSize": "29px"}),
    ], style=PANEL_HEART)

def chart_panel_heart(title, subtitle, graph_id, height=315):
    return html.Div([
        html.H5(title, style={"fontWeight": "750", "fontSize": "17px", "marginBottom": "3px"}),
        html.P(subtitle, style={"color": MUTED_HEART, "fontSize": "12px", "marginBottom": "8px"}),
        dcc.Graph(id=graph_id, style={"height": f"{height}px"}, config={"displayModeBar": False, "responsive": True}),
    ], style=PANEL_HEART)

def format_chart_heart(fig):
    fig.update_layout(
        template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
        font={"family": "Arial, sans-serif", "color": DARK, "size": 12},
        margin={"l": 25, "r": 20, "t": 18, "b": 25},
        legend={"orientation": "h", "y": -0.22, "x": 0},
        colorway=[BLUE, TEAL, RED, "#f6bd60"],
    )
    fig.update_xaxes(showgrid=False, linecolor="#e3eaf2")
    fig.update_yaxes(gridcolor="#edf2f7", zeroline=False)
    return fig

# --- Dataset 2: Diabetes ---
INK = "#12303a"
ACCENT = "#0f6b75"
HEALTHY = "#5fa8bd"
DIABETIC = "#e0522d"
AMBER = "#e9a93a"
MUTED_DIAB = "#62767f"
LINE = "#dbe4e7"
PAGE_BG = "#eef3f4"
RESULT_COLORS_DIAB = {"No diabetes": HEALTHY, "Diabetes": DIABETIC}

PANEL_DIAB = {"background": "white", "border": f"1px solid {LINE}", "borderRadius": "10px", "padding": "18px 20px", "height": "100%"}

def metric_card_diab(title, value_id, color=ACCENT, hint=None):
    children = [
        html.Div(title, style={"fontSize": "13px", "color": MUTED_DIAB, "marginBottom": "6px"}),
        html.Div(id=value_id, children="-", style={"fontWeight": "600", "fontSize": "30px", "lineHeight": "1.1", "color": INK}),
    ]
    if hint:
        children.append(html.Div(hint, style={"fontSize": "12px", "color": MUTED_DIAB, "marginTop": "6px"}))
    return html.Div(children, style={**PANEL_DIAB, "borderLeft": f"5px solid {color}"})

def chart_panel_diab(title, subtitle, graph_id, height=320):
    return html.Div([
        html.H5(title, style={"fontWeight": "600", "fontSize": "16px", "marginBottom": "2px"}),
        html.P(subtitle, style={"color": MUTED_DIAB, "fontSize": "13px", "marginBottom": "8px"}),
        dcc.Graph(id=graph_id, style={"height": f"{height}px"}, config={"displayModeBar": False, "responsive": True}),
    ], style=PANEL_DIAB)

def format_chart_diab(fig):
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

def empty_figure_diab(message="No patients match these filters"):
    fig = px.scatter()
    fig.add_annotation(text=message, showarrow=False, font={"size": 14, "color": MUTED_DIAB})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return format_chart_diab(fig)


app.index_string = app.index_string.replace(
    "</head>",
    f"""<style>
      body {{ background: {PAGE_BG}; }}
      .dashboard-tabs {{ border-bottom: {LINE}; margin-bottom: 22px; }}
      .dashboard-tab {{ border: 0 !important; background: transparent !important; padding: 16px 23px !important; font-weight: 600; color: {MUTED_DIAB} !important; }}
      .dashboard-tab--selected {{ border-bottom: 3px solid {ACCENT} !important; color: {ACCENT} !important; font-weight: 700; }}
      .Select-control {{ border-radius: 8px; border-color: {LINE}; min-height: 40px; }}
      .dash-dropdown {{ border-radius: 8px; }}
      .rc-slider-track {{ background-color: {ACCENT}; }}
      .rc-slider-handle {{ border-color: {ACCENT}; }}
      input[type=number] {{ width: 100%; padding: 7px 10px; border-radius: 6px; border: 1px solid {LINE}; }}
    </style></head>""",
)

# --- Dataset 3: Chronic Kidney Disease ---
KIDNEY_HEALTHY = "#16a6a0"
KIDNEY_DISEASE = "#ef7079"
KIDNEY_ACCENT = "#2563eb"
KIDNEY_MUTED = "#758399"
def format_chart_kidney(fig):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial"),
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig
def metric_card_kidney(title, component_id, color):
    return dbc.Card(
        dbc.CardBody([
            html.P(
                title,
                style={
                    "color": KIDNEY_MUTED,
                    "fontSize": "13px",
                    "marginBottom": "8px"
                }
            ),
            html.H3(
                id=component_id,
                children="—",
                style={
                    "fontWeight": "800",
                    "color": color,
                    "margin": "0"
                }
            ),
        ]),
        style={
            "border": "none",
            "borderRadius": "16px",
            "boxShadow": "0 4px 15px rgba(0,0,0,0.05)",
            "background": "white"
        }
    )
def chart_panel_kidney(title, subtitle, graph_id):
    return dbc.Card(
        dbc.CardBody([
            html.H5(title, style={"fontWeight": "700"}),
            html.P(
                subtitle,
                style={"color": KIDNEY_MUTED, "fontSize": "13px"}
            ),
            dcc.Graph(id=graph_id)
        ]),
        style={
            "border": "none",
            "borderRadius": "16px",
            "boxShadow": "0 4px 15px rgba(0,0,0,0.05)",
            "background": "white"
        }
    )

# ============================================================
# 5. DASHBOARD LAYOUT
# ============================================================
# --- Dataset 1: Heart Disease ---
overview_tab_heart = html.Div([
    # Heart filters
    html.Div([
        dbc.Row([
            dbc.Col([
                html.Label("Gender"),
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
                html.Label("Age range"),
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
        ]),
    ], style={**PANEL_HEART, "marginBottom": "18px"}),

    # KPI Cards
    dbc.Row([
        dbc.Col(metric_card_heart("Total Patients", "heart-patients-kpi", BLUE), lg=3, sm=6, className="mb-3"),
        dbc.Col(metric_card_heart("Disease Rate", "heart-disease-kpi", RED), lg=3, sm=6, className="mb-3"),
        dbc.Col(metric_card_heart("Average Age", "heart-age-kpi", TEAL), lg=3, sm=6, className="mb-3"),
        dbc.Col(metric_card_heart("Average Systolic BP", "heart-bp-kpi", "#f6a34d"), lg=3, sm=6, className="mb-3"),
    ]),

    # First row of charts
    dbc.Row([
        dbc.Col(chart_panel_heart("Cardiovascular Disease Distribution", "Number of patients with and without cardiovascular disease", "heart-cardio-pie"), lg=5, className="mb-3"),
        dbc.Col(chart_panel_heart("Age Distribution", "Distribution of patients by age", "heart-age-hist"), lg=7, className="mb-3"),
    ]),

    # Second row of charts
    dbc.Row([
        dbc.Col(chart_panel_heart("Blood Pressure vs. Cardiovascular Disease", "Comparison of systolic blood pressure between the two classes", "heart-bp-box"), lg=7, className="mb-3"),
        dbc.Col(chart_panel_heart("Cholesterol Levels", "Cholesterol distribution by disease status", "heart-cholesterol-count"), lg=5, className="mb-3"),
    ]),

    # Third row
    dbc.Row([
        dbc.Col(chart_panel_heart("BMI Distribution", "Body Mass Index distribution among patients", "heart-bmi-hist"), lg=6, className="mb-3"),
        dbc.Col(chart_panel_heart("Age vs. Systolic Blood Pressure", "Relationship between age and systolic blood pressure", "heart-age-bp-scatter"), lg=6, className="mb-3"),
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
            dcc.Tab(label="01 Explore the Data", value="heart-explore", children=overview_tab_heart, className="dashboard-tab", selected_className="dashboard-tab--selected"),
            dcc.Tab(label="02 Evaluate the Model", value="heart-evaluate", children=model_tab_heart, className="dashboard-tab", selected_className="dashboard-tab--selected"),
        ]
    )
])

# --- Dataset 2: Diabetes ---
def field_diab(label, control, md=4):
    return dbc.Col([html.Label(label, style={"fontSize": "13px", "fontWeight": "600", "color": MUTED_DIAB, "marginBottom": "6px"}), control], md=md, className="mb-3")

explore_tab_diab = html.Div([
    html.Div(dbc.Row([
        field_diab("Gender", dcc.Dropdown(id="diab-gender-filter", options=dropdown_options("gender", "All genders"), value="All", clearable=False)),
        field_diab("Age group", dcc.Dropdown(id="diab-age-filter", options=dropdown_options("age_group", "All ages"), value="All", clearable=False)),
        field_diab("BMI category", dcc.Dropdown(id="diab-bmi-filter", options=dropdown_options("bmi_category", "All BMI categories"), value="All", clearable=False)),
    ], style={**PANEL_DIAB, "height": "auto", "marginBottom": "16px", "paddingBottom": "6px"})),
    dbc.Row([
        dbc.Col(metric_card_diab("Patients in view", "diab-patients-kpi", ACCENT, "After the filters above"), lg=3, sm=6, className="mb-3"),
        dbc.Col(metric_card_diab("Diabetes rate", "diab-rate-kpi", DIABETIC, f"All patients: {OVERALL_RATE:.1%}"), lg=3, sm=6, className="mb-3"),
        dbc.Col(metric_card_diab("Average HbA1c (%)", "diab-hba1c-kpi", AMBER, "Diabetes cutoff: 6.5%"), lg=3, sm=6, className="mb-3"),
        dbc.Col(metric_card_diab("Average blood glucose (mg/dL)", "diab-glucose-kpi", HEALTHY, "Average across the view"), lg=3, sm=6, className="mb-3"),
    ]),
    dbc.Row([
        dbc.Col(chart_panel_diab("Outcome balance", "Share of patients with and without diabetes", "diab-class-pie"), lg=4, className="mb-3"),
        dbc.Col(chart_panel_diab("HbA1c by outcome", "Percent of each group at every HbA1c level", "diab-hba1c-hist"), lg=8, className="mb-3"),
    ]),
    dbc.Row([
        dbc.Col(chart_panel_diab("Blood glucose vs. HbA1c", "A random sample of up to 3,000 patients from the view", "diab-glucose-scatter", height=360), lg=7, className="mb-3"),
        dbc.Col(html.Div([
            html.H5("Diabetes rate by group", style={"fontWeight": "600", "fontSize": "16px", "marginBottom": "2px"}),
            html.P("Dashed line is the rate across all patients", style={"color": MUTED_DIAB, "fontSize": "13px", "marginBottom": "8px"}),
            dcc.Dropdown(id="diab-breakdown-select", options=[{"label": v, "value": k} for k, v in BREAKDOWNS.items()], value="age_group", clearable=False, className="mb-2"),
            dcc.Graph(id="diab-breakdown-bar", style={"height": "285px"}, config={"displayModeBar": False, "responsive": True}),
        ], style=PANEL_DIAB), lg=5, className="mb-3"),
    ]),
])

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
    html.Div(dbc.Row([
        dbc.Col([
            html.Label("Age Group", className="fw-bold mb-2"),
            dcc.Dropdown(
                id="kidney-age-filter",
                options=[{"label": "All age groups", "value": "All"}] + [
                    {"label": group.replace("_", " ").title(), "value": group}
                    for group in ["child", "young_adult", "middle_aged", "senior"]
                    if group in df_kidney["Age_Group"].unique()
                ],
                value="All", clearable=False,
            ),
        ], md=5, className="mb-3"),
        dbc.Col([
            html.Label("Minimum Age", className="fw-bold mb-2"),
            dcc.Slider(
                id="kidney-age-slider",
                min=int(df_kidney["Age"].min()), max=int(df_kidney["Age"].max()),
                step=1, value=int(df_kidney["Age"].min()),
                marks={i: str(i) for i in range(int(df_kidney["Age"].min()), int(df_kidney["Age"].max()) + 1, 10)},
                tooltip={"placement": "bottom", "always_visible": False},
            ),
        ], md=7, className="mb-3"),
    ], align="center"), style={**PANEL_DIAB, "marginBottom": "18px"}),
    dbc.Row([
        dbc.Col(metric_card_diab("Patients in view", "kidney-patients-kpi", ACCENT), lg=3, sm=6, className="mb-3"),
        dbc.Col(metric_card_diab("CKD Rate", "kidney-rate-kpi", DIABETIC), lg=3, sm=6, className="mb-3"),
        dbc.Col(metric_card_diab("Average Age", "kidney-age-kpi", AMBER), lg=3, sm=6, className="mb-3"),
        dbc.Col(metric_card_diab("Average Serum Creatinine", "kidney-creatinine-kpi", HEALTHY), lg=3, sm=6, className="mb-3"),
    ]),
    dbc.Row([
        dbc.Col(chart_panel_diab("CKD Outcomes", "Class balance for patients with and without CKD", "kidney-class-pie"), lg=5, className="mb-3"),
        dbc.Col(chart_panel_diab("Age Distribution", "Distribution of patients by age", "kidney-age-hist"), lg=7, className="mb-3"),
    ]),
    dbc.Row([
        dbc.Col(chart_panel_diab("Age vs. Serum Creatinine", "Each dot represents one patient", "kidney-age-creatinine-scatter"), lg=7, className="mb-3"),
        dbc.Col(chart_panel_diab("Patients by Age Group", "Number of patients in each age group", "kidney-age-group-count"), lg=5, className="mb-3"),
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
    Input("heart-gender-filter", "value"), 
    Input("heart-age-slider", "value"),
)
def update_overview_heart(selected_gender, age_range):

    # Apply filters on df_heart
    filtered = df_heart[
        (df_heart["age_years"] >= age_range[0]) &
        (df_heart["age_years"] <= age_range[1])
    ].copy()

    if selected_gender != "All":
        filtered = filtered[filtered["gender"] == int(selected_gender)]
        
    
    # KPI values
    patients = f"{len(filtered):,}"
    disease_rate = f"{filtered['cardio'].mean():.1%}"
    avg_age = f"{filtered['age_years'].mean():.1f} years"
    avg_bp = f"{filtered['ap_hi'].mean():.1f} mmHg"

    # Disease distribution
    filtered["cardio_txt"] = filtered["cardio"].map({0: "No Disease", 1: "Cardiovascular Disease"})

    pie_fig = px.pie(
        filtered,
        names="cardio_txt",
        hole=0.58,
        color="cardio_txt",
        color_discrete_map={
            "No Disease": TEAL,
            "Cardiovascular Disease": RED
        }
    )

    pie_fig.update_traces(
        textinfo="percent+label",
        textposition="outside"
    )
    pie_fig.update_layout(showlegend=False)

    # Age distribution
    age_fig = px.histogram(
        filtered,
        x="age_years",
        nbins=20,
        color_discrete_sequence=[TEAL],  
        labels={"age_years": "Age (years)"}
    )
    age_fig.update_yaxes(title="Number of patients")

    # Blood pressure by disease status
    bp_fig = px.box(
        filtered,
        x="cardio",
        y="ap_hi",
        color="cardio",
        color_discrete_map={0: TEAL, 1: RED},
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

    # Cholesterol distribution
    cholesterol_fig = px.histogram(
        filtered,
        x="cholesterol",
        color="cardio",
        barmode="group",
        color_discrete_map={0: TEAL, 1: RED},
        labels={
            "cholesterol": "Cholesterol Level",
            "cardio": "Disease Status"
        }
    )
    cholesterol_fig.update_xaxes(
        tickmode="array",
        tickvals=[1, 2, 3],
        ticktext=["Normal", "Above Normal", "Well Above Normal"]
    )
    cholesterol_fig.update_yaxes(title="Number of patients")

    # BMI distribution
    bmi_fig = px.histogram(
        filtered,
        x="bmi",
        nbins=25,
        color="cardio",
        color_discrete_map={0: TEAL, 1: RED},
        barmode="overlay",
        opacity=0.65,
        labels={
            "bmi": "Body Mass Index (BMI)",
            "cardio": "Disease Status"
        }
    )
    bmi_fig.update_yaxes(title="Number of patients")

    # Age vs systolic blood pressure
    scatter_fig = px.scatter(
        filtered,
        x="age_years",
        y="ap_hi",
        color="cardio",
        color_discrete_map={0: TEAL, 1: RED},
        opacity=0.65,
        labels={
            "age_years": "Age (years)",
            "ap_hi": "Systolic Blood Pressure (mmHg)",
            "cardio": "Disease Status"
        }
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
        blank = empty_figure_diab()
        return "0", "-", "-", "-", blank, blank, blank, blank

    patients = f"{len(filtered):,}"
    rate = f"{filtered['diabetes'].mean():.1%}"
    hba1c = f"{filtered['HbA1c_level'].mean():.2f}"
    glucose = f"{filtered['blood_glucose_level'].mean():.0f}"

    pie_fig = px.pie(
        filtered, names="result", hole=0.58, color="result",
        color_discrete_map=RESULT_COLORS_DIAB,
        category_orders={"result": ["No diabetes", "Diabetes"]},
    )
    pie_fig.update_traces(textinfo="percent+label", textposition="outside",
                          marker={"line": {"color": "white", "width": 2}})
    pie_fig.update_layout(showlegend=False)

    hist_fig = px.histogram(
        filtered, x="HbA1c_level", color="result", nbins=30, barmode="overlay",
        histnorm="percent", opacity=0.75, color_discrete_map=RESULT_COLORS_DIAB,
        category_orders={"result": ["No diabetes", "Diabetes"]},
        labels={"HbA1c_level": "HbA1c level (%)", "result": "Outcome"},
    )
    hist_fig.update_yaxes(title="Percent of group")

    sample = filtered.sample(min(len(filtered), 3000), random_state=1).sort_values("diabetes")
    scatter_fig = px.scatter(
        sample, x="HbA1c_level", y="blood_glucose_level", color="result",
        color_discrete_map=RESULT_COLORS_DIAB, opacity=0.55,
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
                      annotation_font_color=MUTED_DIAB)
    bar_fig.update_yaxes(title="Diabetes rate (%)", rangemode="tozero")

    return (patients, rate, hba1c, glucose,
            format_chart_diab(pie_fig), format_chart_diab(hist_fig),
            format_chart_diab(scatter_fig), format_chart_diab(bar_fig))

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
    Input("kidney-age-filter", "value"),
    Input("kidney-age-slider", "value"),
)
def update_kidney_overview(selected_age_group, min_age):

    # Filter the kidney dataset
    filtered = df_kidney[df_kidney["Age"] >= min_age].copy()

    if selected_age_group != "All":
        filtered = filtered[
            filtered["Age_Group"] == selected_age_group
        ]

    # Handle empty filtered data
    if filtered.empty:
        empty_fig = format_chart_diab(
            px.scatter(title="No data available for the selected filters")
        )
        return (
            "0", "N/A", "N/A", "N/A",
            empty_fig, empty_fig, empty_fig, empty_fig,
        )

    # KPI Cards
    patients = f"{len(filtered):,}"

    ckd_rate = (
        filtered["Chronic_Kidney_Disease"].eq("si").mean()
    )
    ckd_rate = f"{ckd_rate:.1%}"

    average_age = f"{filtered['Age'].mean():.1f} years"

    average_creatinine = (
        f"{filtered['Serum_Creatinine'].mean():.2f}"
    )

    # Display labels for the target
    filtered["CKD_Result"] = filtered[
        "Chronic_Kidney_Disease"
    ].map({
        "no": "No CKD",
        "si": "CKD"
    })

    # 1. CKD Outcomes
    pie_fig = px.pie(
        filtered,
        names="CKD_Result",
        hole=0.58,
        color="CKD_Result",
        color_discrete_map={
            "No CKD": HEALTHY,
            "CKD": DIABETIC
        },
        category_orders={
            "CKD_Result": ["No CKD", "CKD"]
        },
    )
    pie_fig.update_traces(
        textinfo="percent+label",
        textposition="outside"
    )
    pie_fig.update_layout(showlegend=False)

    # 2. Age Distribution
    hist_fig = px.histogram(
        filtered,
        x="Age",
        nbins=20,
        color="CKD_Result",
        color_discrete_map={
            "No CKD": HEALTHY,
            "CKD": DIABETIC
        },
        labels={"Age": "Age (years)"}
    )
    hist_fig.update_yaxes(title="Number of patients")

    # 3. Age vs. Serum Creatinine
    scatter_fig = px.scatter(
        filtered,
        x="Age",
        y="Serum_Creatinine",
        color="CKD_Result",
        color_discrete_map={
            "No CKD": HEALTHY,
            "CKD": DIABETIC
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
            "CKD_Result": "CKD Result"
        },
    )

    # 4. Patients by Age Group
    count_fig = px.histogram(
        filtered,
        x="Age_Group",
        color="CKD_Result",
        barmode="group",
        color_discrete_map={
            "No CKD": HEALTHY,
            "CKD": DIABETIC
        },
        category_orders={
            "Age_Group": [
                "child",
                "young_adult",
                "middle_aged",
                "senior"
            ]
        },
        labels={
            "Age_Group": "Age Group",
            "CKD_Result": "CKD Result"
        },
    )
    count_fig.update_yaxes(title="Number of patients")
    count_fig.update_xaxes(tickangle=-15)

    return (
        patients,
        ckd_rate,
        average_age,
        average_creatinine,
        format_chart_diab(pie_fig),
        format_chart_diab(hist_fig),
        format_chart_diab(scatter_fig),
        format_chart_diab(count_fig),
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