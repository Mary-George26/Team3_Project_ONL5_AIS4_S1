# Heart Disease Analytics Dashboard (Dash + Plotly + Pandas)
import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc, Input, Output




df = pd.read_csv("heart_disease.csv")

# Add readable label columns
df["Status"] = df["cardio"].map({0: "No Heart Disease", 1: "Heart Disease"})
df["Gender"] = df["gender"].map({1: "Female", 2: "Male"})
df["Cholesterol"] = df["cholesterol"].map({1: "Normal", 2: "Above Normal", 3: "Well Above Normal"})
df["Age Group"] = pd.cut(df["age_years"], bins=[0, 40, 50, 60, 200],
                         labels=["Under 40", "40-49", "50-59", "60+"], right=False)








# Colors used in the charts
colors = {"No Heart Disease": "#8EB8E5", "Heart Disease": "#0B2545"}
blue = "#2E6FBF"



#insights

rate_by_gender = df.groupby("Gender")["cardio"].mean() * 100
rate_by_age = df.groupby("Age Group", observed=True)["cardio"].mean() * 100
rate_by_chol = df.groupby("Cholesterol")["cardio"].mean() * 100
bp_by_status = df.groupby("Status")["ap_hi"].mean()

insights = [
    f"Heart disease rate: {rate_by_gender['Female']:.1f}% for females and {rate_by_gender['Male']:.1f}% for males.",
    f"The age group '{rate_by_age.idxmax()}' has the highest rate ({rate_by_age.max():.1f}%), while '{rate_by_age.idxmin()}' has the lowest ({rate_by_age.min():.1f}%).",
    f"Patients with 'Well Above Normal' cholesterol have a rate of {rate_by_chol['Well Above Normal']:.1f}%, compared with {rate_by_chol['Normal']:.1f}% for 'Normal' cholesterol.",
    f"Average systolic blood pressure is {bp_by_status['Heart Disease']:.0f} mmHg with heart disease and {bp_by_status['No Heart Disease']:.0f} mmHg without.",
]











# -layout
app = Dash(__name__)
app.title = "Heart Disease Analytics"

def card(title, value_id):
    return html.Div([html.P(title, className="kpi-title"),
                     html.H2(id=value_id, className="kpi-value")], className="kpi-card")

def chart(chart_id, title):
    return html.Div([html.H4(title), dcc.Graph(id=chart_id, config={"displayModeBar": False})],
                    className="chart-box")

def dropdown(label, dropdown_id, options):
    return html.Div([html.Label(label),
                     dcc.Dropdown(["All"] + options, "All", id=dropdown_id, clearable=False)],
                    className="filter")

app.layout = html.Div([
    # Sidebar with filters
    html.Div([
        html.H3("Filters"),
        dropdown("Gender", "gender", ["Female", "Male"]),
        dropdown("Age Group", "age_group", ["Under 40", "40-49", "50-59", "60+"]),
        dropdown("Heart Disease Status", "status", ["Heart Disease", "No Heart Disease"]),
        dropdown("Cholesterol Level", "cholesterol", ["Normal", "Above Normal", "Well Above Normal"]),
        html.P("Gender is read from the dataset codes: 1 = Female, 2 = Male.", className="note"),
    ], className="sidebar"),

    # Main content
    html.Div([
        html.Div([html.H1("Heart Disease Analytics"),
                  html.P("Explore how patient age, gender, cholesterol and blood pressure relate to heart disease.")],
                 className="header"),

        html.Div([card("Total Patients", "kpi-total"),
                  card("Heart Disease Cases", "kpi-cases"),
                  card("Without Heart Disease", "kpi-healthy"),
                  card("Heart Disease %", "kpi-percent"),
                  card("Average Age", "kpi-age")], className="kpi-row"),

        html.Div([chart("fig-status", "Heart Disease Distribution"),
                  chart("fig-gender", "Heart Disease Rate by Gender"),
                  chart("fig-age", "Heart Disease Rate by Age Group"),
                  chart("fig-chol", "Heart Disease Rate by Cholesterol Level"),
                  chart("fig-box", "Systolic Blood Pressure by Status"),
                  chart("fig-scatter", "Systolic vs Diastolic Blood Pressure (sample)")],
                 className="chart-grid"),

        html.Div([html.H3("Key Insights (full dataset)"),
                  html.Ul([html.Li(text) for text in insights]),
                  html.P("These are patterns in the data only. They do not prove that one factor causes heart disease.",
                         className="note")], className="insights"),
    ], className="main"),
], className="page")














# ----------  Callback: filters -> KPIs and charts ----------
@app.callback(
    Output("kpi-total", "children"), Output("kpi-cases", "children"),
    Output("kpi-healthy", "children"), Output("kpi-percent", "children"),
    Output("kpi-age", "children"),
    Output("fig-status", "figure"), Output("fig-gender", "figure"),
    Output("fig-age", "figure"), Output("fig-chol", "figure"),
    Output("fig-box", "figure"), Output("fig-scatter", "figure"),
    Input("gender", "value"), Input("age_group", "value"),
    Input("status", "value"), Input("cholesterol", "value"),
)
def update_dashboard(gender, age_group, status, cholesterol):
    # Apply the filters
    data = df
    if gender != "All":
        data = data[data["Gender"] == gender]
    if age_group != "All":
        data = data[data["Age Group"] == age_group]
    if status != "All":
        data = data[data["Status"] == status]
    if cholesterol != "All":
        data = data[data["Cholesterol"] == cholesterol]

    # KPIs
    total = len(data)
    cases = int(data["cardio"].sum())
    percent = cases / total * 100 if total > 0 else 0
    avg_age = data["age_years"].mean() if total > 0 else 0

    # Charts
    fig_status = px.pie(data, names="Status", hole=0.5, color="Status", color_discrete_map=colors)

    gender_rate = data.groupby("Gender")["cardio"].mean().mul(100).round(1).reset_index()
    fig_gender = px.bar(gender_rate, x="Gender", y="cardio", text="cardio",
                        labels={"cardio": "Heart disease rate (%)"}, color_discrete_sequence=[blue])

    age_rate = data.groupby("Age Group", observed=True)["cardio"].mean().mul(100).round(1).reset_index()
    fig_age = px.bar(age_rate, x="Age Group", y="cardio", text="cardio",
                     labels={"cardio": "Heart disease rate (%)"}, color_discrete_sequence=[blue])

    chol_rate = data.groupby("Cholesterol")["cardio"].mean().mul(100).round(1).reset_index()
    fig_chol = px.bar(chol_rate, x="Cholesterol", y="cardio", text="cardio",
                      category_orders={"Cholesterol": ["Normal", "Above Normal", "Well Above Normal"]},
                      labels={"cardio": "Heart disease rate (%)"}, color_discrete_sequence=[blue])

    fig_box = px.box(data, x="Status", y="ap_hi", color="Status", color_discrete_map=colors,
                     labels={"ap_hi": "Systolic BP (mmHg)"})

    sample = data.sample(n=min(2000, total), random_state=1)  # sample keeps the chart fast
    fig_scatter = px.scatter(sample, x="ap_hi", y="ap_lo", color="Status", opacity=0.6,
                             color_discrete_map=colors,
                             labels={"ap_hi": "Systolic BP (mmHg)", "ap_lo": "Diastolic BP (mmHg)"})

    # Same look for all charts
    for fig in [fig_status, fig_gender, fig_age, fig_chol, fig_box, fig_scatter]:
        fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=10, b=10),
                          height=320, legend_title_text="")
    for fig in [fig_gender, fig_age, fig_chol]:
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(yaxis_range=[0, 100])
    fig_box.update_layout(showlegend=False)

    return (f"{total:,}", f"{cases:,}", f"{total - cases:,}", f"{percent:.1f}%", f"{avg_age:.1f}",
            fig_status, fig_gender, fig_age, fig_chol, fig_box, fig_scatter)







if __name__ == "__main__":
    app.run(debug=True)
