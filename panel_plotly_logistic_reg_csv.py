# try with panel and plotly
import panel as pn
import plotly.express as px
import pandas as pd
import numpy as np
pn.extension('plotly')
import plotly.graph_objects as go
import io

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# load iris dataset (keep original column names)
iris = load_iris()
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                  columns=iris['feature_names'] + ['target'])
df['target'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# define widgets (use original df column names)
x_axis = pn.widgets.Select(name='X Axis', options=list(df.columns[:-1]), value=list(df.columns[:-1])[0])
y_axis = pn.widgets.Select(name='Y Axis', options=list(df.columns[:-1]), value=list(df.columns[:-1])[1])
size = pn.widgets.Select(name='Size', options=list(df.columns[:-1]), value=list(df.columns[:-1])[2])
color = pn.widgets.Select(name='Color', options=list(df.columns), value='target')

# logistic regression model (trained on original-named columns)
model = LogisticRegression(max_iter=200)
model.fit(df.drop(columns=['target']), df['target'])
model_accuracy = model.score(df.drop(columns=['target']), df['target']) * 100

# File input widget
file_input = pn.widgets.FileInput(
    name='Upload CSV (must include original iris feature column names)', accept='.csv'
)

def handle_file_upload(file_bytes):
    """Accept bytes and return a DataFrame with predicted_species or None.
    Uploaded CSV must use the original iris feature names (e.g. 'sepal length (cm)').
    """
    if not file_bytes:
        return None
    try:
        uploaded_df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as e:
        print("Error reading uploaded CSV:", e)
        return None

    required = list(iris['feature_names'])  # original names: 'sepal length (cm)', ...
    if not all(col in uploaded_df.columns for col in required):
        print("Uploaded CSV must contain the required columns:", required)
        return None

    uploaded_df = uploaded_df.copy()
    for c in required:
        uploaded_df[c] = pd.to_numeric(uploaded_df[c], errors='coerce')
    uploaded_df.dropna(subset=required, inplace=True)
    if uploaded_df.empty:
        return None

    predictions = model.predict(uploaded_df[required])
    uploaded_df['predicted_species'] = predictions
    return uploaded_df

@pn.depends(x_axis, y_axis, size, color, file_input)
def update_plot(x, y, s, c, file_bytes):
    """Return a Plotly fig showing original df and uploaded points (if any)."""
    uploaded = handle_file_upload(file_bytes)
    color_label = 'Species' if c == 'target' else c
    labels = {x: x, y: y, c: color_label}
    fig = px.scatter(df, x=x, y=y, size=s, color=c, labels=labels,
                     title=f'Scatter plot of {y} vs {x}')
    if uploaded is not None and not uploaded.empty:
        try:
            sizes = uploaded[s].astype(float).fillna(8).tolist()
        except Exception:
            sizes = [8] * len(uploaded)
        fig.add_trace(go.Scatter(
            x=uploaded[x],
            y=uploaded[y],
            mode='markers',
            marker=dict(size=sizes, symbol='x', color='black',
                        line=dict(width=2, color='DarkSlateGrey')),
            name='Uploaded Data'
        ))
    fig.update_layout(transition_duration=500)
    return fig

# DataFrame widget to show uploaded / predicted data (kept as a real DataFrame value)
uploaded_table = pn.widgets.DataFrame(pd.DataFrame(), width=800, height=200,
                                      sizing_mode="stretch_width", name='Uploaded Data Table')

def _update_uploaded_table(event):
    """Watch file_input.value and set uploaded_table.value to a real DataFrame."""
    df_uploaded = handle_file_upload(event.new)
    uploaded_table.value = df_uploaded if df_uploaded is not None else pd.DataFrame()

# initialize if already uploaded
if file_input.value:
    _update_uploaded_table(type("E", (), {"new": file_input.value}))

# watch for future uploads
file_input.param.watch(_update_uploaded_table, 'value')

# Single clean layout: sidebar + main content
layout = pn.Row(
    pn.Column(
        "# Controls",
        "## Select X and Y axes, Size, and Color options.",
        x_axis, y_axis, size, color,
        pn.Spacer(height=10),
        "## Upload CSV to predict (must include original iris feature column names).",
        file_input,
        pn.Spacer(height=10),
        f"## Model accuracy: {round(model_accuracy, 2)}%",
        width=320,
        sizing_mode="fixed",
    ),
    pn.Column(
        "# Iris Dataset Scatter Plot with Logistic Regression Predictions",
        pn.pane.Markdown("### Prediction for uploaded data"),
        pn.pane.Markdown("#### Uploaded data with predicted target species:"),
        uploaded_table,
        update_plot,  # pn will render the reactive plot here
        sizing_mode="stretch_width",
    ),
    sizing_mode="stretch_both",
)

layout.servable()

# run it at command line with    "panel serve panel_plotly_logistic_reg_csv.py --show --autoreload"