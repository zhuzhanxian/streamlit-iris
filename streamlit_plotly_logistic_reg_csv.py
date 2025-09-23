#Implement the same functionality as dash_plotly_logistic_reg_csv.py but using Streamlit
import pandas as pd
import numpy as np  
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import plotly.graph_objects as go

#load iris dataset
iris = load_iris()  
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                  columns= iris['feature_names'] + ['target'])
df['target'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
#train a logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(df.iloc[:,:-1], df['target'])
predictions = model.predict(df.iloc[:,:-1])
model_accuracy =model.score(df.iloc[:,:-1], df['target'])*100

import streamlit as st
st.title("Iris Dataset Scatter Plot with Logistic Regression Predictions")

#sidebar for inputs
st.sidebar.header("Select X and Y axes, Size, and Color options for scatter plot")    
x_axis = st.sidebar.selectbox('X Axis', options=list(df.columns[:-1]), index=0)
y_axis = st.sidebar.selectbox('Y Axis', options=list(df.columns[:-1]), index=1)
size = st.sidebar.selectbox('Size', options=list(df.columns[:-1]), index=2)
color = st.sidebar.selectbox('Color', options=list(df.columns), index=4)
st.write("\n")


uploaded_file = st.sidebar.file_uploader("Upload a CSV file with the same feature columns to see predictions.",
                                          type="csv")   
def handle_file_upload(uploaded_file):
    """Accept uploaded_file (st.file_uploader) and return a DataFrame with predicted_species or None."""
    if not uploaded_file:
        return None
    try:
        uploaded_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading uploaded CSV: {e}")
        return None
    required = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    if not all(col in uploaded_df.columns for col in required):
        st.error(f"Uploaded CSV must contain the required columns: {required}")
        return None
    # coerce to numeric and drop rows with missing required values
    uploaded_df = uploaded_df.copy()
    for c in required:
        uploaded_df[c] = pd.to_numeric(uploaded_df[c], errors='coerce')
    uploaded_df.dropna(subset=required, inplace=True)
    if uploaded_df.empty:
        st.error("Uploaded CSV has no valid rows after cleaning.")
        return None
    # make predictions
    predictions = model.predict(uploaded_df[required])
    uploaded_df['target'] = predictions
    return uploaded_df
uploaded_df = handle_file_upload(uploaded_file)


fig = px.scatter(df, x=x_axis, y=y_axis, size=size, color=color,
                 title="Iris Dataset Scatter Plot with Logistic Regression Predictions",
                 labels={x_axis: x_axis, y_axis: y_axis, color: color})

if uploaded_df is not None:
    st.success("File uploaded and processed successfully!")
    st.write(f"#### Uploaded data with predicted target species:")
    st.dataframe(uploaded_df)
else:
    st.info("Awaiting CSV file to be uploaded.")


st.write(f"#### Prediction accuracy on original data: {round(model_accuracy,2)}%")

if uploaded_df is not None:
    fig.add_trace(
        go.Scatter( x=uploaded_df[x_axis], y=uploaded_df[y_axis], mode='markers',
                    marker=dict(size=uploaded_df[size], symbol='x', color='black',
                                line=dict(width=2, color='DarkSlateGrey')),
                    name ="predicted target for uploaded data")  )
fig.update_layout(transition_duration=500)    

st.plotly_chart(fig, use_container_width=True)

# run with "streamlit run streamlit_plotly_logistic_reg_csv.py" on command line
