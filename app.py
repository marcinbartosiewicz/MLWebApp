import streamlit as st
from streamlit_extras.stateful_button import button as extrabutton
import pycaret.regression as pr
import pycaret.classification as pc
from pandas_profiling import ProfileReport
import pandas as pd
from streamlit_pandas_profiling import st_profile_report


def modelling(type, chosen_target, df):
    type.setup(df, target=chosen_target)
    setup_df = type.pull()
    st.dataframe(setup_df)
    best_model = type.compare_models()
    compare_df = type.pull()
    st.dataframe(compare_df)
    pr.save_model(best_model, 'best_model')
    st.session_state['best_model']=True


if 'dataset' in st.session_state:
    df = pd.read_csv(st.session_state['dataset'], index_col=None)

with st.sidebar:
    st.image(
        "https://previews.123rf.com/images/tumsasedgars/tumsasedgars1901/tumsasedgars190100183/119004326-machine-learning-concept-chart-with-keywords-and-icons-on-white-background.jpg")
    st.title("Auto ML App")
    choice = st.radio("Navigation", ["Upload", "Profiling", "Modelling", "Download"])
    st.info("This project application helps you build and explore your data.")

if choice == "Upload":
    st.title("Upload Your Dataset")

    file = st.file_uploader("Upload Your Dataset in CSV format", type='csv')
    if extrabutton("Click If you don't have your dataset. You can use one of two example datasets.", key="upload1"):

        chosen_dataset = st.selectbox('Choose the Dataset',
                                      ['Titanic(Classification)', 'DiamondPricePrediction(Regression)'])
        if extrabutton("If you choose, then click", key="upload2"):
            st.write("You choose:", chosen_dataset)
            if chosen_dataset == 'Titanic(Classification)':
                file = 'titanic.csv'
            else:
                file = 'diamond.csv'

    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)
        st.session_state['dataset'] = 'dataset.csv'
        st.success('Uploading completed!')

if choice == "Profiling":
    try:
        st.title("Exploratory Data Analysis")
        profile = ProfileReport(df, title="Dataset Profile Report")
        st_profile_report(profile)
    except NameError:
        st.error('You need to upload file first', icon="🚨")

if choice == "Modelling":
    st.title("Modelling")
    try:
        chosen_target = st.selectbox('Choose the Target Column', df.columns)
        typeofproblem = st.selectbox('Is your task Classification or Regression?', ['Classification', 'Regression'])
        st.session_state['typeofproblem'] = typeofproblem
        if extrabutton("Click if you have selected the type of task", key="typeofproblem2"):

            if df[chosen_target].isnull().sum() > 0:

                st.write(f"It looks like your target column has null({df[chosen_target].isnull().sum()}) values.")
                st.write('To Run Modelling, we need to remove null values')
                chosen_method = st.selectbox('Choose the Method to remove null values',
                                             ['fill with median', 'fill with mean', 'drop'])
                if extrabutton('Run remove null', key='nullremover'):
                    if chosen_method == 'fill with median':
                        median_target = df[chosen_target].median()
                        df[chosen_target].fillna(median_target, inplace=True)

                    elif chosen_method == 'fill with mean':
                        mean_target = df[chosen_target].mean()
                        df[chosen_target].fillna(mean_target, inplace=True)
                    elif chosen_method == 'drop':
                        df.dropna(subset=[chosen_target], inplace=True)
                    df.to_csv('dataset.csv', index=None)

            if not df[chosen_target].isnull().any():
                st.write("Your Target does not have null values")
                if extrabutton('Run Modelling',key='runmodelling'):

                    with st.spinner('Running Modelling...'):

                        st.warning('This may take up to several minutes',icon='⏳')
                        try:
                            if typeofproblem == 'Regression':
                                modelling(type=pr, chosen_target=chosen_target, df=df)
                            else:
                                modelling(type=pc, chosen_target=chosen_target, df=df)

                        except ValueError:
                            st.error("Something went wrong")

                    st.success('Modelling completed!')
                    st.write('You can download the best model for your dataset from the Download section')
    except NameError:
        st.error('You need to upload file first', icon="🚨")

if choice == "Download":
    st.title("Download")
    if 'best_model' in st.session_state:
        try:
            with open('best_model.pkl', 'rb') as f:
                st.download_button('Download Model', f, file_name="best_model.pkl")
        except FileNotFoundError:
            st.error('Something went wrong', icon="🚨")
    else:
        st.error('First you need to do modeling', icon="🚨")
