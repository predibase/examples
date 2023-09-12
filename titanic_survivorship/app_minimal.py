import os
import time

from predibase import PredibaseClient
from predibase.pql import get_session

import streamlit as st
import pandas as pd


# Get the api token, and set the serving endpoint for staging
token = os.getenv("PREDIBASE_API_TOKEN", "YOUR_API_TOKEN")
session = get_session(token=token)

# Get current user to output session tenant
pc = PredibaseClient(session)
deployment = pc.get_deployment("titanic_nn")


def run():
    st.set_page_config(page_title="Titanic Passenger Survival AI")
    st.markdown("# Titanic Passenger Survival AI")

    form = st.form(key='my_form')

    # Reference the dataset profile to design feature inputs.
    Sex = form.radio("Sex", ["male", "female"])

    age_min_value = float(0.17)
    age_max_value = float(80)
    Age = form.slider(
        "Age",
        min_value=age_min_value,
        max_value=age_max_value,
        value=(age_max_value - age_min_value) / 2,
        step=float(1),
    )

    SibSp_min_value = 0
    SibSp_max_value = 8
    SibSp = form.slider(
        "SibSp",
        min_value=SibSp_min_value,
        max_value=SibSp_max_value,
        value=int((SibSp_max_value - SibSp_min_value) / 2),
        step=1,
    )

    Embarked = form.radio("Embarked", ["S", "C", "Q"])

    Parch_min_value = 0
    Parch_max_value = 9
    Parch = form.slider(
        "Parch", min_value=0, max_value=9, value=int((Parch_max_value - Parch_min_value) / 2), step=1
    )

    PClass = form.radio("Pclass", ["1", "2", "3"])

    Fare_min_value = 0
    Fare_max_value = 512.329
    Fare = form.slider(
        "Fare", min_value=int(Fare_min_value), max_value=int(Fare_max_value),
        value=int((Fare_max_value - Fare_min_value) / 2), step=1
    )

    submit_button = form.form_submit_button(label='Predict Survival')

    spinner = st.spinner("Predicting...")
    if submit_button:
        with spinner:
            df = pd.DataFrame(
                [
                    {
                        "Sex": Sex,
                        "Age": Age,
                        "SibSp": SibSp,
                        "Embarked": Embarked,
                        "Parch": Parch,
                        "Pclass": PClass,
                        "Fare": Fare,
                    }
                ]
            )
            start = time.time()

            results = deployment.predict(df)

            prediction = results.iloc[0]['Survived_predictions']

            st.markdown(f"# {prediction}")
            bar_chart_data = pd.DataFrame(
                [
                    {"label": "False", "probability": results.iloc[0]["Survived_probabilities"][0]},
                    {"label": "True", "probability": results.iloc[0]["Survived_probabilities"][1]},
                ]
            )
            st.bar_chart(data=bar_chart_data, x="label", y="probability", use_container_width=True)

            end = time.time()
            st.write(f"Time to predict: {end - start:.2f} seconds")


if __name__ == "__main__":
    run()
