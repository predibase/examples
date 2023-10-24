import os
import time

from predibase import PredibaseClient
from predibase.pql import get_session

import streamlit as st
import pandas as pd


def run():
    st.set_page_config(page_title="Titanic Passenger Survival AI", layout="wide")
    st.markdown("""<h1 style="text-align:center;">Titanic Passenger Survival AI</h1>""", unsafe_allow_html=True)

    st.markdown(
        """
        <p style="text-align:center;">
        <img src="https://app.predibase.com/logos/predibase/predibase.svg" width="25" />
        Powered by <a href="https://predibase.com">Predibase</a>
        </p>
        <p style="text-align:center;">
        <a href="https://www.kaggle.com/c/titanic">Data</a> from<img src="https://www.kaggle.com/static/images/logos/kaggle-logo-transparent-300.png" width="50"/>
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    left_column, center_column, right_column = st.columns(3)

    with left_column:
        st.markdown("""The sinking of the Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the widely considered "unsinkable" RMS Titanic sank after colliding with an iceberg.

Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

This app showcases models that predict "what sorts of people were more likely to survive?" using passenger data (ie name, age, gender, socio-economic class, etc).""", unsafe_allow_html=True)
        st.image("titanic.png", caption="""An AI-generated image of the Titanic.""")

    # Get the api token, and set the serving endpoint for staging
    token = os.getenv("PREDIBASE_API_TOKEN", "YOUR_API_TOKEN")
    session = get_session(token=token)

    # Get current user to output session tenant
    pc = PredibaseClient(session)

    # Set up 2 columns.
    # left_column, center_column, right_column = st.columns([1, 1, 1])

    form = center_column.form(key='my_form')

    # Reference the dataset profile to design feature inputs.
    with center_column:
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

    with right_column:
        deployment_name = st.selectbox(
            "Supervised Deployment",
            [pc.get_deployment("titanic_nn").name, pc.get_deployment("titanic").name],
            index=0,
            help="The supervised model used for prediction.",
        )

        deployment = pc.get_deployment(deployment_name)

        submit_button = form.form_submit_button(label='Predict Survival')

        right_column_spinner = st.spinner("Predicting...")
        if submit_button:
            st.markdown("""[Model version](https://app.predibase.com/models/version/2930)""")

            with right_column_spinner:
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
                color = 'green' if prediction else 'red'

                st.markdown(
                    f"<h1 style='text-align: center; color: {color};'>{prediction}!</h1>",
                    unsafe_allow_html=True,
                )
                bar_chart_data = pd.DataFrame(
                    [
                        {"label": "False", "probability": results.iloc[0]["Survived_probabilities"][0]},
                        {"label": "True", "probability": results.iloc[0]["Survived_probabilities"][1]},
                    ]
                )
                st.bar_chart(data=bar_chart_data, x="label", y="probability", use_container_width=True)

                end = time.time()
                st.write(f"Time to predict: {end - start:.2f} seconds")

                info_expander = st.expander("Model information")
                with info_expander:
                    st.markdown(f"""
                    - [Model version](https://staging.predibase.com/models/version/37324)
                    - Deployment version: {deployment.deployment_version}
                    - Engine name: {deployment.engine_name}
                    - Model name: {deployment.model_name}
                    - Model version: {deployment.model_version}
                    - Deployment URL: {deployment.deployment_url}
                    """)

                curl_expander = st.expander("Sample curl command")
                with curl_expander:
                    sample_curl_command = ("""

            curl -v -H "Authorization: Bearer $PREDIBASE_API_TOKEN" "https://serving.staging.predibase.com/7c7efa/deployments/v2/models/justin-titanic/infer" \
            -d @- <<EOM | jq "."
            {
            "id": "1",
            "inputs": [
                {"name": "Pclass", "shape": [1], "datatype": "BYTES", "data": ["3"]},
                {"name": "Sex", "shape": [1], "datatype": "BYTES", "data": ["male"]},
                {"name": "Age", "shape": [1], "datatype": "BYTES", "data": ["34.5"]},
                {"name": "SibSp", "shape": [1], "datatype": "BYTES", "data": ["0.0"]},
                {"name": "Parch", "shape": [1], "datatype": "BYTES", "data": ["0.0"]},
                {"name": "Fare", "shape": [1], "datatype": "FP64", "data": [7.8292]},
                {"name": "Embarked", "shape": [1], "datatype": "BYTES", "data": ["Q"]},
                {"name": "split", "shape": [1], "datatype": "BYTES", "data": ["1"]}
            ]
            }
            EOM
                    """)

                    st.code(sample_curl_command, language="bash")

                sample_code = (f"""
        from predibase import PredibaseClient
        from predibase.pql import get_session

        import pandas as pd


        # Get the api token, and set the serving endpoint for staging
        token = os.getenv("PREDIBASE_API_TOKEN", "YOUR_API_TOKEN")
        session = get_session(token=token)

        # Get current user to output session tenant
        pc = PredibaseClient(session)

        deployment = pc.get_deployment('{deployment.name}')""") + ("""

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

        results = deployment.predict(df)
                """)

                code_expander = st.expander("Sample python code")
                with code_expander:
                    st.code(sample_code, language="python")

                st.markdown("""## Explainability using Llama-2-13b-chat""")

                llm = pc.LLM("pb://deployments/llama-2-13b-chat")

                responses = llm.prompt(
                    f"""A machine learning model has predicted that this passenger on the Titanic {'survived' if results.iloc[0]['Survived_predictions'] else 'did not survive'}.

Here is information about the passenger:

Sex: {Sex},
Age: {Age},
SibSp: {SibSp},
Embarked: {Embarked},
Parch: {Parch},
Pclass: {PClass},
Fare: {Fare}

Why did the model make this prediction?""",
                    options={"max_new_tokens": 500}
                )
                for r in responses:
                    st.write(r.response)


if __name__ == "__main__":
    run()
