# Model Augmented Generation
__Personalized Email Campaigns__

This repo contains both a tutorial and a Streamlit app showcasing the combination of Supervised ML with LLM generation.

<br/>

## Tutorial
- Personalized_Email_Campaigns.ipynb: Jupyter notebook with the full example tutorial

In this tutorial, we walk you through an end to end example of how to use Predibase to train a recommender system model
for product recommendations and then use the output of that model to generate personalized email campaigns for each user.

## Sample Application

This sample application recommends fashion products to a user and then generates an email tailored to the customer and the
recommended product.

### Files
- app.py: Streamlit app
- utils.py: Utility functions for the app
- recommendations.csv: Sample recommendations used in the application

### Usage

1. Install the requirements
    ```shell
    pip install -r requirements.txt
    ```

2. Set your Predibase API token as an environment variable
    ```shell
    export PREDIBASE_API_TOKEN=<your-token>
    ```

3. Run the app
    ```shell
    streamlit run app.py
    ```