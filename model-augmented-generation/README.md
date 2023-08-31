# Demo: Model Augmented Generation

This repo contains a Streamlit app showcasing the combination of Supervised ML with LLM generation.
The demo example recommends fashion products to a user and then generates an email tailored to the customer and the 
recommended product.

## Tutorial
- Personalized_Email_Campaigns.ipynb: Jupyter notebook with the full example tutorial

In this tutorial, we walk you through an end to end example of how to use Predibase to train a recommender system model
for product recommendations and then use the output of that model to generate personalized email campaigns for each user.

## Sample Application
- app.py: Streamlit app
- utils.py: Utility functions for the app
- recommendations.csv: Sample recommendations used in the application

## Usage
Enter AWS credentials in `.streamlit/secrets.toml`, then run the Streamlit app:

```shell
streamlit run app.py
```