# Demo: Model Augmented Generation

This repo contains a Streamlit app showcasing the combination of Supervised ML with LLM generation.
The demo example recommends fashion products to a user and then generates an email tailored to the customer and the 
recommended product.

## Usage
Enter AWS credentials in `.streamlit/secrets.toml`, then run the Streamlit app:

```shell
streamlit run app.py
```

## Relevant Files

* `app.py`: Streamlit app
* `utils.py`: Utility functions for the app
* `Personalized_Email_Campaigns.ipynb`: Jupyter notebook with the full example tutorial
