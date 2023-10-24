## Setup

Create a venv.

```
git clone git@github.com:predibase/examples.git
cd examples/pdf_qa

cd ..
python3 -m venv env
source env/bin/activate
```

Install requirements.

```
pip install -r requirements.txt
```

Set API keys in `.env` file. See `.env.example` for an example.

## Run the app

```
streamlit run app.py
```
