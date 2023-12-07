import re
import time
import streamlit as st
from predibase import PredibaseClient

pc = PredibaseClient()


def generate_sql(query_description: str, ):

    input_prompt = f"""Given an input question, use sqlite syntax to generate a sql query by choosing 
    one or multiple of the following tables. Write query in between <SQL></SQL>.

    For this Problem you can use the following database schema:
    <database_schema>
        1. Table: `employees`
           Columns:
               - employee_id (primary key)
               - first_name
               - last_name
               - title
               - salary
               - department
               - department_id (foregin key - referencing `departments.department_id`)
               - join_date
        2. Table: `departments`
           Columns:
               - department_id (primary key)
               - department_name
               - department_size
        3. Table: `sales`
           Columns:
               - sale_id (primary key)
               - date
               - product_id (foreign key - referencing `products.product_id`)
               - customer_id
               - quantity
               - unit_price
               - total_price
               - discount
               - seller_id (foreign key - referencing `employees.employee_id
               - payment_status
               - order_status
               - returned_flag
        4. Table: `products`
            Columns:
                - product_id (primary key)
                - product_name
                - description
                - category_id (foreign key referencing `categories.category_id`)
                - price
                - stock_quantity
                - supplier_id (foreign key referencing `suppliers.supplier_id`)
                - created_at
                - updated_at
        5. Table: `categories`
            Columns:
                - category_id (primary key)
                - category_name
                - description
                - created_at
                - updated_at
        6. Table: `suppliers`
            Columns:
                - supplier_id (primary key)
                - supplier_name
                - contact_name
                - address
                - city
                - postal_code
                - country
                - phone
                - email
                - created_at
                - updated_at
    </database_schema>
                
    Please provide the SQL query for this question: 
    Question: {query_description}
    Query: """

    llm = pc.LLM("pb://deployments/mistral-7b-instruct")

    response = llm.prompt(input_prompt, max_new_tokens=512)
    
    return response.response


def stream_sql(generated_sql: str):
    """
    Function for animating generation of text for the email components.

    :param generated_sql: The generated text to animate.
    :return: None
    """
    # Simulate stream of response with milliseconds delay

    extracted_sql = re.findall(r"<SQL>\\?n?(.*)\\?n?<\/SQL>", generated_sql, re.DOTALL)
    if extracted_sql:
        extracted_sql = extracted_sql[0]
    else:
        extracted_sql = generated_sql

    # Format SQL into multi-line block
    formatted_sql = "\n ".join(extracted_sql.splitlines()).split(" ")
    full_response = ""
    output_placeholder = st.empty()

    for token in formatted_sql:
        full_response += token + " "
        time.sleep(0.05)
        # Add a blinking cursor to simulate typing
        output_placeholder.code(body=full_response + "▌", language="sql")
    output_placeholder.code(body=full_response, language="sql")
