import streamlit as st
from util import stream_sql, generate_sql

import logging

logger = logging.getLogger(__name__)


#####################
# Application Setup #
#####################

# Page title
st.title("Intelligent SQL Generator")

############
# Sidebar #
############

# Sidebar title
st.sidebar.title("Database Structure")

# Sidebar description
st.sidebar.markdown(
    """
    This application is a demo of an intelligent SQL generator. It uses the following database schema:
    """
)

# Side bar database schema
st.sidebar.markdown(
    """
    1. Table: `employees` \n
       Columns: \n
           - employee_id (primary key)
           - first_name
           - last_name
           - title
           - salary
           - department
           - department_id (foregin key)
           - join_date
    2. Table: `departments` \n
       Columns: \n
           - department_id (primary key)
           - department_name
           - department_size
    3. Table: `sales` \n
       Columns: \n
           - sale_id (primary key)
           - date
           - product_id (foreign key)
           - customer_id
           - quantity
           - unit_price
           - total_price
           - discount
           - seller_id (foreign key)
           - payment_status
           - order_status
           - returned_flag
    4. Table: `products` \n
        Columns: \n
           - product_id (primary key)
           - product_name
           - description
           - category_id (foreign key)
           - price
           - stock_quantity
           - supplier_id (foreign key)
           - created_at
           - updated_at
    5. Table: `categories` \n
        Columns: \n
           - category_id (primary key)
           - category_name
           - description
           - created_at
           - updated_at
    """
)

###############
# Query input #
###############

input_query = st.text_area(
    label="Query Description",
    value=None,
)

# Display SQL
st.subheader("SQL")

if input_query is not None:
    with st.spinner('Generating SQL...'):

        # Generate SQL
        generated_sql = generate_sql(input_query)

        # Display SQL
        stream_sql(generated_sql)
