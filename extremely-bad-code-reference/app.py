import streamlit as st
import pandas as pd
import json
import time
from kernel import main
import asyncio
import re

sidebar = st.sidebar
current_tab, history_tab, auto_hist_tab = st.tabs(["Current Session", "AutoPilot History", "CoPilot History"])

# set session history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "charts" not in st.session_state:
    st.session_state.charts = []

if "alt_charts" not in st.session_state:
    st.session_state.alt_charts = []

if "current_session" not in st.session_state:
    st.session_state.current_session = []

# prompt user for file upload
with sidebar:
    st.title("Data Dashboard")

    # data uploader
    data_upload = st.file_uploader("Data", type=["CSV"])

    if data_upload is not None:
        data_object = pd.read_csv(data_upload)
        if 'date' in data_object.columns:
            data_object["date"] = pd.to_datetime(data_object["date"])
        
        # store data in session
        st.session_state.data = data_upload

        # write data
        with sidebar:
            st.write("I uploaded the data!")
            initial_data = st.dataframe(data_object)
    
    
        # # Create the JSON-like structure with column names, types, and descriptions
        # column_details = {}
        # for column in data_object.columns:
        #     column_details[column] = {
        #     "type": str(data_object[column].dtype),
        #     "description": descriptions_dict.get(column, "No description available.")
        # }

        # Convert the structure to a JSON-like string
        #descriptions = json.dumps(column_details, indent=4)

prompt = st.chat_input("What can I help you with?")
if prompt:
    try:
        st.text("EXECUTING PLANNER")
        plan, gpt_output = asyncio.run(main(prompt))
        st.code(plan)
        st.code(gpt_output)
        # code_pattern = re.compile(r"\'\'\'(.*?)\'\'\'", re.DOTALL)
        # match = code_pattern.search(gpt_output)
        # if match:
        #     executable = match.group(1)
        #     print(executable)
    except Exception as e:
        st.write("Something went wrong when running the LLM\n", str(e))

