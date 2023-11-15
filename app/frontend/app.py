import streamlit as st
import pandas as pd

context = None

# set title
st.title("dataChain")

# create sidebar
sidebar = st.sidebar

# create clickable tabs
current_tab, history_tab = st.tabs(["Current Session", "History"])

# set session history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "charts" not in st.session_state:
    st.session_state.charts = []

if "current_session" not in st.session_state:
    st.session_state.current_session =  []

# prompt user for file upload
with sidebar:
    st.title("Data Dashboard")
    upload = st.file_uploader("", type=["CSV"])

# populate sidebar after upload
if upload is not None:
    data = pd.read_csv(upload)
    # TODO: MAKE THIS MORE EXTENTABLE
    if 'date' in data.columns:
        data["date"] = pd.to_datetime(data["date"])

    # store data in session
    st.session_state.data = data
    with sidebar:
        st.write("Data is uploaded")
        initial_data = st.dataframe(data)

    # svae data as context
    context = DataContext()
    context.set_data(data)

def display_messages() -> None:
    for message in st.session_state.messages:
        st.markdown(message["content"])

def display_charts() -> None:
    # append charts to history tab
    with history_tab:
        for chart in reversed(st.session_state.charts):
            st.markdown(f"Prompt: {chart['name']} Time: {chart['timestamp']}")
            st.image(chart["path"])

            with open(chart["path"], "rb") as image:
                hist_btn = st.button(label = "view code", key = 'a' + str(chart["timestamp"]))
                if hist_btn:
                    with sidebar: #TODO: make this a function
                        initial_data.empty()
                        # JUST WRITE OUT IN STEPS WITH WHAT WAS DONE

prompt = st.chat_input("What can I help you with?")

# message loop
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    #TODO: 
    # THIS IS WHERE WE ACTUALLY START THE PIPELINE
    # Get context, actually run the backend streamlit
    # maybe use planner to do functions, save outputs into variables and also
    # into our history (streamlit history and embeddings for memory)
    # then output them here into the front end