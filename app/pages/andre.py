import streamlit as st
import pandas as pd
import os
from backend.sage_backend import Autopilot, s3Setup
#import nbformat
from nbconvert import HTMLExporter
import subprocess

st.markdown("<h1 style='text-align: center; font-weight:bold; font-family:candara; padding-top: 0rem;'> sageMaker</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;padding-top: 0rem;'>Notebook Generation with AWS</h2>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;padding-top: 0rem;'>by Andre and The Boys</h5>", unsafe_allow_html=True)

chosen_dataset = None
chosen_model = None
prompt = None

if "datasets" not in st.session_state:
    datasets = {}
    st.session_state["datasets"] = datasets

else:
    datasets = st.session_state["datasets"]

if "s3" not in st.session_state:
    st.session_state["s3"] = s3Setup()

else:
    s3 = st.session_state["s3"]

if "autopilot" not in st.session_state:
    st.session_state["autopilot"] = Autopilot("streamlittest") #TODO: change job name

else:
    autopilot = st.session_state["autopilot"]

#name of bucket within aws where data is
bucket_name = 'testbucketandre1'
#name of data that you want to use for ML experiment
object_name = 'data.csv'
file_name = 'data.csv'
#this stays the same, but is an AWS role capable of running experiments
role = 'arn:aws:iam::006176443241:role/service-role/AmazonSageMaker-ExecutionRole-20231107T235015'
#name of file you want to output all of the experiment data too, change everytime you run a new experiment
local_dir = 'sagemaker_output' #TODO: change 

# create side bar
with st.sidebar:
    st.markdown("<h2 style='text-align: center;padding-top: 0rem;'>Data Dashboard</h2>", unsafe_allow_html=True)
    data_container = st.empty()
    uploaded_file = st.file_uploader("", type=["CSV"])

    # radio buttons
    index_no = 0
    if uploaded_file:
        file_name = uploaded_file.name
        datasets[file_name] = pd.read_csv(uploaded_file)
        index_no = len(datasets) - 1

    # Radio buttons for dataset choice
    chosen_dataset = data_container.radio("Choose your data:", datasets.keys(), index=index_no)


if bool(datasets):
    st.dataframe(datasets[chosen_dataset])
    target_variable = st.selectbox("",datasets[chosen_dataset].columns)
    file_path = os.path.join("data//", chosen_dataset) #TODO: error handle here
    st.write("Perform analyze with target variable: " + target_variable)
    if st.button("Proceed"):
        st.write("""DEVELOPER MESSAGE: For time sake, we will assume SageMaker run 
                 is completed and appropriate files have returned. This has been tested.""")
        #st.write(os.getcwd())

        # end to end autopilot experiment
        
        #if s3.upload_file_to_bucket(bucket_name, file_path, object_name):
        #    st.write("File successfully uploaded to bucket.")
        
        #contents = s3.list_bucket_contents(bucket_name)

        #if contents is not None:
        #    st.write("Bucket contents:")
        #    st.write(contents)

        #autopilot.run_autopilot_job(role, bucket_name, file_name, target_variable, autopilot.job_name)
        #autopilot.extract_autopilot_product(autopilot.job_name, local_dir)
         #TODO: HERE
        # #candidate_notebook_path = os.path.join(local_dir, "//SageMakerAutopilotCandidateDefinitionNotebook.ipynb")
        # notebook_path = os.path.join(local_dir, "SageMakerAutopilotDataExplorationNotebook.ipynb")
        # #notebook_path = os.path.join(local_dir, "0dinoMIT.ipynb")

        # result = subprocess.run(["jupyter-nbconvert", "--to", "html", "--template basic", notebook_path, "--output-dir", "html_notebooks"], check=True)
        # if result.returncode == 0:
        #     st.write("Notebook converted successfully.")
        # else:
        #     st.write("Error in notebook conversion:", result.stderr)
        #TODO: HERE

        # # Load the Jupyter Notebook
        # with open(notebook_path, 'r', encoding='utf-8') as notebook_file:
        #     notebook_content = nbformat.read(notebook_file, as_version=4)

        # # Create an HTML exporter
        # html_exporter = HTMLExporter()

        # # Convert the notebook to HTML
        # (html_output, resources) = html_exporter.from_notebook_node(notebook_content)

        # # Specify the output directory and file name
        # output_dir = 'html_notebooks'
        # output_file_name = 'output.html'
        # output_file_path = f'{output_dir}/{output_file_name}'

        # # Save the HTML output to a file
        # with open(output_file_path, 'w', encoding='utf-8') as output_file:
        #     output_file.write(html_output)

        # # Check if the conversion was successful
        # if html_output:
        #     print("Notebook converted successfully.")
        # else:
        #     print("Error in notebook conversion.")
