import streamlit as st
import os

from dotenv import load_dotenv,find_dotenv
from tempfile import NamedTemporaryFile
from PIL import Image
from src.utils import mip_processing_utils as mip_utils

load_dotenv(find_dotenv(), override = True)

# Initialize variables
uploaded_image = None
image_path = 'tmp/'
st.session_state.results = {}
st.session_state.results['status'] = False
site = None
slice_idx = None
hw_index = None
channel_idx = None

# Setup config
inference_config = {
    "endpoint": os.getenv("MIP_ENDPOINT"),
    "api_key": os.getenv("MIP_API_KEY"),
    "azureml_model_deployment": os.getenv("MIP_DEPLOY_NAME"),
}

# Helper functiont to run inference and set results
def infer(image_path, text_prompt):
    image, masks, text_features = mip_utils.run_inference(
        inference_config, image_path, text_prompt
    )

    st.session_state.results['status'] = True
    st.session_state.results['image'] = image
    st.session_state.results['masks'] = masks
    st.session_state.results['text_features'] = text_features
    return  

#########################
#### START APP CODE #####
#########################

# Sidebar 

with st.sidebar as sb:
    st.write("Welcome to MedImageParse")
    
    uploaded_image = st.file_uploader("Upload Image", type=["png","jpg","jpeg","nii","dcm","nii.gz"])
    suffix = uploaded_image.name.split('.')[-1]
    if suffix == 'gz':
        suffix = uploaded_image.name.split('.')[-2] + '.' + suffix

    if uploaded_image is not None and suffix == 'dcm':
        ct = st.radio("Is this a CT scane image?", ("Yes", "No"), index = 1)
    elif uploaded_image is not None and (suffix == 'nii' or suffix == 'nii.gz'):
        slice_idx = st.slider("Select Slice", 0, 100, 0)
        hw_index = st.text_input("Enter Height and Width index", "(0,1)")
        ct = st.radio("Is this a CT scane image?", ("Yes", "No"), index = 1)
    
    if ct == "Yes":
        site = st.radio("What is the CT site?", ("Abdomen", "Lung", "Pelvis", "Liver", "Colon", "Pancreas")).lower()

# Main Page

if uploaded_image is not None:
    with NamedTemporaryFile(delete=False, suffix = f".{suffix}") as f:
        f.write(uploaded_image.getvalue())
        temp_path = f.name
    
    image_array = mip_utils.display_image(temp_path, ct=="Yes", slice_idx, hw_index, channel_idx, site)
    st.image(image_array, use_column_width=True)
    # st.image(temp_path, use_column_width=True)

    prompt = st.text_input("Enter a prompt")
    get_results = st.button("Submit")
    if get_results:
        infer(temp_path, prompt)

else:
    st.write("Upload an Image to get started <--")

# Display Reusults
if st.session_state.results['status']:
    st.write("Results")
    fig = mip_utils.plot_segmentation_masks(st.session_state.results['image'], st.session_state.results['masks'], prompt)
    st.pyplot(fig)
    st.write(f"Text features: {st.session_state.results['text_features']}")
