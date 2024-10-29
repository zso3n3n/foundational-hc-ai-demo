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

inference_config = {
    "endpoint": os.getenv("MIP_ENDPOINT"),
    "api_key": os.getenv("MIP_API_KEY"),
    "azureml_model_deployment": os.getenv("MIP_DEPLOY_NAME"),
}

def infer(image_path, text_prompt):
    image, masks, text_features = mip_utils.run_inference(
        inference_config, image_path, text_prompt
    )

    st.session_state.results['status'] = True
    st.session_state.results['image'] = image
    st.session_state.results['masks'] = masks
    st.session_state.results['text_features'] = text_features
    return  


#### START APP CODE #####

with st.sidebar as sb:
    st.write("Welcome to MedImageParse")
    # TODO add support for nii and dcm files
    uploaded_image = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

# Setup column page layout


if uploaded_image is not None:
    suffix = uploaded_image.name.split('.')[-1]
    with NamedTemporaryFile(delete=False, suffix = f".{suffix}") as f:
        f.write(uploaded_image.getbuffer())
        temp_path = f.name

    st.image(Image.open(temp_path), caption="Uploaded Image", use_column_width=True)

else:
    st.write("Upload an Image to get started <--")


prompt = st.text_input("Enter a prompt")
get_results = st.button("Submit")
if get_results:
    infer(temp_path, prompt)


if st.session_state.results['status']:
    st.write("Results")
    fig = mip_utils.plot_segmentation_masks(st.session_state.results['image'], st.session_state.results['masks'], prompt)
    st.pyplot(fig)
    #for mask in st.session_state.results['masks']:
     #   st.image(mask,use_column_width=True)
    st.write(f"Text features: {st.session_state.results['text_features']}")
