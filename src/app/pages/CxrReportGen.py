import streamlit as st
import os

from dotenv import load_dotenv,find_dotenv
from tempfile import NamedTemporaryFile
from PIL import Image
from src.utils import cxr_utils

load_dotenv(find_dotenv(), override = True)

# Initialize variables
frontal_image = None
lateral_image = None
generate_results = False
st.session_state.results = {}
st.session_state.results['status'] = False

inference_config = {
    "endpoint": os.getenv("CXR_ENDPOINT"),
    "api_key": os.getenv("CXR_API_KEY"),
    "azureml_model_deployment": os.getenv("CXR_DEPLOY_NAME"),
}

#########################
#### START APP CODE #####
#########################

# SIDEBAR
with st.sidebar as sb:
    frontal_image = st.file_uploader("Upload Frontal Image (Required)", type=["png","jpg","jpeg"])
    lateral_image = st.file_uploader("Upload Lateral Image (Optional)", type=["png","jpg","jpeg"])
    st.markdown("""
        ### 🔎 Need a Sample Image?
        [**MedPix**](https://medpix.nlm.nih.gov/advancedsearch) - _National Library of Medicine_   
        
    """)

# MAIN PAGE
st.title("Microsoft CxRReportGen")
st.markdown("##### _\"Grounded report generation with localized findings for x-ray images.\"_")

tab1, tab2 = st.tabs(["✅ Test the Model", "📖 Learn More"])

with tab1:
    header = st.empty()
    if frontal_image and lateral_image:
        images = [frontal_image, lateral_image]
        header.image(images, width=350, caption=['Frontal', 'Lateral'])
        generate_results = st.button("Generate Report")
        pass
    elif frontal_image:
        header.image(frontal_image, use_column_width=True)
        generate_results = st.button("Generate Report")
    elif lateral_image:
        header.warning("Frontal image is required to generate a report. Please upload a frontal image as well.")
    else:
        header.subheader("⬅️ Upload an Image to Get Started...")

    if st.session_state.results['status']:
        header.pyplot(cxr_utils.show_image_with_bbox(frontal_image, st.session_state.results['findings'], lateral_image))
        st.write(f"**Findings:** {st.session_state.results['findings']}")

    if generate_results:
        with st.spinner():
            findings = cxr_utils.score_image(frontal_image, lateral_image, "", "", "None")
            st.session_state.results['status'] = True
            st.session_state.results['findings'] = findings

    

with tab2:
    st.markdown("""
    ### 🌐 Overview  
    The CXRReportGen model utilizes a multimodal architecture, integrating a BiomedCLIP image encoder with a Phi-3-Mini text encoder to help an application interpret complex medical imaging studies of chest X-rays. CXRReportGen follows the same framework as MAIRA-2. When built upon and integrated into an application, CXRReportGen may help developers generate comprehensive and structured radiology reports, with visual grounding represented by bounding boxes on the images.
    
    ---
    ### 🏗️ Model Architecture  
    The CXRReportGen model combines a radiology-specific image encoder with a large language model and takes as inputs a more comprehensive set of data than many traditional approaches. The input data includes the current frontal image, the current lateral image, the prior frontal image, the prior report, and the indication, technique, and comparison sections of the current report. These additions significantly enhance report quality and reduce incorrect information, ultimately demonstrating the feasibility of grounded reporting as a novel and richer task in automated radiology.
    """)
    st.image(Image.open("documentation_images/cxr.png"), use_column_width=True)        
    st.markdown("""     
    ---
    ### 🗒️ More Details  
    - Microsoft's Responsible AI Principles and approach available [here](https://www.microsoft.com/en-us/ai/principles-and-approach/)
    - The license for CXRReportGen is the MIT license. For questions or comments, please contact: hlsfrontierteam@microsoft.com
    """)    