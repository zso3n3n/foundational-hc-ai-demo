import streamlit as st
import os

from dotenv import load_dotenv,find_dotenv
from tempfile import NamedTemporaryFile
from PIL import Image
from src.utils import mip_processing_utils as mip_utils

load_dotenv(find_dotenv(), override = True)

# Initialize variables
uploaded_image = None
st.session_state.results = {}
st.session_state.results['status'] = False

inference_config = {
    "endpoint": os.getenv("MIP_ENDPOINT"),
    "api_key": os.getenv("MIP_API_KEY"),
    "azureml_model_deployment": os.getenv("MIP_DEPLOY_NAME"),
}

#########################
#### START APP CODE #####
#########################

# SIDEBAR
with st.sidebar as sb:
    # TODO add support for nii and dcm files
    st.warning("Support for NIFTI and DICOM images coming soon")
    uploaded_image = st.file_uploader('Upload Image:', type=["png","jpg","jpeg"])
    st.container()
    st.markdown('---')
    st.markdown("""
        ### 🔎 Need a Sample Image?
        [**MedPix**](https://medpix.nlm.nih.gov/advancedsearch) - _National Library of Medicine_   
        
    """)

# MAIN PAGE
st.title("Microsoft MedImageParse")
st.markdown("###### _\"A biomedical foundation model for image parsing of everything, everywhere, all at once.\"_")

tab1, tab2 = st.tabs(["✅ Test the Model", "📖 Learn More"])

with tab1:
    header = st.empty()
    if uploaded_image is not None:
        suffix = uploaded_image.name.split('.')[-1]
        with NamedTemporaryFile(delete=False, suffix = f".{suffix}") as f:
            f.write(uploaded_image.getbuffer())
            temp_path = f.name

        header.image(Image.open(temp_path), use_column_width=True)
        prompt = st.text_input("What would you like to identify?", help="Use format 'object 1 & object 2 & ... & object X' for multi-segmentation")
        get_results = st.button("Submit")
        if get_results:
            with st.spinner("Processing..."):
                image, masks, text_features = mip_utils.run_inference(inference_config, temp_path, prompt)
                st.session_state.results['status'] = True
                st.session_state.results['image'] = image
                st.session_state.results['masks'] = masks
                st.session_state.results['text_features'] = text_features

    else:
        header.subheader("⬅️ Upload an Image to Get Started...")

    st.container()
    st.markdown('---\n')
    st.markdown("#### Supported Modalities")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
                    - X-Ray
                    - MRI
                    - CT
                    - Endoscope
                   
                    """)
    with col2:
        st.markdown("""
                    - Pathology
                    - Ultrasound
                    - Fundus
                    - Dermoscopy
                    """)

    if st.session_state.results['status']:
        fig = mip_utils.plot_segmentation_masks(st.session_state.results['image'], st.session_state.results['masks'], prompt)
        header.pyplot(fig)
        # st.write(f"Text features: {st.session_state.results['text_features']}")


with tab2:
    st.markdown("""
    ### 🌐 Overview  
    Biomedical image analysis is fundamental for biomedical discovery in cell biology, pathology, radiology, and many other biomedical domains. MedImageParse is a biomedical foundation model for imaging parsing that can jointly conduct segmentation, detection, and recognition across 9 imaging modalities. Through joint learning, we can improve accuracy for individual tasks and enable novel applications such as segmenting all relevant objects in an image through a text prompt, rather than requiring users to laboriously specify the bounding box for each object.

    MedImageParse is broadly applicable, performing image segmentation across 9 imaging modalities.

    MedImageParse is also able to identify invalid user inputs describing objects that do not exist in the image. MedImageParse can perform object detection, which aims to locate a specific object of interest, including on objects with irregular shapes.

    On object recognition, which aims to identify all objects in a given image along with their semantic types, MedImageParse can simultaneously segment and label all biomedical objects in an image.

    In summary, MedImageParse shows potential to be a building block for an all-in-one tool for biomedical image analysis by jointly solving segmentation, detection, and recognition.

    It is broadly applicable to all major biomedical image modalities, which may pave a future path for efficient and accurate image-based biomedical discovery when built upon and integrated into an application.
    
    ---         
    ### 🏗️ Model Architecture
    MedImageParse is built upon a transformer-based architecture, optimized for processing large biomedical corpora. Leveraging multi-head attention mechanisms, it excels at identifying and understanding biomedical terminology, as well as extracting contextually relevant information from dense scientific texts. The model is pre-trained on vast biomedical datasets, allowing it to generalize across various biomedical domains with high accuracy.
    
    ---            
    ### 🗒️ More Details
    - For more details regarding evaluation results, ethical considerations and limitations, training information, and fairness evaluations please refer to the [MedImageParse Paper](https://arxiv.org/abs/2405.12971)
    and the cooresponding [GitHub repository](https://microsoft.github.io/BiomedParse/assets/BiomedParse_arxiv.pdf).  
    - Microsoft's Responsible AI Principles and approach available [here](https://www.microsoft.com/en-us/ai/principles-and-approach/)
    - The license for MedImageParse is the MIT license. Please cite the paper if you use the model for your research
    """)