import streamlit as st
from PIL import Image

st.title("Foundational Healthcare AI Models")
st.subheader("👋 Welcome to the Healthcare AI Models Demo App")

st.markdown('---')
st.subheader("⬅️ Select a Model from the Sidebar to Start")
st.markdown(
    '''
    **Important Note:**  
    _These models are not ready for clinical use "as is", but they represent the highest level of performance currently achievable on public benchmarks, outperforming Google’s Med-Gemini, and are pretrained open foundation models that health and life sciences organizations can customize, adapt, and deploy for their specific use cases programatically or through Azure AI Studio._
    '''
)

st.markdown(
    '''
    --- 
    ### ⚕️ Available Models:  
      
    #### **MedImageParse**:    
    A biomedical foundation model for imaging parsing that can jointly conduct segmentation, detection, and recognition across 9 imaging modalities. Through joint learning, we can improve accuracy for individual tasks and enable novel applications such as segmenting all relevant objects in an image through a text prompt, rather than requiring users to laboriously specify the bounding box for each object.   
    [Read More Here](https://arxiv.org/abs/2405.12971)

    #### **CxRReportGen**: (_Coming Soon_)    
    This model is built to help an application interpret complex medical imaging studies of chest X-rays. When built upon and integrated into an application, CXRReportGen may help developers generate comprehensive and structured radiology reports, with visual grounding represented by bounding boxes on the images.   
    [Read More Here](https://arxiv.org/abs/2406.04449)

    #### **PRISM**: (_Coming Soon_)   
    A multi-modal generative foundation model for slide-level analysis of H&E-stained histopathology images. Utilizing Virchow tile embeddings and clinical report texts for pre-training, PRISM combines these embeddings into a single slide embedding and generates a text-based diagnostic report. These can be used for tasks such as cancer detection, sub-typing, and biomarker identification.   
    [Read More Here](https://paige.ai/paige-introduces-prism-a-slide-level-foundation-model-to-empower-the-next-era-of-pathology-cancer-treatment/)

    #### **GigaPath**: (_Coming Soon_)    
    '''
)

st.image(Image.open("documentation_images/image-1.png"), use_column_width=True)

st.markdown(
    '''
    ---
    #### 🔧 Setup

    To get started deploy the model(s) you wish to demonstrate using the Azure Model Catalog  

    '''
)


st.image(Image.open("documentation_images/image.png"), use_column_width=True)

st.markdown(
    '''
    Then, add a [.env](https://pypi.org/project/python-dotenv/) file to the home directory of this repository with the required details.
    ```txt
    MIP_ENDPOINT=<MED IMAGE PARSE ENDPOINT>
    MIP_API_KEY=<MED IMAGE PARSE API KEY>
    MIP_DEPLOY_NAME=<MED IMAGE PARSE DEPLOYMENT NAME>
    .
    .
    .
    (Repeat the same structure for any other model deployments you wish to demo)
    ```
    '''
)