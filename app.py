import streamlit as st
import os
import tempfile
from PIL import Image
from brain_tumor_classifier import predict_image

# Set page configuration
st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="centered"
)

# Page title and description
st.title("Brain Tumor Classification")
st.markdown("""
This application uses a deep learning model to classify brain MRI images into four categories:
- Glioma
- Meningioma
- Pituitary
- No Tumor (Normal)
""")

# File uploader
uploaded_file = st.file_uploader("Upload a brain MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_container_width=True)
    
    # Add a button to trigger prediction
    if st.button("Predict"):
        with st.spinner("Analyzing the image..."):
            try:
                # Create a temporary file with the appropriate extension
                file_extension = os.path.splitext(uploaded_file.name)[1].lower()  # Get the uploaded file's extension
                if file_extension not in [".jpg", ".jpeg", ".png"]:
                    raise ValueError("Unsupported file format. Please upload a JPG, JPEG, or PNG file.")
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                    image.save(temp_file.name)
                    temp_file_path = temp_file.name
                
                # Get prediction using the imported function
                class_name, probability = predict_image(temp_file_path)
                
                # Display results
                st.success(f"Prediction complete!")
                st.subheader("Results:")
                
                # Format the output
                st.info(f"**Diagnosis**: {class_name.title()}")
                st.progress(float(probability) / 100)
                st.markdown(f"**Confidence**: {probability:.2f}%")
                
                # Add additional information based on the class
                if class_name == "no tumor":
                    st.balloons()
                    st.success("No tumor detected in the MRI scan.")
                else:
                    st.warning(f"The model has detected a potential {class_name} tumor.")
                    st.markdown("**Note**: This is an AI-assisted prediction and should not replace professional medical advice.")
            
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
                st.markdown("Please try again with a different image or check if the model is loaded correctly.")
            
            finally:
                # Remove the temporary file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

# Information section
st.markdown("---")
st.subheader("About")
st.markdown("""
This application uses an EfficientNet-B0 model fine-tuned on brain MRI images.
The model classifies images into four categories with high accuracy.

**Note**: This tool is for educational purposes only and should not be used for medical diagnosis.
""")