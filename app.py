import streamlit as st
import torch
import segmentation_models_pytorch as smp
import numpy as np
import cv2
from PIL import Image
import os
import requests

# --- PAGE CONFIG ---
st.set_page_config(page_title="Rooftop Area Detector", layout="centered")

# --- MODEL LOADING ---
@st.cache_resource

def load_unet_model():
    model_filename = 'bhopal_model.pth'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, model_filename)
    
    # YOUR_FILE_ID must be the ID from your Drive link
    file_id = 'YOUR_FILE_ID_HERE' 
    
    # Updated Direct Download URL for Google Drive
    download_url = f'https://docs.google.com/uc?export=download&id={file_id}&confirm=t'

    if not os.path.exists(model_path):
        with st.spinner("Downloading model from Drive..."):
            try:
                session = requests.Session()
                response = session.get(download_url, stream=True)
                
                # Check if the download actually returned a file or an error
                if response.status_code == 200:
                    with open(model_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                else:
                    st.error(f"Download failed. Status code: {response.status_code}")
                    st.stop()
            except Exception as e:
                st.error(f"Download Error: {e}")
                st.stop()

    # Continue with loading logic
    torch.serialization.add_safe_globals([
        smp.decoders.unet.model.Unet,
        smp.decoders.unet.model.UnetDecoder,
        smp.encoders.resnet.ResNetEncoder,
        smp.base.modules.Activation,
    ])

    # Reconstruct architecture based on your ResNet-34 training
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    
    try:
        # Loading on CPU for the Streamlit server
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            model = checkpoint
    except Exception as e:
        # If it still says "invalid load key", the file on disk is corrupted HTML
        if os.path.exists(model_path):
            os.remove(model_path) # Delete the bad file so it retries next time
        st.error(f"Model file is corrupted or not a valid .pth file. Error: {e}")
        st.stop()

    model.eval()
    return model



def preprocess_image(pil_image):
    # 1. Convert PIL Image to BGR (to mimic cv2.imread used in your notebook)
    img = np.array(pil_image.convert('RGB'))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # 2. Fix colors to RGB (matches notebook: cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 3. Resize to the exact training tile size (matches TILE_SIZE = 512)
    img_resized = cv2.resize(img_rgb, (512, 512))
    
    # 4. Normalization (matches notebook: image = image / 255.0)
    img_scaled = img_resized.astype(np.float32) / 255.0
    
    # 5. Transpose (matches notebook: image = image.transpose(2, 0, 1))
    img_input = np.transpose(img_scaled, (2, 0, 1))
    
    return torch.tensor(img_input).float().unsqueeze(0), img_resized



# --- UI INTERFACE ---
st.title("🏠 Flat Rooftop Detector")
st.write("Upload a satellite image to detect rooftops and calculate available area.")

uploaded_file = st.file_uploader("Choose a satellite image (JPG/PNG)...", type=["jpg", "png", "jpeg"])

# Add a slider for resolution (Laymen need this to get accurate area)
# 0.3m is common for high-res satellite like Maxar/Google
resolution = st.slider("Select Image Resolution (meters per pixel)", 0.1, 2.0, 0.3)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    if st.button('Analyze Image'):
        model = load_unet_model()  # This calls the function above
        input_tensor, resized_orig = preprocess_image(image)
        with torch.no_grad():
            output = model(input_tensor)
            # Apply sigmoid as your model was trained with binary classes
            probs = torch.sigmoid(output).squeeze().cpu().numpy()
            binary_mask = (probs > 0.5).astype(np.uint8)
        
            # Calculate Area
            # Total pixels * (resolution squared)
            roof_pixel_count = np.count_nonzero(binary_mask)
            # standard high-res satellite is ~0.3m per pixel
            total_area_m2 = roof_pixel_count * (resolution ** 2)

            st.metric("Detected Rooftop Area", f"{total_area_m2:.2f} m²")
        
        with st.spinner('AI is processing...'):
            # Load and Predict
            model = load_unet_model()
            input_tensor, resized_orig = preprocess_image(image)
            
            with torch.no_grad():
                output = model(input_tensor)
                # Apply sigmoid to get probability and threshold at 0.5
                mask = torch.sigmoid(output).squeeze().cpu().numpy()
                binary_mask = (mask > 0.5).astype(np.uint8)

            # --- CALCULATE AREA ---
            # Area = total positive pixels * (resolution^2)
            pixel_count = np.sum(binary_mask)
            total_area = pixel_count * (resolution ** 2)

            # Calculate Area
            # Total pixels * (resolution squared)
            roof_pixel_count = np.count_nonzero(binary_mask)
            # standard high-res satellite is ~0.3m per pixel
            total_area_m2 = roof_pixel_count * (resolution ** 2)

            st.metric("Detected Rooftop Area", f"{total_area_m2:.2f} m²")

            # --- DISPLAY RESULTS ---
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Detection Mask")
                # Scale mask to 255 for visibility
                st.image(binary_mask * 255, use_container_width=True)
                
            with col2:
                st.subheader("Overlay")
                # Create a colored overlay
                overlay = resized_orig.copy()
                overlay[binary_mask == 1] = [0, 255, 0] # Highlight green
                blended = cv2.addWeighted(resized_orig, 0.7, overlay, 0.3, 0)
                st.image(blended, use_container_width=True)

            st.success(f"Analysis Complete!")
            st.metric(label="Estimated Rooftop Area", value=f"{total_area:.2f} m²")
            st.info(f"Detected {pixel_count} rooftop pixels.")