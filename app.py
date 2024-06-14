from pathlib import Path
import PIL
import io
import base64

import streamlit as st

import settings
import helper

def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

#page Layout
st.set_page_config(
    page_title="PT. LEN Industri (Persero)",
    layout="wide",
    initial_sidebar_state="expanded"
)

#Menambahkan image, posisi atas tengah serta ukuran gambar
logo_path = "./assets/Logo.png"
logo_base64 = get_base64_image(logo_path)
st.markdown(
    f"""
    <style>
    .centered-image {{
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 200px;
    }}
    .centered-text {{
        text-align: center;
        font-size: 16px;
        font-weight: bold;
    }}
    </style>
    <div>
        <img src="data:image/png;base64,{logo_base64}" class="centered-image" alt="PT. LEN Industri Persero Logo">
        <p class="centered-text">PT. LEN Industri Persero</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Page Heading
st.title("Oil Spill Detection")

# Load Model Segmentation in file Setting.py
model_path=Path(settings.Model_Segmentation)

# Load Model yang telah dilatih sebelumnya
try:
    model=helper.load_model(model_path)
except Exception as ex:
    st.error(f"Tidak dapat memuat Model. Silahkan Chek pathnya kembali!: {model_path}")
    
# Judul pada Header 
st.sidebar.header("Image Segmentation")
source_img = st.sidebar.file_uploader("Pilih Image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

col1, col2 = st.columns(2)

with col1:
    try:
        if source_img is None:
            default_image_path = str(settings.Image_Default)
            default_image = PIL.Image.open(default_image_path)
            st.image(default_image_path, caption="Gambar Asli", use_column_width=True)
        else:
            uploaded_image = PIL.Image.open(source_img)
            st.image(source_img, caption="Unggah Gambar", use_column_width=True)
    except Exception as ex:
        st.error("terjadi Kesalahan saat unggah gambar")
        st.error(ex)
        
with col2:
        if source_img is None:
            default_detect_image_path = str(settings.Image_Detect)
            default_detect_image = PIL.Image.open(default_detect_image_path)
            st.image(default_detect_image_path, caption="Hasil Deteksi", use_column_width=True)
        else: 
            if st.sidebar.button("Deteksi"):
                res = model.predict(uploaded_image)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                
                # Menampilkan gambar hasil deteksi
                st.image(res_plotted, caption="Gambar Deteksi", use_column_width=True)
                
                # Untuk Download Gambar Hasil Deteksi
                detected_img_pil = PIL.Image.fromarray(res_plotted)
                buf = io.BytesIO()
                detected_img_pil.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                #Tombol Download
                st.download_button(
                    label="Download Detected Image",
                    data=byte_im,
                    file_name="detected_image.png",
                    mime="image/png"
                )