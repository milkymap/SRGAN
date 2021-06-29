import os 
import streamlit as st 
 
from libraries.strategies import * 
from glob import glob 

from torchvision import transforms as T 

to_pil = T.ToPILImage()
index = 0

with st.sidebar:
	st.header("uploaded image")
	with st.form(key="upload_image"):
	    uploaded_image = None
	    input_buffer = st.file_uploader(
	        label="charger votre image", type=("png", "jpg", "jpeg")
	    )
	    if input_buffer is not None:
	        raw_data = input_buffer.read()
	        uploaded_image = cv2.imdecode(
	            np.frombuffer(raw_data, np.uint8), cv2.IMREAD_COLOR
	        )
	        uploaded_image = cv2.resize(uploaded_image, (512, 512))
	        uploaded_image = Image.fromarray(
	            cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2RGB)
	        )

	    st.form_submit_button(label="upload image")

with st.beta_container():
	st.header('generative adversarial network')
	index = st.slider('frame number', 0, nb_images)
	st.image(to_pil(cv2th(images[idx])))