
import streamlit as st
import os
import torch

from PIL import Image

from prompt_engine import refine_instruction
from image_editor import edit_image
from clip_evaluator import calculate_clip_score
from safety import check_instruction
from watermark import add_watermark


st.set_page_config(
    page_title="P41 AI Photo Editor",
    page_icon="📸",
    layout="wide"
)

os.makedirs("outputs", exist_ok=True)

st.title("P41 - Instruction-Based Photo Editor")

st.write(
    "Upload a photo and describe how you want the AI to edit it."
)

st.info(
    "Example: Make it night, add snow, create a sunset, "
    "or make the sky cloudy."
)

uploaded_file = st.file_uploader(
    "Upload your image",
    type=["jpg", "jpeg", "png"]
)

instruction = st.text_area(
    "Enter your editing instruction",
    placeholder="Example: Make it night with moonlight"
)

st.subheader("Editing Settings")

col1, col2, col3 = st.columns(3)

with col1:
    image_guidance = st.slider(
        "Image Guidance",
        min_value=1.0,
        max_value=3.0,
        value=1.5,
        step=0.1
    )

with col2:
    guidance = st.slider(
        "Text Guidance",
        min_value=1.0,
        max_value=15.0,
        value=7.5,
        step=0.5
    )

with col3:
    steps = st.slider(
        "Editing Steps",
        min_value=10,
        max_value=50,
        value=20
    )


if uploaded_file:
    original_image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original Image")
    st.image(original_image, use_container_width=True)


if st.button("Generate Edited Image", type="primary"):

    if not uploaded_file:
        st.error("Please upload an image.")

    elif not instruction.strip():
        st.error("Please enter an editing instruction.")

    else:
        allowed, message = check_instruction(instruction)

        if not allowed:
            st.error(message)

        else:
            try:
                with st.spinner("Understanding your instruction..."):

                    refined_prompt = refine_instruction(instruction)

                st.subheader("AI Refined Instruction")
                st.write(refined_prompt)

                with st.spinner("Editing image using InstructPix2Pix..."):

                    original_image = Image.open(
                        uploaded_file
                    ).convert("RGB")

                    edited_image = edit_image(
                        original_image,
                        refined_prompt,
                        image_guidance_scale=image_guidance,
                        guidance_scale=guidance,
                        steps=steps
                    )

                with st.spinner("Evaluating image using CLIP..."):

                    score = calculate_clip_score(
                        edited_image,
                        refined_prompt
                    )

                watermarked_image = add_watermark(edited_image)

                output_path = os.path.join(
                    "outputs",
                    "edited_image.png"
                )

                watermarked_image.save(output_path)

                st.success("Image edited successfully!")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Original Image")
                    st.image(
                        original_image,
                        use_container_width=True
                    )

                with col2:
                    st.subheader("Edited Image")
                    st.image(
                        watermarked_image,
                        use_container_width=True
                    )

                st.metric(
                    "CLIP Similarity Score",
                    str(score) + " / 100"
                )

                st.download_button(
                    label="Download Edited Image",
                    data=open(output_path, "rb").read(),
                    file_name="P41_edited_image.png",
                    mime="image/png"
                )

                st.caption(
                    "Watermark: AI EDITED - P41"
                )

            except Exception as e:
                st.error("Something went wrong: " + str(e))
