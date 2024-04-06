from ultralytics import YOLO
from PIL import Image
import cv2
import torch
import streamlit as st

#@st.cache_data()
def load_model(model_path):
    model = torch.hub.load('ultralytics/yolov5','custom',path=model_path)
    return model

def run_detection(model,img):
    results = model(img)
    return results

def main():
    st.title("Detection d'objets avec YOLOv5 et Streamlit")
    model = load_model('best1.pt')

    files = st.file_uploader("Téléchargez une ou plusieurs images",type=["jpg","jpeg","png"],accept_multiple_files=True)
    if files:
        for file in files:
            image = Image.open(file)
            st.image(image,caption=f"Image originale: {file.name}",use_column_width=True)
            if st.button(f"Détection pour {file.name}"):
                results = run_detection(model,image)
                image_with_boxes = results.render()
                st.image(image_with_boxes,caption=f"Image avec détections:{file.name}",use_column_width=True)

                #le nombre de visages détectés
                face_count = len(results.xyxy[0])
                st.write("Nombre d'objets détectés:{face_count}")

if __name__ == "__main__":
    main()