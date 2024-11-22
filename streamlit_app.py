#분류 결과 + 이미지 + 영상 + 텍스트 보여주기
#파일 이름 streamlit_app.py

import streamlit as st
from fastai.vision.all import *
from PIL import Image
import gdown

# Google Drive 파일 ID
file_id = '1--0NCtjG2xuIKA0lRe06yyOICUgWNF-L'

# Google Drive에서 파일 다운로드 함수
@st.cache(allow_output_mutation=True)
def load_model_from_drive(file_id):
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'model.pkl'
    gdown.download(url, output, quiet=False)

    # Fastai 모델 로드
    learner = load_learner(output)
    return learner

def display_left_content(image, prediction, probs, labels):
    st.write("### 왼쪽: 기존 출력 결과")
    if image is not None:
        st.image(image, caption="업로드된 이미지", use_column_width=True)
    st.write(f"예측된 클래스: {prediction}")
    st.markdown("<h4>클래스별 확률:</h4>", unsafe_allow_html=True)
    for label, prob in zip(labels, probs):
        st.markdown(f"""
            <div style="background-color: #f0f0f0; border-radius: 5px; padding: 5px; margin: 5px 0;">
                <strong style="color: #333;">{label}:</strong>
                <div style="background-color: #d3d3d3; border-radius: 5px; width: 100%; padding: 2px;">
                    <div style="background-color: #4CAF50; width: {prob*100}%; padding: 5px 0; border-radius: 5px; text-align: center; color: white;">
                        {prob:.4f}
                    </div>
                </div>
        """, unsafe_allow_html=True)

def display_right_content(labels):
    st.write("### 오른쪽: 동적 분류 결과")
    cols = st.columns(3)

    # 1st Row - Images based on labels
    for i, label in enumerate(labels[:3]):
        with cols[i]:
            st.image(f"https://i.ibb.co/yfgQyTw/i1889310591.jpg
https://i.ibb.co/TcTG84m/ayi-IBk5l78oga-LRNzp5oz6a1-Bv-QQ4-Nm4-KGQziia-WY0ej-RCnk-JUUTf-q-N4-ZI-Sz0k-LMCae-Rhwem-Lc-GOk-LByr.webp
https://i.ibb.co/HV4Vbjm/Xb-h8-Xr-Mh-Dx6-Pxoh-HOt2-SZPAg-UIml-WJb5keee-YKGEe5q-GQi5-r-X9m-ZRWib-MAADb-ABqnz-EWSvos96m-Hcv-Ufn.webp
https://i.ibb.co/Tq2RCyW/So-MFwg-Gnb-SMmxzo-Anuha-Koal-USZZd40v-Hyep-Kq-Hx-Hv-Ac-IP-Zkl-FHXC4-KUKf-RNOPdf-OWXe-Dx-VSa-ABFd-Qc.webp
https://i.ibb.co/MnTzsVB/maxresdefault.jpg
https://i.ibb.co/ssJvJnn/298975135-1-1730980461-w360.webp
https://i.ibb.co/wLKCLGG/001.jpg
https://i.ibb.co/VSCNTkW/529280-493959-4426.jpg
https://i.ibb.co/0fH3zzH/20211103162023-7264w.jpg
https://i.ibb.co/hgfFvpP/image.jpg?text={label}", caption=f"이미지: {label}", use_column_width=True)

    # 2nd Row - YouTube Videos based on labels
    for i, label in enumerate(labels[:3]):
        with cols[i]:
            st.video("https://youtu.be/PHpqDW9AfEk?si=8YG8pyZoa1wk0mfj", start_time=0)
            st.caption(f"유튜브: {label}")

    # 3rd Row - Text based on labels
    for i, label in enumerate(labels[:3]):
        with cols[i]:
            st.write(f"아케인재밋떠여.")

# 모델 로드
st.write("모델을 로드 중입니다. 잠시만 기다려주세요...")
learner = load_model_from_drive(file_id)
st.success("모델이 성공적으로 로드되었습니다!")

labels = learner.dls.vocab

# 레이아웃 설정
left_column, right_column = st.columns(2)

# 파일 업로드 컴포넌트 (jpg, png, jpeg, webp, tiff 지원)
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg", "webp", "tiff"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = PILImage.create(uploaded_file)
    prediction, _, probs = learner.predict(img)

    with left_column:
        display_left_content(image, prediction, probs, labels)

    with right_column:
        display_right_content(labels)
