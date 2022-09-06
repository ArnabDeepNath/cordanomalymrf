from turtle import width
import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from PIL import Image


st.title("Cord Spacing Anomaly Prediction")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.image('logo2.png')
st.sidebar.title('In service of MRF Tyres')
st.sidebar.subheader('Developed at Caliche Labs')
st.sidebar.subheader('Parameters')


# Function to resize image

@st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


    # st.markdown('The images can be **loaded** from here for resizing for effective detection ')
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
    # st.header('Load Image')
img_file_buffer = st.sidebar.file_uploader("Upload an image , (*TIFF format currently not available)", type=[ "jpg", "jpeg",'png' ])
st.sidebar.image('logo3.png',width=70)
cropped_image = None
ROI_number=0
    # st.subheader('Pre-Process Image')
if st.button("Check for defects"):
    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))
        # cropped_image = image_resize(image , 500 , 200 )
        # cropped_image = image[:-500, :8000]
        cv2.imwrite('cropped_img.png', image)
        if image is not None:
            pred_img = cv2.imread('cropped_img.png')
            gray_pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2GRAY)
            thresh_train = cv2.threshold(gray_pred_img, 0 , 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            cnts = cv2.findContours(thresh_train, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            original = pred_img.copy()
            ROI_number = 0
            for c in cnts:
                x,y,w,h = cv2.boundingRect(c)
#     cv2.rectangle(thresh_train, (x, y), (x + w, y + h), (0,0,255), 2)
                ROI = original[y:y+h, x:x+w]
#     cv2.imwrite('Image_{}.png'.format(ROI_number), ROI)
                ROI_number += 1
            min_area = 50
            max_area = 100
            min_w = 10
            min_h = 15
            for c in cnts:
                area = cv2.contourArea(c)
                cv2.drawContours(thresh_train, [c], -1, (0, 255, 0), 1)
#     if w > 10 and h > 10:
#         min_area = 50
            max_area = 100
            min_w = 10
            min_h = 15
            for c in cnts:
                area = cv2.contourArea(c)
                cv2.drawContours(thresh_train, [c], -1, (0, 255, 0), 1)
#     if w > 150 and h > 100:
#         cv2.drawContours(thresh_train, [c], -1, (0, 0, 255), 1)
        #area_thresh = 0
            # min_area = 20*20
            # max_area = 50*800
            contours = cv2.findContours(thresh_train, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]
            min_area = 50
            max_area = 100
            w = 5
            h = 5
            for c in contours:
                area = cv2.contourArea(c)
#     cv2.drawContours(thresh_train, [c], -1, (0, 255, 0), 1)
                if w > 1500 and w < 2000 and h > 500:
                    cv2.drawContours(thresh_train, [c], -1, (0, 0, 255), 1)
#             cv2.rectangle(img_train, (x,y),(x+w,y+h),(36,255,12) , 1)
            thresh_train = cv2.threshold(thresh_train, 228, 255, cv2.THRESH_BINARY)[1]
            for c in contours:
                x , y , w , h = cv2.boundingRect(c)
#     cv2.drawContours(img_train, [c], -1, (0, 255, 0), 1)
                if w > 1500 and w < 2000 and h > 500:
#             cv2.drawContours(img_train, [c], -1, (0, 0, 255), 1)
                    cv2.rectangle(pred_img, (x,y),(x+w,y+h),(36,255,12) , 1)
            contours = cv2.findContours(thresh_train, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]
#area_thresh = 0
            min_area = 20
            max_area = 50
            result = thresh_train.copy()
            for c in contours:
                x , y , w , h = cv2.boundingRect(c)
#     cv2.rectangle(img_train, (x,y),(x+w,y+h),(36,255,12) , 1)
                if w > 450 and h > 35:
#             cv2.drawContours(img_train, [c], -1, (0, 0, 255), 1)
                    cv2.rectangle(pred_img, (x,y),(x+w+20,y+h+20),(0,0,255) , 5)
            if(thresh_train.any()!=0):
                if(thresh_train.any()>0.0):
                        # if(ROI_number < 1060 or ROI_number > 1500):
                    result_class= 'Anomalous tyre'
                        # elif(ROI_number > 1060 or ROI_number < 1000):
                else:
                    result_class='Non Anomalous Tyre'
            elif( thresh_train.all()==0):
                result_class = 'Non Anomalous Tyre' 
            st.text(thresh_train.any()!=0)
    else:
            demo_image = 'demo.jpg'
            image = np.array(Image.open(demo_image))

    st.text("Result: ")
    st.header(result_class)
    st.text('Predicted Image')
    st.image(pred_img)
    st.text('Processed Image')
    st.image(thresh_train)
        # st.image(img_file_buffer)

else:
    demo_image = 'demo.jpg'
    image = np.array(Image.open(demo_image))

    # st.text('PreProcessed Image')
    # st.image(thresh_train)
        
        
if cropped_image is not None:
        # st.markdown('''
        #             Image is Pre-Processed Go to Inferences Section to Make Prediction
        #     ''')
        st.markdown('The images can be **Predicted** from here')
        st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 400px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 400px;
            margin-left: -400px;
        }
        </style>
        """,
        unsafe_allow_html=True,
            )
            
            
        
elif cropped_image is None:
    st.markdown(''' 
                Please Click To Check Tyre Defects! ''')
        
        
    
