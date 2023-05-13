import streamlit as st
from deepface import DeepFace as Df
import cv2
import time as tm
import pandas as pd
import datetime as dt
import numpy as np

r_time = None
tab1, tab2, tab3 = st.tabs(["Face Register", "Face Verify", "Time Log"])
with tab1:
    name = st.text_input("Input Name")
    picture1 = st.camera_input("Take a picture", key = "abc")
    if picture1 is not None:
        if name != "":
            bytes_data1 = picture1.getvalue()
            cv2_img1 = cv2.imdecode(np.frombuffer(bytes_data1, np.uint8), cv2.IMREAD_COLOR)
            st.success("Register Done")
        else:
            st.error("Please Input Name")

with tab2:
    picture2 = st.camera_input("Take a picture", key = "abcd")
    if picture2 != None:
        t_start = tm.process_time()
        bytes_data2 = picture2.getvalue()
        cv2_img2 = cv2.imdecode(np.frombuffer(bytes_data2, np.uint8), cv2.IMREAD_COLOR)
        r_time = dt.datetime.now()
        verifying = Df.verify(cv2_img1, cv2_img2)
        num1 = verifying.get("distance")
        num2 = round((100 - num1), 0)
        st.success("{}: {}%".format(name,num2))
        t_stop = tm.process_time()
        time = t_stop - t_start
        st.text("Process time: {}s".format(time))

with tab3:
    if r_time != None:
        tlog = {"Name": [name], "Time": [r_time]}
        TLog = pd.DataFrame(tlog)
        st.table(TLog)
