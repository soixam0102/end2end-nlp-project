# Core pkgs
import altair
import streamlit as st

# EDA pkgs
import pandas as pd
import numpy as np

# utils
import joblib

pipe_lr = joblib.load(open("models/emotion_classifier_pipe_lr_06_Aug_2021.pkl", "rb"))


# fxn
def predict_emotion(docx):
    return pipe_lr.predict([docx])


def get_prediction_proba(docx):
    result = pipe_lr.predict_proba([docx])
    return result


def main():
    st.title("Emotion Classifier App")
    menu = ["Home", "Monitor", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home-Emotion in text")

        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Type here")
            submit_text = st.form_submit_button(label='Submit')
        if submit_text:
            col1, col2 = st.beta_columns(2)

            # apply fxn here
            prediction = predict_emotion(raw_text)
            probability = get_prediction_proba(raw_text)
            with col1:
                st.success("Original text")
                st.success("Prediction")
                st.write(prediction)
                st.write("Confidence:{}".format(np.max(probability)))
            with col2:
                st.success("Prediction probability")
                # st.write(probability)
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                # st.write(proba_df.T)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotion", "probability"]
                fig = altair.Chart(proba_df_clean).mark_bar().encode(x="emotion", y="probability")
                st.altair_chart(fig, use_container_width=True)

    elif choice == "Monitor":
        st.subheader("Monitor App")
    else:
        st.subheader("About")


if __name__ == '__main__':
    main()
