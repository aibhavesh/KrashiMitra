import streamlit as st

st.set_page_config(layout="wide")

page = st.session_state.get("page", "home")

def go(p):
    st.session_state.page = p
    st.experimental_rerun()

if page == "home":
    st.components.v1.html(open("frontend/pages/1_home.html").read(), height=900)
    if st.button("Start"):
        go("questions")

elif page == "questions":
    st.components.v1.html(open("frontend/pages/2_questions.html").read(), height=1000)

elif page == "upload":
    st.components.v1.html(open("frontend/pages/3_upload.html").read(), height=900)

elif page == "result":
    st.components.v1.html(open("frontend/pages/4_result.html").read(), height=1200)
