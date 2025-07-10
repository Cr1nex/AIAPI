import streamlit as st
import requests
from streamlit_jwt_authenticator import Authenticator


def main():
    # Use st.write with HTML for the title
    st.write("<h1>FastAPI Streamlit Demo</h1>", unsafe_allow_html=True)
    authenticator = Authenticator(url="http://localhost:8000/cookie/")
    authenticator.login()


    # Demo GET request
    if st.button("Get Data"):
        response = requests.get("http://localhost:8000/admin/getusers")
        st.write(response.json())

    # Demo POST request
   
    title = st.text_input("Title:")
    question = st.text_input("Ask the LLM:")
    if st.button("ASK"):
        # Endpoint to be implemented in the API
        response = requests.post("http://localhost:8000/prompts/create-prompt", json={"title": title,"question": question})
        st.write(response.json())
    # Check is user logged in successfully
    if st.session_state["authentication_status"]:
        # Add logout form
        authenticator.logout()


if __name__ == "__main__":
    main()