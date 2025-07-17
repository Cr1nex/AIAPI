import streamlit as st
import requests

API_URL = "http://localhost:8000" 

def login(username, password):
    response = requests.post(
        f"{API_URL}/auth/token",
        data={"username": username, "password": password},
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        return None

def main():
    st.set_page_config(page_title="Manual JWT Login")

    #Session state
    if "access_token" not in st.session_state:
        st.session_state["access_token"] = None
    if "authentication_status" not in st.session_state:
        st.session_state["authentication_status"] = False
    if "username" not in st.session_state:
        st.session_state["username"] = ""

    #Not logged in
    if not st.session_state["authentication_status"]:
        st.title("Login")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            token = login(username, password)
            if token:
                st.session_state["access_token"] = token
                st.session_state["username"] = username
                st.session_state["authentication_status"] = True
                st.success("Login successful!!!")
                st.rerun()
            else:
                st.error("Login failed :(")

    #Logged in
    else:
        st.sidebar.success(f"Welcome, {st.session_state['username']}")
        if st.sidebar.button("Logout"):
            st.session_state["access_token"] = None
            st.session_state["authentication_status"] = False
            st.session_state["username"] = ""
            st.rerun()

        st.title("Authenticated Zone")

        #Test
        if st.button("Who Am I?"):
            headers = {"Authorization": f"Bearer {st.session_state['access_token']}"}
            r = requests.get(f"{API_URL}/auth/me", headers=headers)
            if r.status_code == 200:
                st.success("Token is valid")
                st.json(r.json())
            else:
                st.error("Token invalid or expired")
                st.write(r.text)

        #LLM
        st.header("<|Ask the LLM|>")
        title = st.text_input("Title")
        question = st.text_area("Question")
        if st.button("Ask"):
            payload = {"title": title, "question": question}
            headers = {"Authorization": f"Bearer {st.session_state['access_token']}"}
            r = requests.post(f"{API_URL}/prompts/create-prompt", json=payload, headers=headers)
            if r.status_code == 200:
                st.success("Response received!")
                data=r.json()
                st.write(data[0])
            else:
                st.error("Failed to submit prompt")
                st.write(r.text)

if __name__ == "__main__":
    main()
