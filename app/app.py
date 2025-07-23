import streamlit as st
import requests

API_URL = "http://localhost:8000"

#funcs
def login(username, password):
    response = requests.post(
        f"{API_URL}/auth/token",
        data={"username": username, "password": password},
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    if response.status_code == 200:
        return response.json()["access_token"]
    return None

def register_user(payload):
    response = requests.post(f"{API_URL}/auth/create-user", json=payload)
    return response.status_code == 200

def whoami(token):
    headers = {"Authorization": f"Bearer {token}"}
    return requests.get(f"{API_URL}/auth/me", headers=headers)

def add_session(token):
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(f"{API_URL}/prompts/addsession", headers=headers)
    return r.json() if r.status_code == 200 else []

def get_sessions(token):
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(f"{API_URL}/prompts/chat/sessions", headers=headers)
    return r.json() if r.status_code == 200 else []

def get_session_messages(token, session_id):
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(f"{API_URL}/prompts/chat/session/{session_id}", headers=headers)
    return r.json() if r.status_code == 200 else []

def ask_llm(token, question, session_id=0):
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"question": question, "session_id": session_id}
    if session_id:
        payload["session_id"] = session_id
    else:
        payload["session_id"] = session_id
    r = requests.post(f"{API_URL}/prompts/create-prompt", json=payload, headers=headers)
    return r.json() if r.status_code == 200 else {"error": r.text}

#main
def main():
    st.set_page_config(page_title="Chat + Auth")

    #session State
    if "access_token" not in st.session_state:
        st.session_state["access_token"] = None
    if "authentication_status" not in st.session_state:
        st.session_state["authentication_status"] = False
    if "username" not in st.session_state:
        st.session_state["username"] = ""
    if "view" not in st.session_state:
        st.session_state["view"] = None
    if "chat_sessions" not in st.session_state:
        st.session_state["chat_sessions"] = []
    if "selected_session" not in st.session_state:
        st.session_state["selected_session"] = None

    #sidebar
    with st.sidebar:
        st.title("LLM App")
        st.radio("View", ["Login", "Register"], key="view")

        if st.session_state["authentication_status"]:
            st.success(f" {st.session_state['username']}")
            if st.button("Logout"):
                st.session_state["access_token"] = None
                st.session_state["authentication_status"] = False
                st.session_state["username"] = ""
                st.session_state["selected_session"] = None
                st.rerun()
            if st.button("New Chat"):
                return None
            st.markdown("---")
            st.header("Chat Sessions")
            if st.session_state["chat_sessions"] != get_sessions(st.session_state["access_token"]):
                st.session_state["chat_sessions"] = get_sessions(st.session_state["access_token"])
            if get_sessions(st.session_state["access_token"]) == st.session_state["chat_sessions"]:
                session_options = st.session_state["chat_sessions"]
                st.session_state["selected_session"] = st.selectbox("Choose session", session_options)

    #login
    if st.session_state["view"] == "Login":
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
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Login failed")

        else:
            st.title("LLM Chat")

            
            if st.session_state["selected_session"]:
                messages = get_session_messages(st.session_state["access_token"], st.session_state["selected_session"])
                st.markdown("### Chat History")
                for msg in messages:
                    msg = msg.get("role")
                    if msg == "user":
                        st.markdown(f"**You**: {msg['content']}")
                    if msg is not None:
                        st.markdown(f"**Bot**: {msg.get('content', '')}")
                    else:
                        st.markdown("**Bot**: (no content)")

            st.markdown("---")
            st.header("Ask the LLM")

            question = st.text_area("Enter your question")
            if st.button("Ask"):
                data = ask_llm(st.session_state["access_token"], question, st.session_state["selected_session"])
                if "error" in data:
                    st.error("Prompt failed")
                    st.text(data["error"])
                else:
                    st.success("LLM responded:")
                    st.write(data["answer"])

            if st.button("Who Am I?"):
                r = whoami(st.session_state["access_token"])
                if r.status_code == 200:
                    st.json(r.json())
                else:
                    st.error("Invalid token")

    #register
    elif st.session_state["view"] == "Register":
        st.title("Register")

        with st.form("register_form"):
            username = st.text_input("Username")
            email = st.text_input("Email")
            first_name = st.text_input("First Name")
            last_name = st.text_input("Last Name")
            phone_number = st.text_input("Phone Number")
            password = st.text_input("Password", type="password")

            submitted = st.form_submit_button("Register")
            if submitted:
                user_data = {
                    "username": username,
                    "email": email,
                    "first_name": first_name,
                    "last_name": last_name,
                    "phone_number": phone_number,
                    "password": password
                }
                success = register_user(user_data)
                if success:
                    st.success("Registration successful. Please log in.")
                    
                else:
                    st.error("Registration failed")

if __name__ == "__main__":
    main()
