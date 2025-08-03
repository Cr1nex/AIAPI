import streamlit as st
import requests
import time
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

def ask_llm(token, question, session_id: str = "0"):
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"question": question,"session_id": session_id}
    
    
    r = requests.post(f"{API_URL}/prompts/create-prompt", json=payload, headers=headers)
    return r.json() if r.status_code == 200 else {"error": r.text}
def stream_response(text):
    if "chat_response" not in st.session_state:
        st.session_state.chat_response = ""

    st.session_state.chat_response = "" 

    output_placeholder = st.empty()  

    for word in text.split():
        st.session_state.chat_response += word + " "
        output_placeholder.markdown(st.session_state.chat_response)
        time.sleep(0.03)  
#main
def main():
    st.set_page_config(page_title="Chat + Auth")

    #session state
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
        st.session_state["selected_session"] = "0"
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
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
                st.session_state["selected_session"] = "0"
                st.session_state["chat_history"] = []
                st.session_state["chat_sessions"] = []
                st.rerun()
            if st.button("New Chat"):
                new_session=add_session(st.session_state["access_token"])
                if new_session:
                    st.session_state["chat_sessions"] = get_sessions(st.session_state["access_token"])
                    st.session_state["selected_session"] = str(new_session["new_session_id"])
                    st.session_state["chat_history"] = []
                    st.rerun()
            if st.button("Who Am I?"):
                r = whoami(st.session_state["access_token"])
                if r.status_code == 200:
                    st.json(r.json())
                else:
                    st.error("Invalid token")
            
            st.markdown("---")
            st.header("Chat Sessions")
            if not st.session_state["chat_sessions"]:
                st.session_state["chat_sessions"] = get_sessions(st.session_state["access_token"])

            session_options = [str(sess["session_id"]) for sess in st.session_state["chat_sessions"]]
            session_titles = [sess["title"] for sess in st.session_state["chat_sessions"]]

            if session_options:
                
                

                
                selected_idx = st.selectbox(
                    "Choose session",
                    range(len(session_options)),
                    format_func=lambda i: session_titles[i] if i < len(session_titles) else "Unknown"
                )

                
                selected_option = session_options[selected_idx]
                if selected_option != st.session_state.get("selected_session"):
                    st.session_state["selected_session"] = selected_option
                    st.rerun()

            else:
                st.warning("No sessions available. Start a new chat.")

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
            

            
            st.header("Ask the LLM")
            if st.session_state["selected_session"] != "0":
                messages = get_session_messages(st.session_state["access_token"], st.session_state["selected_session"])
                
                for msg in messages:
                    if msg["question"] == "__init__":
                        continue
                    user_q = msg.get("question", "")
                    bot_a = msg.get("answer", "")
                    st.markdown(f"**🥷**: {user_q}")
                    st.markdown(f"**👾**: {bot_a}")
            else:
                st.info("No chat session selected yet. Start a new chat or send a message to create a session.")   

            question = st.text_input("Ask something")
            


            if st.button("Send"):
                try:
                    session_id = st.session_state["selected_session"]
                    

                    data = ask_llm(st.session_state["access_token"], question, session_id)

                    if "error" in data:
                        st.error(f"API Error: {data['error']}")
                    else:
                        answer = data.get("answer")
                        

                        
                        st.markdown(f"**🥷**: {question}")
                        
                        if answer:
                            stream_response(answer)

                                         
                        if data["session_id"] and data["session_id"] != st.session_state["selected_session"]:
                            st.session_state["selected_session"] = str(data["session_id"])
                            st.session_state["chat_sessions"] = get_sessions(st.session_state["access_token"])
                             
                          
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
            
                 

                    

            

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
