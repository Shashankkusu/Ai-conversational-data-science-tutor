import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Configure Page
st.set_page_config(page_title="AI Data Science Tutor", page_icon="static/img/logo.svg", layout="wide")

# Load API key securely
google_api_key = os.getenv("GOOGLE_API_KEY") or st.secrets["google"]["api_key"]

# Initialize Chat Model
chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7, google_api_key=google_api_key)

# Initialize Memory in Session State
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Function for AI Tutor
def conversational_tutor(user_input):
    """Interacts with the AI and maintains conversation context."""
    
    # Store user input
    st.session_state.memory.save_context({"input": user_input}, {"output": "Processing..."})
    
    conversation_history = st.session_state.memory.load_memory_variables({})

    prompt = f"""
    You are a Data Science tutor. Answer ONLY data science-related questions.
    If the user asks something unrelated, politely decline.
    
    Conversation history: {conversation_history}
    
    User: {user_input}
    """

    # Get response from AI
    response = chat_model.invoke(prompt)
    
    # Extract content properly
    response_text = response if isinstance(response, str) else getattr(response, "content", "I'm sorry, I couldn't generate a response.")

    # Store response
    st.session_state.memory.save_context({"input": user_input}, {"output": response_text})

    return response_text

# Function to Render Sidebar
def render_sidebar():
    with st.sidebar:
        st.title("⚙️ Settings")
        st.write("Customize your experience")
        
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.memory.clear()
            st.rerun()

# Streamlit UI
def main():
    # Apply Custom Background Image
    st.markdown(
    """
    <style>
    .stApp {
        background: url("static/img/background.jpg") no-repeat center center fixed;
        background-size: cover;
    }
    
    /* Customizing input box to look like ChatGPT */
    .chat-input {
        width: 100%;
        padding: 12px;
        border-radius: 12px;
        border: 1px solid #ccc;
        font-size: 16px;
        outline: none;
        background: #222;
        color: white;
    }

    .chat-submit {
        padding: 10px 15px;
        font-size: 16px;
        background: #0A84FF;
        border: none;
        color: white;
        border-radius: 8px;
        cursor: pointer;
        margin-left: 10px;
    }

    .chat-submit:hover {
        background: #006EDC;
    }
    </style>
    """,
    unsafe_allow_html=True
    )

    # Display Logo & Title
    st.image("static/img/logo.svg", width=100)
    st.title("💡 Conversational AI Data Science Tutor")
    st.subheader("Your AI-Powered Guide to Data Science Learning")

    # Render Sidebar
    render_sidebar()

    # Display Chat Messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Custom Chat Input Box like ChatGPT
    with st.container():
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area("Type your message...", height=100, key="user_input")
            submit_button = st.form_submit_button("Send", use_container_width=True)
        
        if submit_button and user_input:
            # Store User Message
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Get AI Response
            response = conversational_tutor(user_input)
            
            # Store AI Response
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Display AI Response
            with st.chat_message("assistant"):
                st.write(response)

if __name__ == "__main__":
    main()
