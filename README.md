# 🤖 Conversational AI Data Science Tutor

This is an AI-powered tutor built using **LangChain**, **Google Gemini 1.5 Pro**, and **Streamlit**.  
It helps users resolve **Data Science-related doubts** with memory for conversation awareness.

## 🚀 Features
- ✅ Conversational Memory – AI remembers previous chats for better context.  
- ✅ Multi-user Support – Each user has a separate chat history.  
- ✅ Secure API Key Handling – Uses Streamlit Secrets Management.  
- ✅ Streamlit UI – Simple, interactive, and responsive chat interface.  
- ✅ Custom Background Image – Enhances visual appeal.  

## 🛠️ Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/yourusername/data-science-tutor.git
   cd data-science-tutor
   ```

2. **Install Dependencies**  
   ```bash
   pip install streamlit langchain google-generativeai langchain-google-genai
   ```

3. **Set Up API Key Securely**  
   ```bash
   mkdir -p ~/.streamlit && nano ~/.streamlit/secrets.toml
   ```
   Add the following content:  
   ```toml
   [google]
   api_key = "your-google-api-key"
   ```

4. **Run the Application**  
   ```bash
   streamlit run app.py
   ```

## 🌐 Deployment (Streamlit Cloud)
1. Push your code to **GitHub**.  
2. Go to [Streamlit Cloud](https://share.streamlit.io/) and deploy your repo.  
3. Set **Secrets in Streamlit Cloud** under **Settings → Secrets**  
   ```toml
   [google]
   api_key = "your-google-api-key"
   ```
4. Click **Deploy** and access your AI tutor online!

## 📝 License  
This project is **open-source** and available under the MIT License.  

## 🙌 Contributing  
Feel free to **fork** this repository, create **issues**, or submit **pull requests**!  

## 📧 Contact  
For questions, reach out via [LinkedIn](https://www.linkedin.com/in/kusu-shanmuka-venkata-satya-sai-shashank-84824a298/) or email **gowrisesharoa@gmail.com**.
