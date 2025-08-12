import streamlit as st
from openai import OpenAI
import pandas as pd
import json
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(page_title="Japanese AI Tutor", page_icon="🗾")

# Initialize session state
if 'learned_words' not in st.session_state:
    st.session_state.learned_words = []
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Sidebar for API key setup
st.sidebar.title("Setup")

# Try to get API key from environment variable first
env_api_key = os.getenv("OPENAI_API_KEY")

if env_api_key:
    api_key = env_api_key
    st.sidebar.success("✅ API Key loaded from environment")
else:
    api_key = st.sidebar.text_input("Enter OpenAI API Key:", type="password")
    st.sidebar.info("💡 Tip: Add OPENAI_API_KEY to your .env file for automatic loading")

if api_key:
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Main app
    st.title("🗾 Japanese AI Learning Companion")
    st.write("Practice Japanese conversation and learn new vocabulary!")
    
    # Tab navigation
    tab1, tab2, tab3 = st.tabs(["💬 Chat Practice", "📊 Progress", "📝 Vocabulary"])
    
    with tab1:
        st.subheader("Conversation Practice")
        
        # Difficulty level
        level = st.selectbox("Your Level:", ["Beginner", "Intermediate", "Advanced"])
        
        # Topic selection
        topic = st.selectbox("Practice Topic:", 
                           ["Daily Greetings", "Food & Restaurants", "Travel", 
                            "Work & School", "Hobbies", "Free Conversation"])
        
        # Chat interface
        user_input = st.text_input("Type in English or Japanese:")
        
        if st.button("Send Message") and user_input:
            # Create system prompt based on level and topic
            system_prompt = f"""You are a helpful Japanese language tutor. 
            Student level: {level}
            Topic: {topic}
            
            Rules:
            1. Respond in both Japanese and English
            2. Introduce 1-2 new words naturally
            3. Correct any mistakes gently
            4. Keep responses encouraging and educational
            5. Format: Japanese text (English translation)
            """
            
            try:
                # Call OpenAI API (new format)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input}
                    ],
                    max_tokens=150,
                    temperature=0.7
                )
                
                ai_response = response.choices[0].message.content
                
                # Store conversation
                st.session_state.conversation_history.append({
                    "user": user_input,
                    "ai": ai_response,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                })
                
                # Display response
                st.write("**AI Tutor:**")
                st.write(ai_response)
                
            except Exception as e:
                st.error(f"Error: {e}")
    
    with tab2:
        st.subheader("Learning Progress")
        
        if st.session_state.conversation_history:
            # Create progress dataframe
            df = pd.DataFrame(st.session_state.conversation_history)
            
            # Show statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Conversations", len(df))
            with col2:
                st.metric("Words Learned", len(st.session_state.learned_words))
            with col3:
                today_chats = len([c for c in st.session_state.conversation_history 
                                if c['timestamp'].startswith(datetime.now().strftime("%Y-%m-%d"))])
                st.metric("Today's Practice", today_chats)
            
            # Show recent conversations
            st.subheader("Recent Conversations")
            for i, conv in enumerate(reversed(st.session_state.conversation_history[-5:])):
                with st.expander(f"Conversation {len(st.session_state.conversation_history)-i}"):
                    st.write(f"**You:** {conv['user']}")
                    st.write(f"**AI:** {conv['ai']}")
                    st.write(f"*{conv['timestamp']}*")
        else:
            st.info("Start chatting to see your progress!")
    
    with tab3:
        st.subheader("Vocabulary Tracker")
        
        # Add new word manually
        col1, col2 = st.columns(2)
        with col1:
            new_word_jp = st.text_input("Japanese Word:")
        with col2:
            new_word_en = st.text_input("English Meaning:")
        
        if st.button("Add Word") and new_word_jp and new_word_en:
            word_entry = {
                "japanese": new_word_jp,
                "english": new_word_en,
                "added_date": datetime.now().strftime("%Y-%m-%d"),
                "practice_count": 0
            }
            st.session_state.learned_words.append(word_entry)
            st.success(f"Added: {new_word_jp} ({new_word_en})")
        
        # Display vocabulary
        if st.session_state.learned_words:
            vocab_df = pd.DataFrame(st.session_state.learned_words)
            st.dataframe(vocab_df)
            
            # Download vocabulary as CSV
            csv = vocab_df.to_csv(index=False)
            st.download_button(
                "Download Vocabulary",
                csv,
                "japanese_vocabulary.csv",
                "text/csv"
            )
        else:
            st.info("Add words as you learn them during conversations!")

else:
    st.warning("Please enter your OpenAI API key in the sidebar to start learning!")
    st.info("""
    **To get started:**
    1. Get an API key from platform.openai.com
    2. Enter it in the sidebar
    3. Start practicing Japanese!
    
    **Features:**
    - Conversational AI tutor
    - Progress tracking
    - Vocabulary management
    - Multiple difficulty levels
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit + OpenAI API | Your Japanese Learning Journey 🚀")