import streamlit as st
from openai import OpenAI
import pandas as pd
import json
import os
from datetime import datetime
from dotenv import load_dotenv
#import custom ML utilities
from ml_utils import (calculate_success_rate, get_current_streak, 
                     get_words_due_for_review, update_word_after_review, get_review_stats)
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
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat Practice", "📊 Progress", "📝 Vocabulary", "🧠 Smart Review"])
    
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
                "practice_count": 0,
                "review_sequence": [],
                "last_review_date": None,
                "next_review_date": datetime.now().strftime("%Y-%m-%d")
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
    with tab4:
        st.subheader("🧠 Smart Review - ML Powered")
        
        # Show overall learning stats
        stats = get_review_stats(st.session_state.learned_words)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Words", stats['total_words'])
        with col2:
            st.metric("Words Reviewed", stats['words_with_reviews'])
        with col3:
            st.metric("Avg Success Rate", f"{stats['average_success_rate']:.1%}")
        with col4:
            st.metric("Total Reviews", stats['total_reviews'])
        
        # Get words due for review
        due_words = get_words_due_for_review(st.session_state.learned_words)
        
        if due_words:
            st.subheader(f"📚 {len(due_words)} words due for review today")
            
            # Initialize quiz session state
            if 'current_quiz_index' not in st.session_state:
                st.session_state.current_quiz_index = 0
            if 'quiz_feedback' not in st.session_state:
                st.session_state.quiz_feedback = ""
            if 'show_answer' not in st.session_state:
                st.session_state.show_answer = False
            
            # Check if quiz is finished
            if st.session_state.current_quiz_index < len(due_words):
                # Current word
                current_word = due_words[st.session_state.current_quiz_index]
                
                # Progress indicator
                progress = (st.session_state.current_quiz_index + 1) / len(due_words)
                st.progress(progress)
                st.write(f"Word {st.session_state.current_quiz_index + 1} of {len(due_words)}")
                
                # Quiz interface
                st.markdown("### What does this mean in English?")
                st.markdown(f"# {current_word['japanese']}")
                
                # Show word stats if it has review history
                if current_word.get('review_sequence'):
                    success_rate = calculate_success_rate(current_word['review_sequence'])
                    streak = get_current_streak(current_word['review_sequence'])
                    st.caption(f"Your stats: {success_rate:.1%} success rate, {streak} current streak")
                
                # User input
                user_answer = st.text_input("Your answer:", key=f"answer_{st.session_state.current_quiz_index}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Check Answer", type="primary"):
                        if user_answer.strip():
                            # Check if answer is correct (case-insensitive)
                            correct_answer = current_word['english'].lower().strip()
                            user_answer_clean = user_answer.lower().strip()
                            
                            is_correct = user_answer_clean == correct_answer
                            
                            # Update the word with ML functions
                            updated_word, success_rate, current_streak = update_word_after_review(
                                current_word, is_correct
                            )
                            
                            # Update the word in session state
                            for i, word in enumerate(st.session_state.learned_words):
                                if word['japanese'] == current_word['japanese']:
                                    st.session_state.learned_words[i] = updated_word
                                    break
                            
                            # Show feedback
                            if is_correct:
                                st.session_state.quiz_feedback = f"✅ Correct! '{current_word['japanese']}' means '{current_word['english']}'"
                                st.success(st.session_state.quiz_feedback)
                                st.info(f"Success rate: {success_rate:.1%} | Current streak: {current_streak}")
                            else:
                                st.session_state.quiz_feedback = f"❌ Not quite. '{current_word['japanese']}' means '{current_word['english']}'"
                                st.error(st.session_state.quiz_feedback)
                                st.info(f"Success rate: {success_rate:.1%} | Current streak: {current_streak}")
                            
                            st.session_state.show_answer = True
                        else:
                            st.warning("Please enter an answer!")
                
                with col2:
                    if st.session_state.show_answer:
                        if st.button("Next Word"):
                            st.session_state.current_quiz_index += 1
                            st.session_state.quiz_feedback = ""
                            st.session_state.show_answer = False
                            st.rerun()
                
                # Show feedback if available
                if st.session_state.quiz_feedback and not st.session_state.show_answer:
                    if "✅" in st.session_state.quiz_feedback:
                        st.success(st.session_state.quiz_feedback)
                    else:
                        st.error(st.session_state.quiz_feedback)
                
            else:
                # Quiz completed
                st.success("🎉 Review session completed!")
                
                # Show session summary
                completed_words = due_words[:st.session_state.current_quiz_index]
                if completed_words:
                    correct_count = sum(1 for word in completed_words 
                                      if word.get('review_sequence') and word['review_sequence'][-1])
                    total_count = len(completed_words)
                    session_success_rate = correct_count / total_count if total_count > 0 else 0
                    
                    st.metric("Session Success Rate", f"{session_success_rate:.1%}")
                    st.write(f"You reviewed {total_count} words and got {correct_count} correct!")
                
                # Reset button
                if st.button("Start New Review Session"):
                    st.session_state.current_quiz_index = 0
                    st.session_state.quiz_feedback = ""
                    st.session_state.show_answer = False
                    st.rerun()
                    
                # Show next review schedule
                st.subheader("📅 Next Review Schedule")
                upcoming_reviews = []
                for word in st.session_state.learned_words:
                    if word.get('next_review_date'):
                        upcoming_reviews.append({
                            'word': word['japanese'],
                            'meaning': word['english'],
                            'next_review': word['next_review_date']
                        })
                
                if upcoming_reviews:
                    review_df = pd.DataFrame(upcoming_reviews)
                    review_df = review_df.sort_values('next_review')
                    st.dataframe(review_df, use_container_width=True)
        
        else:
            st.success("🎉 No words due for review today! Great job staying on top of your studies.")
            st.info("Add more vocabulary or come back tomorrow for scheduled reviews.")
            
            # Show upcoming reviews even when none are due today
            if st.session_state.learned_words:
                st.subheader("📅 Upcoming Reviews")
                upcoming_reviews = []
                for word in st.session_state.learned_words:
                    if word.get('next_review_date'):
                        upcoming_reviews.append({
                            'word': word['japanese'],
                            'meaning': word['english'], 
                            'next_review': word['next_review_date']
                        })
                
                if upcoming_reviews:
                    review_df = pd.DataFrame(upcoming_reviews)
                    review_df = review_df.sort_values('next_review')
                    st.dataframe(review_df.head(10), use_container_width=True)
            
        # Quick vocabulary quiz
        if len(st.session_state.learned_words) > 0:
            st.subheader("Quick Quiz")
            if st.button("Random Word Quiz"):
                import random
                word = random.choice(st.session_state.learned_words)
                st.write(f"**What does this mean?** {word['japanese']}")
                with st.expander("Show Answer"):
                    st.write(f"**Answer:** {word['english']}")

else:
    st.warning("Please enter your OpenAI API key in the sidebar to start learning!")
    st.info("""
**To get started:**
1. Add credits at platform.openai.com/billing
2. Get your API key from platform.openai.com/api-keys
3. Enter it in the sidebar or add to .env file
4. Start practicing Japanese!

**Features:**
- Conversational AI tutor
- Progress tracking with charts
- Vocabulary management
- Multiple difficulty levels
- Quick vocabulary quiz
""")

# Footer
st.markdown("---")
st.markdown("© 2025 Markus Means | [https://github.com/MeansMarkus/AI-ML-Japanese-Tutor]") 
st.markdown("Built with Streamlit + OpenAI API | Your Japanese Learning Journey 🚀")
