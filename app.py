import streamlit as st
from openai import OpenAI
import pandas as pd
import json
import os
from datetime import datetime
from dotenv import load_dotenv
import atexit

# Import configuration and utilities
from config import config_manager, get_config
from ml_utils import (calculate_success_rate, get_current_streak, 
                     get_words_due_for_review, update_word_after_review, get_review_stats)

# Load environment variables
load_dotenv()

# Get configuration
config = get_config()

# Page config
st.set_page_config(
    page_title=config.PAGE_TITLE, 
    page_icon=config.PAGE_ICON,
    layout="wide" if config.DEBUG_MODE else "centered"
)

# Initialize persistence based on configuration
initialize_func = config_manager.get_initialization_function()
save_func = config_manager.get_save_function()
persistence_manager = config_manager.get_persistence_manager()

# Initialize session state with persistent data
initialize_func()

# Debug info (only if debug mode is enabled)
if config.DEBUG_MODE:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🐛 Debug Info")
    st.sidebar.write(f"Persistence: {config.PERSISTENCE_METHOD}")
    st.sidebar.write(f"Data Dir: {config.DATA_DIRECTORY}")
    st.sidebar.write(f"Auto-save: {config.AUTO_SAVE_ENABLED}")
    if hasattr(persistence_manager, 'db_path'):
        st.sidebar.write(f"DB Path: {persistence_manager.db_path}")

# Sidebar for API key setup and data management
st.sidebar.title("⚙️ Setup")

# Try to get API key from environment variable first
env_api_key = os.getenv("OPENAI_API_KEY")

if env_api_key:
    api_key = env_api_key
    st.sidebar.success("✅ API Key loaded from environment")
else:
    api_key = st.sidebar.text_input("Enter OpenAI API Key:", type="password")
    st.sidebar.info("💡 Tip: Add OPENAI_API_KEY to your .env file")

# Data Management Section
st.sidebar.markdown("---")
st.sidebar.subheader("📁 Data Management")

# Show data statistics
try:
    if config.PERSISTENCE_METHOD == "sqlite":
        stats = persistence_manager.get_statistics()
        st.sidebar.metric("Words Stored", stats.get('total_words', 0))
        st.sidebar.metric("Conversations", stats.get('total_conversations', 0))
        st.sidebar.metric("DB Size", f"{stats.get('db_size_mb', 0)} MB")
    else:
        stats = persistence_manager.get_data_stats()
        st.sidebar.metric("Words Stored", stats.get('learned_words_count', 0))
        st.sidebar.metric("Conversations", stats.get('conversation_count', 0))
        st.sidebar.metric("Data Size", f"{stats.get('data_size_mb', 0)} MB")
except Exception as e:
    if config.DEBUG_MODE:
        st.sidebar.error(f"Stats error: {e}")

# Manual save button
if not config.AUTO_SAVE_ENABLED or config.DEBUG_MODE:
    if st.sidebar.button("💾 Save Data"):
        try:
            save_func()
            st.sidebar.success("Data saved!")
        except Exception as e:
            st.sidebar.error(f"Save error: {e}")
            

## Data Actions


# This is the new, consolidated block.
with st.sidebar.expander("📊 Data Actions"):
    # Export Data
    if st.button("⬆️ Export All Data"):
        try:
            if config.PERSISTENCE_METHOD == "sqlite":
                export_path = persistence_manager.export_to_json()
            else:
                export_path = persistence_manager.export_all_data()

            if export_path:
                st.success("Data exported successfully! See download button below.")
                with open(export_path, 'rb') as f:
                    st.download_button(
                        "⬇️ Download Backup",
                        f.read(),
                        file_name=os.path.basename(export_path),
                        mime="application/json"
                    )
        except Exception as e:
            st.error(f"Export error: {e}")

    # Import Data
    uploaded_backup = st.file_uploader("⬇️ Import Backup", type=['json'])
    if uploaded_backup is not None:
        if st.button("Confirm Import"):
            try:
                temp_path = os.path.join(config.DATA_DIRECTORY, "temp_import.json")
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_backup.getvalue())

                if config.PERSISTENCE_METHOD == "sqlite":
                    success = persistence_manager.import_from_json(temp_path)
                else:
                    success = persistence_manager.import_data(temp_path)
                
                if success:
                    st.success("Data imported successfully! Reloading...")
                    st.session_state.learned_words = persistence_manager.load_learned_words()
                    st.session_state.conversation_history = persistence_manager.load_conversation_history()
                    st.session_state.user_settings = persistence_manager.load_user_settings()
                    os.remove(temp_path)
                    st.rerun()
                else:
                    st.error("Import failed. Please check the file format.")
            except Exception as e:
                st.error(f"Import error: {e}")

    # Danger Zone (Clear All Data)
    st.markdown("---")
    with st.expander("🗑️ Danger Zone"):
        st.warning("⚠️ This action will permanently delete ALL your data.")
        clear_confirmed = st.button("⚠️ CONFIRM DELETE ALL DATA")
        
        if clear_confirmed:
            try:
                if persistence_manager.clear_all_data():
                    st.success("All data cleared successfully!")
                    st.session_state.learned_words = []
                    st.session_state.conversation_history = []
                    st.session_state.user_settings = {}
                    st.rerun()
            except Exception as e:
                st.error(f"Clear error: {e}")

if api_key:
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Main app
    st.title(f"{config.PAGE_ICON} {config.PAGE_TITLE}")
    st.write("Practice Japanese conversation and learn new vocabulary!")
    
    # Auto-save indicator
    if config.AUTO_SAVE_ENABLED and (len(st.session_state.learned_words) > 0 or len(st.session_state.conversation_history) > 0):
        st.success("💾 Your progress is automatically saved")
    
    # Tab navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 Chat Practice", "📊 Progress", "📝 Vocabulary", "🧠 Smart Review", "⚙️ Settings"])
    
with tab1:
    # Use a container to apply the card-like styling
    with st.container(border=True):
        st.subheader("Conversation Practice")
        
        # Load user preferences
        default_level = st.session_state.user_settings.get("preferred_level", config.DEFAULT_LEVEL)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Difficulty level
            level = st.selectbox("Your Level:", config.AVAILABLE_LEVELS, 
                                 index=config.AVAILABLE_LEVELS.index(default_level))
        
        with col2:
            # Topic selection
            topic = st.selectbox("Practice Topic:", config.AVAILABLE_TOPICS)
        
        # Update user settings when level changes
        if level != st.session_state.user_settings.get("preferred_level"):
            st.session_state.user_settings["preferred_level"] = level
            if config.AUTO_SAVE_ENABLED:
                try:
                    save_func()
                except Exception as e:
                    if config.DEBUG_MODE:
                        st.error(f"Auto-save error: {e}")
        
        # Chat interface
        user_input = st.text_input("Type in English or Japanese:", placeholder="こんにちは！ (Hello!)")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            send_button = st.button("Send Message", type="primary")
        with col2:
            if st.button("🎲 Get Topic Suggestion"):
                import random
                suggested_phrases = {
                    "Daily Greetings": ["How do you say 'good morning'?", "What's a polite way to say goodbye?"],
                    "Food & Restaurants": ["How do I order sushi?", "What does 'oishii' mean?"],
                    "Travel": ["How do I ask for directions?", "What's the word for 'train station'?"],
                    "Work & School": ["How do you say 'meeting' in Japanese?", "What does 'ganbatte' mean?"],
                    "Hobbies": ["How do I talk about my hobbies?", "What's the word for 'music'?"],
                    "Free Conversation": ["Tell me about Japanese culture", "What's a common Japanese greeting?"]
                }
                if topic in suggested_phrases:
                    suggestion = random.choice(suggested_phrases[topic])
                    st.info(f"💡 Try asking: \"{suggestion}\"")

        # Your original logic for handling the send_button click goes here
        if send_button and user_input:
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
            6. Keep responses concise and focused
            """
            
            try:
                # Call OpenAI API with configured settings
                response = client.chat.completions.create(
                    model=config.OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input}
                    ],
                    max_tokens=config.MAX_TOKENS,
                    temperature=config.TEMPERATURE
                )
                
                ai_response = response.choices[0].message.content
                
                # Store conversation
                conversation_entry = {
                    "user": user_input,
                    "ai": ai_response,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "level": level,
                    "topic": topic
                }
                
                st.session_state.conversation_history.append(conversation_entry)
                
                # Auto-save after adding conversation
                if config.AUTO_SAVE_ENABLED:
                    try:
                        if config.PERSISTENCE_METHOD == "sqlite":
                            persistence_manager.save_conversation(conversation_entry)
                        else:
                            save_func()
                    except Exception as e:
                        if config.DEBUG_MODE:
                            st.error(f"Auto-save error: {e}")
                
                # Display response
                st.write("**AI Tutor:**")
                st.write(ai_response)
                
                # Extract potential new words (simple heuristic)
                if "(" in ai_response and ")" in ai_response:
                    st.info("💡 Found new vocabulary! Add words to your collection in the Vocabulary tab.")
                
            except Exception as e:
                st.error(f"Error: {e}")
                if config.DEBUG_MODE:
                    st.error(f"Full error details: {str(e)}")
    
    with tab2:
        st.subheader("Learning Progress")
        
        if st.session_state.conversation_history:
            # Create progress dataframe
            df = pd.DataFrame(st.session_state.conversation_history)
            
            # Show statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Conversations", len(df))
            with col2:
                st.metric("Words Learned", len(st.session_state.learned_words))
            with col3:
                today_chats = len([c for c in st.session_state.conversation_history 
                                if c['timestamp'].startswith(datetime.now().strftime("%Y-%m-%d"))])
                st.metric("Today's Practice", today_chats)
            with col4:
                # Calculate streak
                dates = sorted(set([c['timestamp'][:10] for c in st.session_state.conversation_history]))
                current_streak = 1 if dates and dates[-1] == datetime.now().strftime("%Y-%m-%d") else 0
                for i in range(len(dates)-2, -1, -1):
                    current_date = datetime.strptime(dates[i], "%Y-%m-%d")
                    next_date = datetime.strptime(dates[i+1], "%Y-%m-%d")
                    if (next_date - current_date).days == 1:
                        current_streak += 1
                    else:
                        break
                st.metric("Current Streak", f"{current_streak} days")
            
            # Progress over time chart
            if len(df) > 1:
                st.subheader("📈 Practice Activity")
                df['date'] = pd.to_datetime(df['timestamp']).dt.date
                daily_counts = df.groupby('date').size().reset_index(name='conversations')
                
                # Add level and topic breakdowns
                level_counts = df.groupby(['date', 'level']).size().unstack(fill_value=0)
                topic_counts = df.groupby(['date', 'topic']).size().unstack(fill_value=0)
                
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    st.write("**Daily Conversations**")
                    st.bar_chart(daily_counts.set_index('date'))
                
                with chart_col2:
                    st.write("**Practice by Level**")
                    if not level_counts.empty:
                        st.bar_chart(level_counts)
            
            # Show recent conversations
            st.subheader("Recent Conversations")
            num_recent = st.slider("Show recent:", 1, min(20, len(df)), 5)
            
            for i, conv in enumerate(reversed(st.session_state.conversation_history[-num_recent:])):
                with st.expander(f"Conversation {len(st.session_state.conversation_history)-i} - {conv.get('topic', 'Unknown')}"):
                    st.write(f"**You:** {conv['user']}")
                    st.write(f"**AI:** {conv['ai']}")
                    st.write(f"*{conv['timestamp']} | Level: {conv.get('level', 'Unknown')}*")
                    
                    # Option to delete conversation
                    if st.button(f"Delete this conversation", key=f"del_conv_{i}"):
                        st.session_state.conversation_history.remove(conv)
                        if config.AUTO_SAVE_ENABLED:
                            save_func()
                        st.rerun()
        else:
            st.info("Start chatting to see your progress!")
            st.write("**Tips to get started:**")
            st.write("1. Choose your level and a topic")
            st.write("2. Try asking simple questions like 'How do you say hello?'")
            st.write("3. Practice regularly to build your streak!")
    
    with tab3:
        st.subheader("Vocabulary Tracker")
        
        # Quick add section
        with st.expander("➕ Add New Word", expanded=False):
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                new_word_jp = st.text_input("Japanese Word:", placeholder="こんにちは")
            with col2:
                new_word_en = st.text_input("English Meaning:", placeholder="hello")
            with col3:
                st.write("") # spacer
                add_word_btn = st.button("Add Word", type="primary")
        
        if add_word_btn and new_word_jp and new_word_en:
            # Check for duplicates
            existing_words = [w['japanese'] for w in st.session_state.learned_words]
            if new_word_jp not in existing_words:
                word_entry = {
                    "japanese": new_word_jp,
                    "english": new_word_en,
                    "added_date": datetime.now().strftime("%Y-%m-%d"),
                    "practice_count": 0,
                    "review_sequence": [],
                    "last_reviewed": None,
                    "next_review": datetime.now().strftime("%Y-%m-%d")
                }
                st.session_state.learned_words.append(word_entry)
                
                # Auto-save after adding word
                if config.AUTO_SAVE_ENABLED:
                    try:
                        if config.PERSISTENCE_METHOD == "sqlite":
                            persistence_manager.save_learned_word(word_entry)
                        else:
                            save_func()
                    except Exception as e:
                        if config.DEBUG_MODE:
                            st.error(f"Auto-save error: {e}")
                
                st.success(f"Added: {new_word_jp} ({new_word_en})")
                st.rerun()
            else:
                st.warning("This word is already in your vocabulary!")
        
        # Display vocabulary with search and filters
        if st.session_state.learned_words:
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                search_term = st.text_input("🔍 Search vocabulary:", placeholder="Search Japanese or English...")
            with col2:
                sort_by = st.selectbox("Sort by:", ["Added Date", "Japanese", "English", "Practice Count"])
            with col3:
                sort_order = st.selectbox("Order:", ["Newest First", "Oldest First"])
            
            vocab_df = pd.DataFrame(st.session_state.learned_words)
            
            # Apply search filter
            if search_term:
                mask = vocab_df['japanese'].str.contains(search_term, case=False, na=False) | \
                       vocab_df['english'].str.contains(search_term, case=False, na=False)
                vocab_df = vocab_df[mask]
            
            # Apply sorting
            sort_column_map = {
                "Added Date": "added_date",
                "Japanese": "japanese", 
                "English": "english",
                "Practice Count": "practice_count"
            }
            sort_col = sort_column_map[sort_by]
            ascending = sort_order == "Oldest First"
            vocab_df = vocab_df.sort_values(sort_col, ascending=ascending)
            
            # Display results
            st.write(f"**Showing {len(vocab_df)} of {len(st.session_state.learned_words)} words**")
            
            # Display as cards for better readability
            for idx, word in vocab_df.iterrows():
                with st.container():
                    col1, col2, col3, col4 = st.columns([3, 3, 2, 1])
                    
                    with col1:
                        st.write(f"**{word['japanese']}**")
                    with col2:
                        st.write(word['english'])
                    with col3:
                        success_rate = 0
                        if word.get('review_sequence'):
                            correct_count = sum(1 for r in word['review_sequence'] if r.get('correct', False))
                            total_count = len(word['review_sequence'])
                            success_rate = correct_count / total_count if total_count > 0 else 0
                        st.write(f"Success: {success_rate:.0%}")
                    with col4:
                        if st.button("🗑️", key=f"del_{idx}", help="Delete word"):
                            st.session_state.learned_words = [w for w in st.session_state.learned_words if w['japanese'] != word['japanese']]
                            if config.AUTO_SAVE_ENABLED:
                                if config.PERSISTENCE_METHOD == "sqlite":
                                    persistence_manager.delete_learned_word(word['japanese'])
                                else:
                                    save_func()
                            st.rerun()
                    
                    st.divider()
            
            # Download vocabulary as CSV
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                csv = vocab_df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv,
                    "japanese_vocabulary.csv",
                    "text/csv"
                )
            with col2:
                # Bulk operations
                if st.button("🗑️ Clear All Vocabulary"):
                    if st.button("⚠️ CONFIRM - Delete all vocabulary"):
                        st.session_state.learned_words = []
                        if config.AUTO_SAVE_ENABLED:
                            save_func()
                        st.rerun()
            
        else:
            st.info("No vocabulary yet! Add words as you learn them during conversations.")
            st.write("**Quick start tips:**")
            st.write("1. Add common greetings like こんにちは (hello)")
            st.write("2. Learn basic words from your chat conversations")
            st.write("3. Use the Smart Review feature to practice regularly")
    
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
            
            # Quiz settings
            col1, col2 = st.columns(2)
            with col1:
                if len(due_words) > 1:
                    max_words = st.slider(
                        "Max words in this session:",
                        1,
                        min(config.MAX_QUIZ_WORDS, len(due_words)),
                        min(10, len(due_words))
                    )
                else:
                    max_words = 1
                    st.info("Only one word is due for review in this session.")
            with col2:
                quiz_mode = st.selectbox("Quiz mode:", ["Japanese → English", "English → Japanese", "Mixed"])
            
            # Limit words for this session
            session_words = due_words[:max_words]
            
            # Initialize quiz session state
            if 'current_quiz_index' not in st.session_state:
                st.session_state.current_quiz_index = 0
            if 'quiz_feedback' not in st.session_state:
                st.session_state.quiz_feedback = ""
            if 'show_answer' not in st.session_state:
                st.session_state.show_answer = False
            if 'quiz_mode' not in st.session_state:
                st.session_state.quiz_mode = quiz_mode
            
            # Reset if mode changed
            if st.session_state.quiz_mode != quiz_mode:
                st.session_state.current_quiz_index = 0
                st.session_state.quiz_feedback = ""
                st.session_state.show_answer = False
                st.session_state.quiz_mode = quiz_mode
            
            # Check if quiz is finished
            if st.session_state.current_quiz_index < len(session_words):
                # Current word
                current_word = session_words[st.session_state.current_quiz_index]
                
                # Progress indicator
                progress = (st.session_state.current_quiz_index + 1) / len(session_words)
                st.progress(progress)
                st.write(f"Word {st.session_state.current_quiz_index + 1} of {len(session_words)}")
                
                # Determine question direction based on mode
                import random
                if quiz_mode == "Mixed":
                    jp_to_en = random.choice([True, False])
                elif quiz_mode == "Japanese → English":
                    jp_to_en = True
                else:  # English → Japanese
                    jp_to_en = False
                
                # Quiz interface
                if jp_to_en:
                    st.markdown("### What does this mean in English?")
                    st.markdown(f"# {current_word['japanese']}")
                    correct_answer = current_word['english']
                else:
                    st.markdown("### How do you say this in Japanese?")
                    st.markdown(f"# {current_word['english']}")
                    correct_answer = current_word['japanese']
                
                # Show stats for current word
                if current_word.get('review_sequence'):
                    success_rate = calculate_success_rate(current_word)
                    streak = get_current_streak(current_word)
                    st.caption(f"Your stats: {success_rate:.1%} success rate, {streak} current streak")
                
                # User input
                user_answer = st.text_input("Your answer:", key=f"answer_{st.session_state.current_quiz_index}_{jp_to_en}")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("Check Answer", type="primary"):
                        if user_answer.strip():
                            # Check if answer is correct (case-insensitive)
                            correct_answer_clean = correct_answer.lower().strip()
                            user_answer_clean = user_answer.lower().strip()

                            is_correct = user_answer_clean == correct_answer_clean

                            # Update the word with ML functions
                            updated_word, success_rate, current_streak = update_word_after_review(
                                current_word, is_correct
                            )

                            # Update the word in session state
                            for i, word in enumerate(st.session_state.learned_words):
                                if word['japanese'] == current_word['japanese']:
                                    st.session_state.learned_words[i] = updated_word
                                    break

                            # Auto-save after review
                            if config.AUTO_SAVE_ENABLED:
                                try:
                                    if config.PERSISTENCE_METHOD == "sqlite":
                                        persistence_manager.save_learned_word(updated_word)
                                    else:
                                        save_func()
                                except Exception as e:
                                    if config.DEBUG_MODE:
                                        st.error(f"Auto-save error: {e}")

                            # Show feedback
                            if is_correct:
                                st.session_state.quiz_feedback = f"✅ Correct! '{current_word['japanese']}' means '{current_word['english']}'."
                            else:
                                st.session_state.quiz_feedback = f"❌ Incorrect. The correct answer is '{correct_answer}'."
                                st.session_state.show_answer = True
                        else:
                            st.session_state.quiz_feedback = "Please enter an answer before checking."
                            st.session_state.show_answer = False

                with col2:
                    if st.button("Show Answer"):
                        st.session_state.show_answer = True
                        st.session_state.quiz_feedback = f"🔎 The correct answer is: '{correct_answer}'."

                with col3:
                    if st.button("Next Word"):
                        if st.session_state.current_quiz_index < len(session_words) - 1:
                            st.session_state.current_quiz_index += 1
                            st.session_state.quiz_feedback = ""
                            st.session_state.show_answer = False

                # Display feedback
                if st.session_state.quiz_feedback:
                    st.info(st.session_state.quiz_feedback)

            else:
                st.success("🎉 Review session complete! Great job!")
                if st.button("Restart Review Session"):
                    st.session_state.current_quiz_index = 0
                    st.session_state.quiz_feedback = ""
                    st.session_state.show_answer = False

        else:
            st.info("No words are due for review today. Keep practicing and check back tomorrow!")

    with tab5:
        st.subheader("⚙️ Settings")
        st.write("Adjust your preferences and app configuration here.")
        # Example settings (expand as needed)
        st.write(f"Persistence method: **{config.PERSISTENCE_METHOD}**")
        st.write(f"Data directory: **{config.DATA_DIRECTORY}**")
        st.write(f"Auto-save enabled: **{config.AUTO_SAVE_ENABLED}**")
        st.write(f"OpenAI model: **{config.OPENAI_MODEL}**")
        st.write(f"Max tokens: **{config.MAX_TOKENS}**")
        st.write(f"Temperature: **{config.TEMPERATURE}**")
        st.write(f"Max quiz words per session: **{config.MAX_QUIZ_WORDS}**")
        st.write("To change advanced settings, edit your config file.")

import streamlit as st
import time

# --- Inject custom CSS from style.css ---
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)







# --- A small script to demonstrate the custom danger button CSS ---
# This part is just for demonstration, you'd integrate it with your actual logic
st.markdown("""
<script>
    const buttons = window.parent.document.querySelectorAll('[data-testid="stButton"] button');
    buttons.forEach(button => {
        if (button.textContent.includes('Clear All Data')) {
            button.classList.add('danger-button');
        }
    });
</script>
""", unsafe_allow_html=True)