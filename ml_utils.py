from datetime import datetime, timedelta
import json

def calculate_success_rate(word):
    """Calculate success rate for a word based on review history"""
    if 'review_history' not in word or not word['review_history']:
        return 0.0
    
    correct_reviews = sum(1 for review in word['review_history'] if review['correct'])
    total_reviews = len(word['review_history'])
    return correct_reviews / total_reviews if total_reviews > 0 else 0.0

def get_current_streak(word):
    """Get current streak of correct answers"""
    if 'review_history' not in word or not word['review_history']:
        return 0
    
    streak = 0
    # Go through review history in reverse (most recent first)
    for review in reversed(word['review_history']):
        if review['correct']:
            streak += 1
        else:
            break
    return streak

def calculate_next_review_interval(success_rate, current_streak):
    """Calculate when the word should be reviewed next (in days)"""
    # Handle None values - set defaults for new words
    if current_streak is None:
        current_streak = 0
    if success_rate is None:
        success_rate = 0.0
    
    # Base interval starts at 1 day
    base_interval = 1
    
    # Adjust based on success rate
    if success_rate >= 0.9:  # 90%+ success rate
        base_interval = 7  # Weekly reviews
    elif success_rate >= 0.7:  # 70-89% success rate
        base_interval = 3  # Every 3 days
    elif success_rate >= 0.5:  # 50-69% success rate
        base_interval = 2  # Every 2 days
    else:  # Less than 50% success rate
        base_interval = 1  # Daily reviews
    
    # Increase interval based on current streak
    interval = base_interval * (1 + current_streak // 2)  # Now safe from None
    
    # Cap the maximum interval at 30 days
    return min(interval, 30)

def get_words_due_for_review(learned_words):
    """Get words that are due for review today"""
    today = datetime.now().date()
    due_words = []
    
    for word in learned_words:
        # If no next_review date, it's due for review
        if 'next_review' not in word:
            due_words.append(word)
        else:
            try:
                next_review_date = datetime.strptime(word['next_review'], '%Y-%m-%d').date()
                if next_review_date <= today:
                    due_words.append(word)
            except (ValueError, TypeError):
                # If date parsing fails, assume it's due
                due_words.append(word)
    
    return due_words

def update_word_after_review(word, is_correct):
    """Update word data after a review session"""
    # Initialize review history if it doesn't exist
    if 'review_history' not in word:
        word['review_history'] = []
    
    # Add this review to history
    review_entry = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'correct': is_correct
    }
    word['review_history'].append(review_entry)
    
    # Calculate new success rate and streak
    success_rate = calculate_success_rate(word)
    current_streak = get_current_streak(word)
    
    # Calculate next review interval
    next_interval = calculate_next_review_interval(success_rate, current_streak)
    
    # Set next review date
    next_review_date = datetime.now() + timedelta(days=next_interval)
    word['next_review'] = next_review_date.strftime('%Y-%m-%d')
    
    # Update last reviewed
    word['last_reviewed'] = datetime.now().strftime('%Y-%m-%d')
    
    return word, success_rate, current_streak

def get_review_stats(learned_words):
    """Get overall statistics about the learning progress"""
    if not learned_words:
        return {
            'total_words': 0,
            'words_with_reviews': 0,
            'average_success_rate': 0.0,
            'total_reviews': 0
        }
    
    total_words = len(learned_words)
    words_with_reviews = 0
    total_success_rates = 0
    total_reviews = 0
    
    for word in learned_words:
        if 'review_history' in word and word['review_history']:
            words_with_reviews += 1
            success_rate = calculate_success_rate(word)
            total_success_rates += success_rate
            total_reviews += len(word['review_history'])
    
    average_success_rate = total_success_rates / words_with_reviews if words_with_reviews > 0 else 0.0
    
    return {
        'total_words': total_words,
        'words_with_reviews': words_with_reviews,
        'average_success_rate': average_success_rate,
        'total_reviews': total_reviews
    }