from datetime import datetime
from datetime import timedelta


#function to calculate success rate from a sequence of boolean values
#where True indicates success and False indicates failure
def calculate_success_rate(sequence):
    if len(sequence) == 0:
        return 0.0
    
    success_count = sum(sequence)  # Smart use of True=1, False=0!
    return success_count / len(sequence)  #returns a float between 0.0 and 1.0


def get_current_streak(sequence):
    if not sequence:
        return 0
    streak = 0
    #counts from the end of the sequence until it finds a False
    #or until it reaches the beginning of the sequence
    for i in reversed(sequence):
        if i:
            streak += 1
        else:
            break

def calculate_next_review_interval(success_rate, current_streak, base_interval=1):
    # Base interval in days
    if success_rate >= 0.8:
        interval = base_interval * (1 + current_streak // 2)  # Increase interval for high success
    elif success_rate >= 0.5:
        interval = base_interval * (1 + current_streak // 3)  # Moderate increase
    else:
        interval = base_interval  # No increase for low success

    return min(interval, 30)  # Max at 30 days, dont want antyone waiting too long

def get_words_due_for_review(learned_words):
    today = datetime.now()
    due_words = []
    for word in learned_words:
        if word.get('next_review_date'):
            next_review_date = datetime.strptime(word['next_review_date'], "%Y-%m-%d")
            if next_review_date <= today:
                due_words.append(word)
        else:
            # New word with no review history - due for first review
            due_words.append(word)
    return due_words

def update_word_after_review(word, current_answer):
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Update review sequence
    word['review_sequence'].append(current_answer)
    
    # Calculate success rate
    success_rate = calculate_success_rate(word['review_sequence'])
    
    # Get current streak
    current_streak = get_current_streak(word['review_sequence'])
    

    # Calculate next review interval
    next_interval = calculate_next_review_interval(success_rate, current_streak)
    
    # Update next review date
    word['next_review_date'] = (datetime.now() + timedelta(days=next_interval)).strftime("%Y-%m-%d")
    
    # Update last review date
    word['last_review_date'] = today

    # update practice count
    word['practice_count'] = word.get('practice_count', 0) + 1
    
    return word

def get_review_stats(learned_words):
    if not learned_words:
        return {
            'total_words': 0,
            'words_with_reviews': 0,
            'average_success_rate': 0.0,
            'total_reviews': 0
        }
    
    total_words = len(learned_words)
    
    # Get list of words that have reviews
    reviewed_words = [word for word in learned_words if word.get('review_sequence')]
    words_with_reviews = len(reviewed_words)
    
    total_reviews = sum(len(word['review_sequence']) for word in reviewed_words)
    average_success_rate = (sum(calculate_success_rate(word['review_sequence']) for word in reviewed_words) / len(reviewed_words)) if reviewed_words else 0.0
    
    return {
        'total_words': total_words,
        'words_with_reviews': words_with_reviews,
        'average_success_rate': average_success_rate,
        'total_reviews': total_reviews
    }


