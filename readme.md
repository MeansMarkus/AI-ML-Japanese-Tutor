# Japanese AI Tutor

A modern, interactive Japanese vocabulary and kanji learning app powered by AI and machine learning. This project helps users efficiently learn and review Japanese words using adaptive quizzes, spaced repetition, and personalized feedback. Data is stored locally using either JSON or SQLite, with robust import/export and backup features.

---

## Features

- **Smart Review System:** Adaptive spaced repetition algorithm schedules reviews for optimal retention.
- **Quiz Modes:** Practice Japanese → English, English → Japanese, or Mixed.
- **Personalized Feedback:** Immediate feedback on answers, with success rate and streak tracking.
- **Progress Tracking:** Visual metrics for total words, reviews, and learning statistics.
- **Flexible Data Storage:** Choose between JSON and SQLite for persistence.
- **Import/Export & Backups:** Easily backup or transfer your learning data.
- **Configurable UI:** Change themes, quiz settings, and more via `config.py` or the sidebar (in debug mode).
- **Debug Tools:** Enable debug mode for advanced configuration and error messages.

---

## How It Works

### 1. Data Model

- Each vocabulary word is stored as a dictionary with fields such as:
  - `japanese`: The Japanese word or phrase.
  - `english`: The English meaning.
  - `review_sequence`: A list of review attempts, each with date and correctness.
  - `next_review`: The next scheduled review date (spaced repetition).
  - `last_reviewed`: The last time the word was reviewed.

### 2. Spaced Repetition Algorithm

- The app uses a custom spaced repetition algorithm inspired by SM-2 (used in Anki).
- After each review, the app calculates:
  - **Success Rate:** Percentage of correct answers for the word.
  - **Current Streak:** Number of consecutive correct answers.
  - **Next Review Interval:** Based on success rate and streak, the interval (in days) until the next review is increased or decreased.
- This ensures that words you struggle with appear more often, while mastered words are reviewed less frequently.

### 3. Quiz Session Logic

- When you start a review session:
  - The app selects all words due for review (where `next_review` is today or earlier).
  - You can choose how many words to review in this session (slider).
  - Quiz mode can be Japanese → English, English → Japanese, or Mixed.
  - Each answer is checked for correctness (case-insensitive).
  - Feedback is shown immediately, and your review history is updated.
  - The app tracks your progress, success rate, and streaks.

### 4. Data Persistence

- **JSON Mode:** All words and review data are stored in a single JSON file (`data/learned_words.json`).
- **SQLite Mode:** Data is stored in a local SQLite database (`data/japanese_tutor.db`), supporting larger datasets and more robust storage.
- You can switch between modes via `config.py`, the sidebar (in debug mode), or environment variables.
- Import/export and backup features allow you to move data between formats or devices.

### 5. Configuration

- All settings are centralized in `config.py`, including:
  - Persistence method (`PERSISTENCE_METHOD`)
  - UI theme and icons
  - Quiz and review settings (e.g., max words per session)
  - OpenAI model and API settings (for advanced features)
- You can override settings with environment variables or the sidebar (in debug mode).

### 6. User Interface

- Built with [Streamlit](https://streamlit.io/) for a fast, interactive web experience.
- Sidebar provides setup, data management, import/export, and advanced options.
- Main tabs include:
  - **Add Vocabulary:** Add new words and meanings.
  - **Smart Review:** Practice words due for review.
  - **Progress:** Visualize your learning stats.
  - **Settings:** View and adjust app configuration.

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/MeansMarkus/AI-ML-Japanese-Tutor.git
cd AI-ML-Japanese-Tutor
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

- Copy `.env.example` to `.env` and set your OpenAI API key (if using AI features).
- Edit `config.py` to adjust app settings, persistence method, and quiz options.

### 4. Run the App

```bash
streamlit run app.py
```

---

## Data Persistence

The app supports two data storage methods:

- **JSON:** Human-readable, stored at `data/learned_words.json`
- **SQLite:** Robust database, stored at `data/japanese_tutor.db`

You can switch between them by:
- Editing `PERSISTENCE_METHOD` in `config.py`
- Using the sidebar dropdown (in debug mode)
- Setting the `JAPANESE_TUTOR_PERSISTENCE_METHOD` environment variable

See [DataPersistence_guide.txt](DataPersistence_guide.txt) for full details.

---

## Usage

1. **Add Vocabulary:** Enter new Japanese words and their meanings.
2. **Review Words:** Use the Smart Review tab to practice words due for review.
3. **Track Progress:** View your stats and streaks in the Progress tab.
4. **Manage Data:** Export, import, or clear your data from the sidebar.

---

## Configuration

All major settings are in `config.py`, including:

- Persistence method (`PERSISTENCE_METHOD`)
- UI theme and icons
- Quiz and review settings
- OpenAI model and API settings

You can also override settings with environment variables or the sidebar (debug mode).

---

## Advanced Features

- **Debug Mode:** Enable in `config.py` for extra sidebar options and error messages.
- **Backups:** Automatic and manual backups are stored in `data/backups/`.
- **Import/Export:** Move data between JSON and SQLite, or between devices.

---

## Troubleshooting

- **Slider Error:** If you see a slider error, make sure you have more than one word due for review.
- **File/DB Errors:** Check that the `data/` folder exists and is writable.
- **Debugging:** Enable `DEBUG_MODE` in `config.py` for detailed error output.

---

## Roadmap / Ideas for Machine Learning Expansion

- **Personalized Review Scheduling:** Use ML models to predict optimal review intervals based on user performance.
- **Adaptive Quiz Difficulty:** Dynamically adjust quiz difficulty using reinforcement learning or bandit algorithms.
- **Error Pattern Analysis:** Cluster user mistakes to target weak areas.
- **Speech Recognition:** Integrate pronunciation practice and scoring.
- **Natural Language Feedback:** Use LLMs to generate explanations and mnemonics.

---

## Contributing

Pull requests and suggestions are welcome! Please open an issue to discuss major changes.

---

## License

MIT License

---

## Acknowledgments

- [Streamlit](https://streamlit.io/)
- [OpenAI](https://openai.com/)
- Markus Means

---
