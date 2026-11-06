# AI Engineer Bootcamp: Course Code

Source code for the lectures in the AI Engineer Bootcamp course.

## Setup

1. Install **Python 3.12** (newer Python versions are not supported by some of the pinned packages yet).
2. Create and activate a virtual environment:
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```
3. Install the pinned dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the folder you're working in with the keys the lecture uses, for example:
   ```
   OPENAI_API_KEY=your-key-here
   COHERE_API_KEY=your-key-here
   ```

## Updates

- **November 2026:** OpenAI shut down `gpt-4` and `gpt-3.5-turbo`. The code that used them now uses
  `gpt-4o-mini`. Dependencies are now pinned to the versions used in the lectures, because newer
  LangChain releases moved several imports used in the course.
