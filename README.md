# Twitter Sentiment Analysis: VADER vs. RoBERTa

🔗 **Live app:** [huggingface.co/spaces/Suman-24/twitter-sentiment-analysis](https://huggingface.co/spaces/Suman-24/twitter-sentiment-analysis)

## What this project is about

I wanted to answer a simple but interesting question: when you analyze the sentiment of a tweet, does it actually matter *which* method you use? Tweets are messy — sarcasm, slang, hashtags, emojis — so I built an app that runs the same tweet through two very different approaches and lets you compare the results side by side.

- **VADER** — a fast, rule-based model that scores sentiment using a pre-built lexicon of words and simple heuristics (like punctuation and capitalization). It has no real understanding of context, but it's lightweight and instant.
- **RoBERTa** — a transformer-based model (via `cardiffnlp`) that's been fine-tuned specifically on tweets. It actually understands context, so it tends to do better with sarcasm, negation, and slang — but it's heavier and slower.

Putting them head-to-head made it obvious just how differently a "dumb but fast" model and a "smart but slower" model can read the same piece of text.

## What the app does

The app has three pages:

1. **Live Analysis** — enter a keyword, and the app fetches recent tweets in real time using the Twitter/X API (via Tweepy), then runs both models on them so you can see live sentiment side by side.
2. **Model Comparison** — a deeper look at how VADER and RoBERTa perform against each other, including how their predictions line up (or don't) on the same data.
3. **About** — background on the project, the models, and the reasoning behind the comparison.

## Tech stack

- **Sentiment models:** VADER (NLTK) and `cardiffnlp/twitter-roberta-base-sentiment` (Hugging Face Transformers)
- **Live data:** Twitter/X API via Tweepy
- **Deployment:** Docker container on Hugging Face Spaces (port 7860)
- **Secrets management:** Twitter bearer token stored securely as a Hugging Face Space secret, never hardcoded

## Dataset & evaluation

For the offline model comparison, I used the Sentiment140 dataset. Along the way I ran into a labeling quirk in the raw dataset that was quietly skewing the evaluation metrics — once I corrected the label mapping, the comparison numbers became far more realistic and trustworthy. I also caught and fixed a small but important typo in the RoBERTa model name (`cariffnlp` → `cardiffnlp`) that was silently breaking model loading.

Small bugs like these are a good reminder that in NLP projects, a lot of the real work is in the plumbing — not just picking the right model.

## Why I built this

This is part of my portfolio as I move into data science and AI roles, with a particular interest in NLP and applied machine learning. I wanted a project that wasn't just "train a model and report accuracy," but that actually explored a real trade-off — speed and simplicity vs. context and accuracy — in a way that's interactive and easy for anyone to try for themselves.

## Try it yourself

Head over to the [live Space](https://huggingface.co/spaces/Suman-24/twitter-sentiment-analysis), type in a keyword, and watch VADER and RoBERTa disagree with each other in real time.

## Running it locally

If you'd rather run it on your own machine instead of using the hosted Space:

```bash
# Clone the repo
git clone https://github.com/Suman-2411/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis

# Install dependencies
pip install -r requirements.txt
```

You'll need a Twitter/X API bearer token for the Live Analysis page to fetch tweets. Create a `.env` file in the project root:

```
TWITTER_BEARER_TOKEN=your_token_here
```

Then start the app:

```bash
python app.py
```

The app will be available at `http://localhost:7860`.

> **Note:** update the commands above if your entry point, dependency file, or folder structure differs — swap `app.py`/`requirements.txt` for whatever your repo actually uses.

## License

This project is licensed under the MIT License — feel free to use, modify, and build on it.

## Author

**Suman Thangadurai**
- GitHub: [Suman-2411](https://github.com/Suman-2411)
- LinkedIn: [suman-thangadurai](https://linkedin.com/in/suman-thangadurai)
