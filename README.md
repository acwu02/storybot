# Storybot

Storybot is a tool to chunk, embed, and query longer-form texts, particularly narrative works, to assist during the drafting process.

It runs on OpenAI's GPT models to generate responses and ChromaDB to store vector embeddings.

Note: You need a working OpenAI API key to use Storybot.

## Setup

From the root directory, run:

```
pip install -r requirements.txt
```

Next, create an .env file:

```
touch .env
```

Open it with your favorite text editor, retrieve your OpenAI API key, and define it in .env as follows:

```
OPENAI_API_KEY=[insert API key here]
```

You're all set!

## Workflow

First, upload a story to be parsed in either .txt or .pdf form to the root directory. 

Then, chunk it:

```
python3 storybot.py -c
```

Try out a query or two to make sure relevant chunks are being retrieved;

```
python3 storybot.py -q
```

To send messages to the LLM:

```
python3 storybot.py -m
```

## Design

Storybot performs document retrieval as follows. Let $q$ be the embedding of a given query and $\mathcal{D} = {d_1, d_2, ..., d_n}$
``
