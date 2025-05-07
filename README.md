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

### Chunking

Storybot provides four chunking heuristics:

- Fixed-size
- By arbitrary delimiters
- Semantic w/ BERT (greedy): immediately separates if cosine similarity between adjacent sentences falls below a given threshold
- Semantic w/ BERT (recursive): uses a divide-and-conquer strategy to split based on lowest adjacent sentence similarities

### Retrieval

Storybot performs document retrieval as follows. Let $q$ be the embedding of a given query and $\{ c_1, c_2, ..., c_n \}$ be the embeddings of given chunks. We calculate similarity between $q$ and $c_i$ as follows:

$$
\text{sim}(q, c_i) = \text{cos}(q, c_i) + w_i
$$

where:

$$w_i = D\[i, j^* \]
$$

$D$ is a symmetric distance matrix which applies a Gaussian decay penalty based on the index distance between chunks $i$ and $j$:

$$
D\[i, j\] = \text{exp}(-\gamma(i - j)^2)
$$

Specifically, we find the index distance between $c_i$ and $c_{j^*} $, where 

$$
j^* = \text{argmax}_j \text{cos}(q, c_j)
$$

This leverages the assumption that in narrative texts, adjacent chunks are often contextually related, unlike in more structured documents (eg. research papers) whose sections are more semantically isolated.
