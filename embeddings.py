from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForTokenClassification, BertModel, BertTokenizer, pipeline
from sklearn.metrics.pairwise import cosine_similarity

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import numpy as np
import os
import re

import torch

MAX_EMBED_SIZE = 1536

class Embeddings():
    def __init__(self, story_name, embed_path, gamma):
        self.story_name = story_name
        self.embed_path = embed_path
        self.client = OpenAI()
        self.embeds = {}
        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )
        self.chroma = chromadb.PersistentClient(path=self.embed_path)
        self.collection = self.chroma.get_or_create_collection(
            name=f"{self.story_name}_embeds",
            embedding_function=self.openai_ef
        )
        self.doc_embed = None
        self.gamma = gamma

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def __getitem__(self, key):
        return self.embeds[key]

    def embed_sentence(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors='pt')
        outputs = self.model(**inputs).last_hidden_state[:, 0, :]
        return outputs

    def cosine_sim(self, s1, s2):
        X = self.embed_sentence(s1).detach().numpy()
        Y = self.embed_sentence(s2).detach().numpy()
        return cosine_similarity(X, Y)

    def get_cosine_sims(self):
        db = self.query_all()
        self.n = len(db['ids'])
        self.populate_embeds()
        E = self.get_all_embeddings()
        n = np.linalg.norm(E, axis=1)
        normalized_embeddings = E / n[:, np.newaxis]
        S = np.dot(normalized_embeddings, normalized_embeddings.T)
        np.fill_diagonal(S, 0)
        return S

    def embed_chunks(self, chunks):
        self.chunks = chunks
        # initial embed calculation
        self.collection.upsert(
            documents=[chunk for i, chunk in chunks.items()],
            ids=[f"id{i}" for i in range(len(chunks))]
        )

    def update_embed_context(self):
        # calculate document embed and update chunk embeds
        self.doc_embed = self.calculate_document_embed()
        old_embeds = self.get_all_embeddings()
        new_embeds = list(old_embeds + 0.5 * self.doc_embed)
        new_embeds = list([list(embed) for embed in new_embeds])
        self.collection.upsert(
            embeddings=[embed for embed in new_embeds],
            ids=[f"id{i}" for i in range(len(new_embeds))]
        )

    def sort_ids(self, ids):
        def extract_number(string):
            match = re.search(r'\d+', string)
            return int(match.group()) if match else float('inf')
        return sorted(ids, key=extract_number)

    # for debugging
    def test_ner(self):
        db = self.query_all()
        self.n = len(db['ids'])
        ids = db['ids']
        embeds = db['embeddings']
        docs = db['documents']
        self.populate_embeds(ids, embeds, docs)

        _, first_doc = self.embeds[2]
        # print(first_doc)
        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        nlp = pipeline("ner", model=model, tokenizer=tokenizer)
        results = nlp(first_doc)
        print(results)

    # TODO make this more efficient
    def query_chunks(self, queries, k):
        q = self.embed_query(queries[0])
        self.populate_embeds()
        E = self.get_sorted_embeds()
        # Retrieve closest neighbor
        closest, _ = self.k_closest_search(q, E, 1, None)
        # Compute D, symmetric matrix of distances
        D = self.get_distance_matrix()
        # Recompute k-closest neighbors w/ D
        top_k_idxs, scores = self.k_closest_search(q, E, k, D[closest])
        top_k_chunks = self.get_closest_chunks(top_k_idxs)
        return top_k_chunks, scores

    def get_closest_chunks(self, idxs):
        chunks = []
        for i in idxs:
            chunks.append(self.embeds[i][1])
        return chunks

    def k_closest_search(self, q, E, k, w):
        q_n = q / np.linalg.norm(q)
        E_n = E / np.linalg.norm(E, axis=1, keepdims=True)
        s = np.dot(E_n, q_n)
        # if w is not None:
        #     s = np.ravel(s + w)
        top_k = np.argpartition(s, -k)[-k:]
        top_k = top_k[np.argsort(-s[top_k])]
        return top_k, s[top_k]

    def embed_query(self, query):
        response = self.client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding

    def get_sorted_embeds(self):
        E = np.zeros((self.n, MAX_EMBED_SIZE))
        for i, (embed, _) in self.embeds.items():
            e_i = np.array(embed)
            E[i] = e_i
        return E

    def populate_embeds(self):
        db = self.query_all()
        self.n = len(db['ids'])
        ids = db['ids']
        embeds = db['embeddings']
        docs = db['documents']
        for id, embed, doc in zip(ids, embeds, docs):
            i = int(re.findall(r'\d+', id)[0])
            self.embeds[i] = (embed, doc)

    def get_distance_matrix(self):
        i = np.arange(self.n)
        D = np.exp(-self.gamma * (i[:, None] - i[None, :]) ** 2)
        np.fill_diagonal(D, 1)
        return D

    def calculate_document_embed(self):
        chunks = self.get_all_embeddings()
        return np.mean(chunks, axis=0)

    def query_all(self):
        return self.collection.get(include=['embeddings', 'documents'])

    def get_all_embeddings(self):
        return np.array(self.query_all()['embeddings'])

if __name__ == '__main__':
    embeddings = Embeddings("PART_1_long", "PART_1_long_embed", None)
    e = embeddings.embed_sentence("this is a test sentence")
    print(e)