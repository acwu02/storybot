from openai import OpenAI
from tinydb import TinyDB, Query
from fileparser import FileParser
from embeddings import Embeddings
import json
import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--embed', action='store_true')
parser.add_argument('-m', '--message', action='store_true')
parser.add_argument('-q', '--query', action='store_true')
parser.add_argument('-c', '--cosines', action='store_true')
parser.add_argument('--debug_ner', action='store_true')
args, unknown_args = parser.parse_known_args()

client = OpenAI()

config_path = 'config.json'
system_prompt = ''
summarize_chunk = ''
summarize_story = ''
answer_given_context = ''
answer_no_context = ''
gpt3 = ''
gpt4 = ''

with open(config_path, 'r') as file:
    config = json.load(file)
    system_prompt = config['prompts']['system']
    summarize_chunk = config['prompts']['summarize_chunk']
    summarize_story = config['prompts']['summarize_story']
    answer_given_context = config['prompts']['answer_given_context']
    answer_no_context = config['prompts']['answer_no_context']
    gpt3 = config['models']['gpt-3']
    gpt4 = config['models']['gpt-4']

assert system_prompt
assert summarize_chunk
assert summarize_story
assert answer_given_context
assert answer_no_context
assert gpt3
assert gpt4

filename = "PART_1_AND_2"

class API():
    def __init__(
        self,
        file_path,
        chunk_path,
        summaries_path,
        embed_path,
        db_path,
        chunk_size,
        overlap,
        temperature,
        k,
        gamma
    ):
        self.summary = ""
        self.file_path = file_path
        self.chunk_path = chunk_path
        self.summaries_path = summaries_path
        self.embed_path = embed_path
        self.db_path = db_path
        self.gamma = gamma
        self.chunks = dict()
        self.story_name = self.file_path[:self.file_path.find('.pdf')]
        self.embeddings = Embeddings(self.story_name, self.embed_path, self.gamma)
        self.fileparser = FileParser(file_path, chunk_size, overlap, self.embeddings, None, None) # TODO fill in missing args
        self.db = TinyDB(self.db_path)
        self.temperature = temperature
        self.k = k

    def request(self, model, role, message):
        request = self.message_to_request(role, message)
        completion = client.chat.completions.create(
            model=model,
            messages=[request],
            temperature=self.temperature
        )
        response = completion.choices[0].message.content
        return response


    def message_to_request(self, role, message):
        return {"role": role, "content": message}

    # TODO figure out how much context wanted
    def update_db(self, response):
        self.db.insert({"role": "assistant", "content": response})

    def chunk(self):
        chunks = self.fileparser.parse_file()
        self.chunks = dict([(i, chunk) for i, chunk in enumerate(chunks)])
        self.embed()
        self.write_chunks()

    def embed(self):
         self.embeddings.embed_chunks(self.chunks)

    def get_responses(self):
        responses = [self.request(
            gpt3,
            "system",
            summarize_chunk + chunk
        ) for i, chunk in self.chunks.items()]
        self.responses = dict([(i, response)
                    for i, response in enumerate(responses)])
        with open(self.summaries_path, 'w') as f:
            for i, response in self.responses.items():
                f.write(response)
                f.write('\n')
        print("Summaries written")

    def write_chunks(self):
        with open(self.chunk_path, 'w') as f:
            for i, chunk in self.chunks.items():
                f.write(f"CHUNK {i}")
                f.write(chunk)
                f.write('\n')

    def query(self, query):
        chunks, scores = self.embeddings.query_chunks([query], self.k)
        self.print_query_chunks(chunks, scores)

    def read_summary(self):
        summary = ""
        with open(self.summaries_path, 'r') as f:
            summary = f.read()
        return summary

    def retrieve_from_embeds(self, query):
        # TODO
        pass

    def send_message(self, message):
        summary = self.read_summary()
        k_closest_chunks, scores = self.embeddings.query_chunks([message], self.k)
        self.print_query_chunks(k_closest_chunks, scores)
        response = self.request(
            gpt4,
            "system",
            "SUMMARY: " + summary +
            # "EXCERPTS: " + " ".join(k_closest_chunks) +
            "QUESTION: " + message
        )
        print(response)

    def print_query_chunks(self, chunks, scores):
        for i, doc in enumerate(chunks):
            print(f"DOC {i}")
            print(f"SCORE: {scores[i]}")
            for j in range(50):
                print("-", end='')
            print('\n' + doc + '\n')

    def print_chunks(self):
        for i, chunk in self.chunks.items():
            print("CHUNK" + str(i) + ":")
            for i in range(50):
                print("-", end='')
            print('\n')
            print(chunk)

    def print_responses(self):
        for i, response in self.responses.items():
            print("RESPONSE" + str(i) + ":")
            for i in range(50):
                print("-", end='')
            print('\n')
            print(response)

if __name__ == '__main__':
    api = API(
        f"{filename}.pdf",
        f"{filename}_chunks.txt",
        f"{filename}_summaries.txt",
        f"{filename}_embeddings",
        f"{filename}_db.json",
        2000, 200, 0.7, 5, 0.0001
    )

    api.fileparser.parse_file()

    if args.debug_ner:
        api.embeddings.test_ner()

    if args.message:
        message = input("Enter message: ")
        api.send_message(message)
    elif args.query:
        query = input("Enter query: ")
        api.query(query)
    elif args.cosines:
        cosine_sims = api.embeddings.get_cosine_sims()
        with open('cosine_sims.txt', 'w') as f:
            for row in cosine_sims:
                f.write(' '.join(map(str, row)) + '\n')
                f.write(str(row) + '\n')
    elif args.embed:
        api.chunk()
    else:
        if sys.argv[1]:
            query =  " ".join(sys.argv[1:])
            print(query)
            api.query(query)
        else:
            api.chunk()
            api.get_responses()
