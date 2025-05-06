from openai import OpenAI
from tinydb import TinyDB, Query
from fileparser import FileParser
from embeddings import Embeddings
import json
import os
import sys
import argparse
import pyfiglet

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--chunk', action='store_true')
parser.add_argument('-m', '--message', action='store_true')
parser.add_argument('-q', '--query', action='store_true')
parser.add_argument('-s', '--summarize', action='store_true')
parser.add_argument('-r', '--reset-chromadb', action='store_true')
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

chunk_size = None
temperature = None
k = None
gamma = None

with open(config_path, 'r') as file:
    config = json.load(file)
    system_prompt = config['prompts']['system']
    summarize_chunk = config['prompts']['summarize_chunk']
    summarize_story = config['prompts']['summarize_story']
    answer_given_context = config['prompts']['answer_given_context']
    answer_no_context = config['prompts']['answer_no_context']
    chunk_size = config['params']['chunk_size']
    overlap = config['params']['overlap']
    temperature = config['params']['temperature']
    k = config['params']['num_chunks']
    gamma = config['params']['distance_weight']
    gpt3 = config['models']['gpt-3']
    gpt4 = config['models']['gpt-4']

assert system_prompt
assert summarize_chunk
assert summarize_story
assert answer_given_context
assert answer_no_context
assert chunk_size
assert overlap
assert temperature
assert k
assert gamma
assert gpt3
assert gpt4


class API():
    def __init__(
        self,
        file_path,
        chunk_size,
        overlap,
        temperature,
        k,
        gamma
    ):
        self.summary = ""
        self.file_path = file_path
        self.filename = os.path.splitext(os.path.basename(self.file_path))[0]
        self.chunk_path = f"./outputs/{self.filename}_chunks.txt"
        self.summaries_path = f"./outputs/{self.filename}_summaries.txt"
        self.embed_path = f"./chromadb_outputs/{self.filename}_embeddings"
        self.db_path = f"./chromadb_outputs/{self.filename}_db.json"

        self.gamma = gamma
        self.chunks = dict()

        self.embeddings = Embeddings(
            self.filename, self.embed_path, gamma=self.gamma)
        self.fileparser = FileParser(file_path, chunk_size=chunk_size, overlap=overlap,
                                     embeddings=self.embeddings, threshold=None, min_chunk_len=None)
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
        chunks = self.fileparser.chunk_fixed_size()
        self.chunks = dict([(i, chunk) for i, chunk in enumerate(chunks)])
        self.embed()
        self.write_chunks()

    def embed(self):
        self.embeddings.embed_chunks(self.chunks)

    def get_chunks_from_file(self):
        if os.path.exists(self.chunk_path):
            with open(self.chunk_path, 'r') as f:
                lines = ''.join(f.readlines()).split('CHUNK')
                for i, line in enumerate(lines):
                    if line:
                        line = line.split(':')[1].strip()
                        self.chunks[i] = line

    def summarize(self):
        self.get_chunks_from_file()
        responses = [self.request(
            gpt3,
            "system",
            summarize_chunk + chunk
        ) for _, chunk in self.chunks.items()]
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
                f.write(f"CHUNK {i}:\n\n{chunk}\n\n")

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
        k_closest_chunks, _ = self.embeddings.query_chunks(
            [message], self.k)
        response = self.request(
            gpt4,
            "system",
            "SUMMARY: " + summary +
            "EXCERPTS: " + " ".join(k_closest_chunks) +
            "QUESTION: " + message
        )
        print(response)

if __name__ == '__main__':

    if not args.chunk and not args.message and not args.query and not args.summarize and not args.reset_chromadb:
        print("No arguments provided. Please use -h for help.")
        sys.exit(1)

    print(pyfiglet.figlet_format("StoryBot", font="slant"))
    print("Welcome to StoryBot! \n")
    print("This is a tool for chunking, embedding, and querying text files. \n")
    print("It uses OpenAI's GPT models for text processing. \n")
    print("To get started, remember to set up your OpenAI API key in .env. \n")
    print("Also, make sure to have configured your config.json file with the correct prompts and model names. \n")

    filename = input("Enter the path to the file to process: ")

    api = API(
        filename, chunk_size, overlap, temperature, k, gamma
    )

    print("API initialized.")
    print("File path: " + api.file_path)

    if args.message:
        message = input("Enter message: ")
        api.send_message(message)
    elif args.query:
        query = input("Enter query: ")
        api.query(query)
    elif args.summarize:
        api.summarize()
        print("Summaries written to " + api.summaries_path)
    elif args.reset_chromadb:
        api.embeddings.reset()
        print("Chromadb reset")
    elif args.chunk:
        api.chunk()
        print("Chunking complete. Written to " + api.chunk_path)