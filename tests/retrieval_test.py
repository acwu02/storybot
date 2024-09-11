"""
!!!
RUN THESE TESTS FROM THE PARENT DIRECTORY
!!!
"""

from embeddings import Embeddings
from fileparser import FileParser

import unittest

STORY_NAME = "PART_1_long"
EMBED_PATH = f"{STORY_NAME}_embeddings_test",
CHUNK_SIZE = 4000
OVERLAP_SIZE = 400
GAMMA = 0.1
K = 5

QUERIES = dict()
QUERIES['jd'] = []
QUERIES['elliott'] = []
QUERIES['lexi'] = []

class EmbeddingsTest(unittest.TestCase):

    def setUp(self):
        self.fileparser = FileParser(f"{STORY_NAME}.pdf", CHUNK_SIZE, OVERLAP_SIZE)
        self.embeddings = Embeddings(f"{STORY_NAME}.pdf", EMBED_PATH[0], GAMMA)
        self.chunks = self.fileparser.parse_file()
        self.embeddings.embed_chunks(self.chunks)

        # Find number of relevant docs for evaluating recall
        def find_relevant_docs(self):
            for query in QUERIES.keys():
                for chunk in self.chunks:
                    if chunk.find(query) != -1:
                        QUERIES[query].append(chunk)
            print(QUERIES)

        find_relevant_docs()

    def test1(self):
        for query in QUERIES.keys():
            chunks, scores = self.embeddings.query_chunks(query, K)
            return

if __name__ == '__main__':
    unittest.main()