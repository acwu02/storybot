from embeddings import Embeddings
from tqdm import tqdm

import fitz
import re

class FileParser():
    def __init__(self, file, chunk_size, overlap, embeddings, threshold, min_chunk_len):
        self.file = file
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks = []
        self.embeddings = embeddings
        self.threshold = threshold
        self.min_chunk_len = min_chunk_len

    def load_file(self):
        if self.file.endswith('.txt'):
            return self.load_txt()
        elif self.file.endswith('.pdf'):
            return self.load_pdf()

    def load_txt(self):
        with open(self.file, 'r') as file:
            return file.read()

    def load_pdf(self):
        doc = fitz.open(self.file)
        content = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            content += page.get_text()
        return content

    def chunk_fixed_size(self):
        assert self.chunk_size > self.overlap, "chunk size cannot be less than or equal to overlap"
        chunks = []
        text = self.load_file()
        start = 0
        while start + self.chunk_size <= len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start = end - self.overlap
        if start < len(text):
            chunks.append(text[start:])
        self.chunks = chunks
        return self.chunks

    def chunk_by_partition(self, text):
        pattern = r'\n[^a-zA-Z]+\n'
        matches = list(re.finditer(pattern, text))
        chunks = []
        chunks.append(text[:matches[0].span()[0]])
        for i in range(len(matches) - 1):
            start, end = matches[i].span()[1], matches[i + 1].span()[0]
            chunks.append(text[start:end])
        chunks.append(text[matches[-1].span()[1]:])
        return chunks

    # TODO introduce window of previous cosine sims
    # TODO merge subchunks of len 1 w/ closest neighboring subchunk
    # TODO penalize smaller chunks
    def semantic_chunking_greedy(self):
        text = self.load_file()
        chunks = self.chunk_by_partition(text)
        for c in tqdm(chunks):
            sentences = c.split('.')
            subchunks = [[] for _ in range(len(sentences))]
            curr = 0
            subchunks[0].append(sentences[0])
            for i in range(len(sentences) - 1):
                s1, s2 = sentences[i], sentences[i + 1]
                cosine_sim = self.embeddings.cosine_sim(s1, s2)
                if cosine_sim < self.threshold:
                    curr += 1
                subchunks[curr].append(s2)
            subchunks = [c for c in subchunks if len(c) > 0]
            self.chunks.extend(subchunks)
        return self.chunks

    def semantic_chunking_recursive(self):
        text = self.load_file()
        chunks = self.chunk_by_partition(text)
        for c in chunks:
            sentences = c.split('.')
            sentences.pop()
            cosine_sims = [(self.embeddings.cosine_sim(s1, s2), i)
                           for i, (s1, s2) in enumerate(zip(sentences, sentences[1:]))]
            subchunks = []
            self.semantic_chunking_recursive_helper(sentences, cosine_sims, subchunks, 0)
            self.chunks.extend(subchunks)
        return self.chunks

    def semantic_chunking_recursive_helper(self, chunk, cosine_sims, subchunks, start_idx):

        if len(chunk) <= self.min_chunk_len or not cosine_sims:
            subchunks.append(chunk)
            return

        min_cosine_sim, min_idx = min(cosine_sims)
        if min_cosine_sim < self.threshold:
            min_idx = min_idx - start_idx
            L = chunk[:min_idx]
            R = chunk[min_idx:]

            if L and R:
                self.semantic_chunking_recursive_helper(L, cosine_sims[:min_idx], subchunks, 0)
                self.semantic_chunking_recursive_helper(R, cosine_sims[min_idx:], subchunks, start_idx + min_idx + 1)
            else:
                subchunks.append(chunk)
        else:
            subchunks.append(chunk)

    # for debugging
    def cosine_sim_test(self):
        text = self.load_file()
        chunks = self.chunk_by_partition(text)
        sentences = chunks[0]
        cosine_sims_1 = []
        for i in range(len(sentences) - 1):
            s1, s2 = sentences[i], sentences[i + 1]
            cosine_sim = self.embeddings.cosine_sim(s1, s2)
            cosine_sims_1.append(cosine_sim)
        cosine_sims_2 = [self.embeddings.cosine_sim(s1, s2)
                    for (s1, s2) in zip(sentences, sentences[1:])]
        print(cosine_sims_1)
        print(cosine_sims_2)
        assert(cosine_sims_1 == cosine_sims_2), "error: cosine sims are not the same"

    def print_chunks(self):
        for i, chunk in enumerate(self.chunks):
            print("CHUNK" + str(i) + ":")
            for i in range(50):
                print("-", end='')
            print('\n')
            print(chunk)

# Example usage
if __name__ == '__main__':
    embeddings = Embeddings("PART_1_long", "PART_1_long_embed", 0.1)
    fileparser = FileParser("PART_1_long.pdf", 4000, 400, embeddings, 0.8, 50)

    chunks = fileparser.chunk_fixed_size()

    print(chunks)

    # Write chunks to file
    with open("chunks.txt", "w") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"CHUNK {i}:\n")
            f.write(''.join(chunk))
            f.write("\n\n")


