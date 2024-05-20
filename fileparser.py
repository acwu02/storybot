import PyPDF2

class FileParser():
    def __init__(self, file, chunk_size, overlap):
        self.file = file
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks = []

    def load_file(self):
        all_text = ""
        with open(self.file, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = []
            for page in reader.pages:
                text.append(page.extract_text())
            all_text = ''.join(text)

        all_text = all_text.replace('\n', ' ')
        return all_text

    def parse_file(self):
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

    def print_chunks(self):
        for i, chunk in enumerate(self.chunks):
            print("CHUNK" + str(i) + ":")
            for i in range(50):
                print("-", end='')
            print('\n')
            print(chunk)

