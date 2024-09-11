"""
!!!
RUN THESE TESTS FROM THE PARENT DIRECTORY
!!!
"""

import unittest
import os
import math
from fpdf import FPDF


from fileparser import FileParser

class FileParserTest(unittest.TestCase):

    def setUp(self):

        # generate string content of PDF
        def generate_string():
            with open("string.txt", 'r') as file:
                string = file.read()
                return string

        # generate PDF file
        def generate_pdf(string):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size = 12)
            pdf.cell(200, 10, txt = string, ln = True, align = 'C')
            pdf.output("test.pdf")

        # create pdf
        self.pdf_string = generate_string()
        generate_pdf(self.pdf_string)

    def tearDown(self):
        # delete PDF
        assert os.path.exists("test.pdf")
        os.remove("test.pdf")
        assert self.pdf_string
        self.pdf_string = None

    # test loading of PDF content
    def test_load(self):
        fileparser = FileParser("test.pdf", 0, 0)
        pdf_text = fileparser.load_file()
        self.assertEqual(pdf_text, self.pdf_string)

    # test parsing of PDF w/ overlap 0
    def test_parse(self):
        fileparser = FileParser("test.pdf", 100, 0)
        fileparser.parse_file()
        num_chunks = len(fileparser.chunks)
        self.assertEqual(num_chunks, math.ceil(len(self.pdf_string) / 100))

    # test parsing of PDF w/ overlap
    def test_parse_overlap(self):
        fileparser = FileParser("test.pdf", 100, 10)
        fileparser.parse_file()
        num_chunks = len(fileparser.chunks)
        self.assertEqual(num_chunks, 32)
        for chunk1, chunk2 in zip(fileparser.chunks, fileparser.chunks[1:]):
            end1 = chunk1[fileparser.chunk_size - fileparser.overlap:]
            start2 = chunk2[:fileparser.overlap]
            self.assertEqual(end1, start2)

    # test parsing of PDF w/ greater chunk size than doc length
    def test_parse_chunk_overflow(self):
        fileparser = FileParser("test.pdf", len(self.pdf_string) + 100, 0)
        fileparser.parse_file()
        num_chunks = len(fileparser.chunks)
        self.assertEqual(num_chunks, 1)

    # test parsing of PDF where length of string is equal to chunk size
    def test_parsing_string_equal(self):
        fileparser = FileParser("test.pdf", 10, 1)
        fileparser.parse_file()
        # TODO

    # test on Beijing_Butterflies.pdf
    # def test_bb(self):
    #     fileparser = FileParser("Beijing_Butterflies.pdf", 500, 50)
    #     fileparser.parse_file()

if __name__ == '__main__':
    unittest.main()




