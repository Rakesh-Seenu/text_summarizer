import pandas as pd
from PyPDF2 import PdfReader
from docx import Document

class DocumentProcessor:
    """Handles extracting text from different document types."""
    
    def __init__(self):
        self.documents = {}

    def process_file(self, file):
        file_type = file.name.split(".")[-1].lower()

        if file_type == "pdf":
            return self._process_pdf(file)
        elif file_type == "docx":
            return self._process_docx(file)
        elif file_type == "csv":
            return self._process_csv(file)
        else:
            return "Unsupported file format."

    def _process_pdf(self, file):
        reader = PdfReader(file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

    def _process_docx(self, file):
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs])

    def _process_csv(self, file):
        df = pd.read_csv(file)
        return df.to_string()
