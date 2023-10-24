import streamlit as st
from zipfile import ZipFile
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
import base64

# ------- OCR ------------
import pdf2image
import pytesseract


@st.cache_data
def images_to_txt(path, language):
    images = pdf2image.convert_from_bytes(path)
    all_text = []
    for i in images:
        pil_im = i
        text = pytesseract.image_to_string(pil_im, lang=language)
        all_text.append(text)
    return all_text, len(all_text)


@st.cache_data
def convert_pdf_to_txt_pages(path):
    texts = []
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    # fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    size = 0
    c = 0
    file_pages = PDFPage.get_pages(path)
    num_pages = len(list(file_pages))
    for page in PDFPage.get_pages(path):
        interpreter.process_page(page)
        t = retstr.getvalue()
        if c == 0:
            texts.append(t)
        else:
            texts.append(t[size:])
        c = c + 1
        size = len(t)
    # text = retstr.getvalue()

    # fp.close()
    device.close()
    retstr.close()
    return texts, num_pages


@st.cache_data
def convert_pdf_to_txt_file(pdf_file):
    """Adapted from https://discuss.streamlit.io/t/text-data-extractor-pdf-to-text/25210."""
    rsrcmgr = PDFResourceManager()
    return_string = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, return_string, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    file_pages = PDFPage.get_pages(pdf_file)
    num_pages = len(list(file_pages))
    for page in PDFPage.get_pages(pdf_file):
        text = return_string.getvalue()
        interpreter.process_page(page)

    device.close()
    return_string.close()
    return text, num_pages


@st.cache_data
def save_pages(pages):
    files = []
    for page in range(len(pages)):
        filename = "page_" + str(page) + ".txt"
        with open("./file_pages/" + filename, "w", encoding="utf-8") as file:
            file.write(pages[page])
            files.append(file.name)

    # create zipfile object
    zip_path = "./file_pages/pdf_to_txt.zip"
    zip_obj = ZipFile(zip_path, "w")
    for f in files:
        zip_obj.write(f)
    zip_obj.close()

    return zip_path


def display_pdf(file):
    """Displays a PDF in Streamlit."""
    base64_pdf = base64.b64encode(file).decode("utf-8")

    # Display the PDF in an iframe.
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000"
        type="application/pdf"></iframe>"""
    st.markdown(pdf_display, unsafe_allow_html=True)
