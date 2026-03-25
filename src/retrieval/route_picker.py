from langchain_classic.document_loaders import WebBaseLoader
import pandas as pd
import numpy as np

def fetch_web_context(query):
    urls = [
        "https://ibex.co/",
        "https://www.ibex.co/industries/government-services/",
        "https://www.ibex.co/industries/healthtech-wellness/",
        "https://www.ibex.co/industries/technology/",
        "https://www.ibex.co/industries/retail-ecommerce/",
        "https://waveix.ibex.co/",
        "https://waveix.ibex.co/ai-virtual-agent/",
        "https://waveix.ibex.co/translate/",
        "https://www.ibex.co/staff-augmentation/"
    ]

    all_text = ""

    for url in urls:
        loader = WebBaseLoader(url)
        docs = loader.load()
        
        for doc in docs:
            all_text += doc.page_content + "\n"

    return all_text[:3000]  