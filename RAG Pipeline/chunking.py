import fitz  # PyMuPDF
import pdfplumber
import re
import json
from uuid import uuid4
from typing import List, Dict, Any
import os


# ---------- SECTION/TEXT PARSER (PYMUPDF) ----------

def is_section_heading(text: str) -> bool:
    """
    Determine if a given text block is a section heading.

    Args:
        text (str): Text block to check.

    Returns:
        bool: True if text matches typical section heading pattern (e.g. "1.2 SectionTitle").
    """
    return bool(re.match(r'^\d+(\.\d+)*\s+[A-Z][a-z]+', text.strip()))

def is_figure_caption(text: str) -> bool:
    """
    Check if a text block is a figure caption to exclude it.

    Args:
        text (str): Text block to check.

    Returns:
        bool: True if text looks like a figure caption (e.g. "Figure 1.2: ...").
    """
    return bool(re.match(r'^\s*Figure\s*\d+(\.\d+)*:?', text.strip(), re.IGNORECASE))

def extract_text_chunks(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text chunks from PDF pages, grouping by sections.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        List of dictionaries representing text chunks.
            Each dict contains:
                - id: unique UUID string for the chunk
                - type: 'section' or 'text' (indicating if it's a section heading or regular text)
                - page: page number 
                - section_title: for sections only 
                - content: text content of the chunk 
    """
    doc = fitz.open(pdf_path)
    chunks = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("blocks")
        current_section = None

        for b in sorted(blocks, key=lambda x: x[1]):  # top-to-bottom
            x0, y0, x1, y1, text, *_ = b
            text = text.strip()
            if not text or is_figure_caption(text):
                continue  # Skip empty or figure caption blocks

            if is_section_heading(text):
                current_section = {
                    "id": str(uuid4()),
                    "type": "section",
                    "page": page_num + 1,
                    "section_title": text,
                    "content": "",
                    "y0": y0
                }
                chunks.append(current_section)
            else:
                if current_section:
                    current_section["content"] += "\n" + text
                else:
                    chunks.append({
                        "id": str(uuid4()),
                        "type": "text",
                        "page": page_num + 1,
                        "content": text,
                        "y0": y0
                    })

    return chunks


# ---------- TABLE PARSER (PDFPLUMBER + FITZ for titles) ----------

def find_table_title(pdf_path: str, page_number: int, title_regex=r'^Table\s*\d+(\.\d+)*:?.*') -> str:
    """
    Attempt to find a table title block above a table by scanning page text blocks.

    Args:
        pdf_path (str): Path to the PDF file.
        page_number (int): Zero-based page index.
        title_regex (str): Regex pattern for matching table titles.

    Returns:
        str: The table title text if found; otherwise default string.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_number]
    blocks = page.get_text("blocks")

    # Sort blocks top to bottom, right above the table
    sorted_blocks = sorted(blocks, key=lambda x: x[1])
    for b in sorted_blocks:
        x0, y0, x1, y1, text, *_ = b
        if re.match(title_regex, text.strip(), re.IGNORECASE):
            return text.strip()
    return f"Table on page {page_number + 1}"


def extract_all_tables(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract tables from the entire PDF using pdfplumber, associate titles via PyMuPDF.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        List[Dict[str, Any]]: List of dicts representing table chunks with keys:
            - id: unique UUID
            - type: 'table'
            - page: page number (1-indexed)
            - table_title: string
            - columns: list of column headers
            - rows: list of table rows (each a list of strings)
            - y0: approximate vertical position (fixed as 9999 here)
    """
    extracted_tables = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            for t_index, table in enumerate(tables):
                if not table or len(table) < 2:
                    continue  

                header = table[0]
                rows = table[1:]
                table_title = find_table_title(pdf_path, i)

                extracted_tables.append({
                    "id": str(uuid4()),
                    "type": "table",
                    "page": i + 1,
                    "table_title": table_title,
                    "columns": header,
                    "rows": rows,
                    "y0": 9999 # Placeholder for vertical position ( had issues with y0 in tables)
                })

    return extracted_tables


# ---------- COMBINE & SAVE ----------

def combine_chunks(text_chunks: List[Dict], table_chunks: List[Dict]) -> List[Dict]:
    """
    Combine text and table chunks and sort by page and vertical position.

    Args:
        text_chunks (List[Dict]): List of text section chunks.
        table_chunks (List[Dict]): List of table chunks.

    Returns:
        List[Dict]: Combined and sorted list of chunks.
    """
    all_chunks = text_chunks + table_chunks
    return sorted(all_chunks, key=lambda x: (x["page"], x.get("y0", 0)))


def save_json(data: List[Dict], output_path: str):
    """
    Save a list of chunk dictionaries as pretty-printed JSON.

    Args:
        data (List[Dict]): Data to save.
        output_path (str): File path to save JSON to.
    """
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved {len(data)} chunks to {output_path}")


# ---------- MAIN ----------

if __name__ == "__main__":
    pdf_path = "handbook.pdf"  # Replace with your actual file
    output_combined = "combined_chunks.json"
    output_text = "sections.json"
    output_tables = "tables.json"

    if os.path.exists(pdf_path):
        text_chunks = extract_text_chunks(pdf_path)
        table_chunks = extract_all_tables(pdf_path)
        combined_chunks = combine_chunks(text_chunks, table_chunks)

        save_json(text_chunks, output_text)
        save_json(table_chunks, output_tables)
        save_json(combined_chunks, output_combined)
    else:
        print(f"❌ File not found: {pdf_path}")
