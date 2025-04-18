import os
import uuid
from typing import List, Tuple


def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    """
    Split text into chunks of specified size
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0

    for word in words:
        if current_size + len(word) + 1 > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_size = len(word)
        else:
            current_chunk.append(word)
            current_size += len(word) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def process_file(file_content: str) -> Tuple[str, List[Tuple[str, int]]]:
    """
    Process file content and return document_id and chunks with their IDs
    """
    # Generate unique document ID
    document_id = str(uuid.uuid4())

    # Split content into chunks
    chunks = chunk_text(file_content)

    # Create list of (chunk, chunk_id) tuples
    chunk_data = [(chunk, i) for i, chunk in enumerate(chunks)]

    return document_id, chunk_data



