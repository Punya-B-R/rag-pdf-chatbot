def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text with semantic awareness"""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", ".", " ", ""]
    )
    return splitter.split_text(text)