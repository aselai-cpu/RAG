"""
Document Loaders - Infrastructure layer for loading documents from various sources.

This module provides loaders for different document types (PDF, text, clipboard).
"""
from typing import Optional
import PyPDF2
from io import BytesIO
from src.domain.entities.document import Document


class DocumentLoader:
    """
    Service for loading documents from various sources.

    This class handles the complexity of extracting text from different
    file formats and creating domain Document entities.
    """

    @staticmethod
    def load_from_pdf(file_bytes: bytes, file_name: str) -> Document:
        """
        Load a document from PDF bytes.

        Args:
            file_bytes: The PDF file content as bytes
            file_name: Name of the PDF file

        Returns:
            Document entity containing the extracted text

        Raises:
            ValueError: If PDF cannot be read or is empty
        """
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_bytes))

            # Extract text from all pages
            text_content = []
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():
                    text_content.append(text)

            if not text_content:
                raise ValueError("PDF contains no readable text")

            content = "\n\n".join(text_content)

            return Document(
                content=content,
                source_type="pdf",
                file_name=file_name,
                metadata={
                    "page_count": len(pdf_reader.pages),
                    "extracted_pages": len(text_content),
                },
            )

        except Exception as e:
            raise ValueError(f"Failed to read PDF: {str(e)}")

    @staticmethod
    def load_from_text(file_bytes: bytes, file_name: str) -> Document:
        """
        Load a document from a text file.

        Args:
            file_bytes: The text file content as bytes
            file_name: Name of the text file

        Returns:
            Document entity containing the text

        Raises:
            ValueError: If text file cannot be read or is empty
        """
        try:
            # Try UTF-8 first, fall back to latin-1
            try:
                content = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                content = file_bytes.decode("latin-1")

            if not content.strip():
                raise ValueError("Text file is empty")

            return Document(
                content=content,
                source_type="text",
                file_name=file_name,
                metadata={"encoding": "utf-8"},
            )

        except Exception as e:
            raise ValueError(f"Failed to read text file: {str(e)}")

    @staticmethod
    def load_from_clipboard(text: str) -> Document:
        """
        Load a document from clipboard text.

        Args:
            text: The text content from clipboard

        Returns:
            Document entity containing the text

        Raises:
            ValueError: If text is empty
        """
        if not text.strip():
            raise ValueError("Clipboard text is empty")

        return Document(
            content=text,
            source_type="clipboard",
            file_name=None,
            metadata={},
        )


class DocumentValidator:
    """
    Service for validating documents before processing.
    """

    @staticmethod
    def validate_file_size(file_bytes: bytes, max_size_mb: int = 10) -> None:
        """
        Validate that a file is not too large.

        Args:
            file_bytes: The file content as bytes
            max_size_mb: Maximum file size in megabytes

        Raises:
            ValueError: If file is too large
        """
        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > max_size_mb:
            raise ValueError(
                f"File too large: {size_mb:.2f}MB (max: {max_size_mb}MB)"
            )

    @staticmethod
    def validate_file_type(file_name: str, allowed_extensions: list) -> None:
        """
        Validate that a file has an allowed extension.

        Args:
            file_name: Name of the file
            allowed_extensions: List of allowed extensions (e.g., ['.pdf', '.txt'])

        Raises:
            ValueError: If file type is not allowed
        """
        extension = "." + file_name.split(".")[-1].lower() if "." in file_name else ""
        if extension not in allowed_extensions:
            raise ValueError(
                f"File type not allowed: {extension}. Allowed: {', '.join(allowed_extensions)}"
            )
