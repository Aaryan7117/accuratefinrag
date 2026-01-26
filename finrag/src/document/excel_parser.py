"""
Excel Parser for Financial Documents
=====================================

This module handles Excel file parsing for financial reports and datasets.
Supports .xlsx and .xls formats with structure preservation.

Design Decisions:
- Use pandas for robust Excel reading
- Treat each worksheet as a document section
- Convert dataframes to table elements
- Preserve sheet names and cell references
"""

import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

import pandas as pd


class ElementType(Enum):
    """Types of document elements."""
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    LIST = "list"
    UNKNOWN = "unknown"


@dataclass
class DocumentElement:
    """
    A single element from the parsed document.

    Every element tracks its source location for citation purposes.
    """
    element_type: ElementType
    content: str
    page_number: int  # For Excel, this will be sheet index
    bbox: Optional[Tuple[float, float, float, float]] = None  # Not applicable for Excel
    heading_level: Optional[int] = None
    table_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Section:
    """
    A document section containing multiple elements.

    For Excel, each worksheet becomes a section.
    """
    section_id: str
    title: str
    level: int = 1  # All sheets are top-level
    item_number: Optional[str] = None
    page_start: int = 0
    page_end: int = 0
    elements: List[DocumentElement] = field(default_factory=list)
    subsections: List["Section"] = field(default_factory=list)


@dataclass
class ParsedDocument:
    """
    Complete parsed document with structure.

    Contains all data from Excel sheets.
    """
    doc_id: str
    filename: str
    title: Optional[str] = None
    total_pages: int = 0  # Number of sheets
    sections: List[Section] = field(default_factory=list)
    elements: List[DocumentElement] = field(default_factory=list)
    tables: List[DocumentElement] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_all_text(self) -> str:
        """Get all text content concatenated."""
        return "\n\n".join(
            el.content for el in self.elements
            if el.element_type != ElementType.TABLE
        )


class ExcelParser:
    """
    Excel parser for financial documents.

    Extraction Strategy:
    1. Read all worksheets
    2. Convert each sheet to a table element
    3. Create sections for each sheet
    4. Preserve sheet names and structure
    """

    def __init__(self):
        """Initialize parser."""
        pass

    def parse(self, excel_path: str | Path) -> ParsedDocument:
        """
        Parse an Excel document.

        Args:
            excel_path: Path to the Excel file

        Returns:
            ParsedDocument with all extracted content
        """
        excel_path = Path(excel_path)

        # Generate stable document ID from file hash
        doc_id = self._generate_doc_id(excel_path)

        # Initialize document
        doc = ParsedDocument(
            doc_id=doc_id,
            filename=excel_path.name,
            metadata={
                "source_path": str(excel_path),
                "file_size": excel_path.stat().st_size,
            }
        )

        # Read Excel file
        try:
            excel_data = pd.read_excel(excel_path, sheet_name=None)
        except Exception as e:
            raise ValueError(f"Failed to read Excel file: {e}")

        doc.total_pages = len(excel_data)

        # Process each sheet
        for sheet_idx, (sheet_name, df) in enumerate(excel_data.items()):
            # Create section for this sheet
            section = Section(
                section_id=f"sheet_{sheet_idx}",
                title=sheet_name,
                level=1,
                page_start=sheet_idx,
                page_end=sheet_idx
            )

            # Convert dataframe to table element
            table_element = self._dataframe_to_element(df, sheet_name, sheet_idx)
            section.elements.append(table_element)
            doc.elements.append(table_element)
            doc.tables.append(table_element)

            doc.sections.append(section)

        # Set title to first sheet name
        if doc.sections:
            doc.title = doc.sections[0].title

        return doc

    def _generate_doc_id(self, excel_path: Path) -> str:
        """
        Generate stable document ID from file content hash.
        """
        hasher = hashlib.sha256()
        with open(excel_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:16]

    def _dataframe_to_element(self, df: pd.DataFrame, sheet_name: str, sheet_idx: int) -> DocumentElement:
        """
        Convert pandas DataFrame to DocumentElement.
        """
        # Convert to markdown table format for content
        content = df.to_markdown(index=False)

        # Create table ID
        table_id = f"{sheet_name}_{sheet_idx}"

        return DocumentElement(
            element_type=ElementType.TABLE,
            content=content,
            page_number=sheet_idx,
            table_id=table_id,
            metadata={
                "sheet_name": sheet_name,
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
            }
        )


# Convenience function for simple usage
def parse_excel(excel_path: str | Path) -> ParsedDocument:
    """Parse an Excel file and return structured document."""
    parser = ExcelParser()
    return parser.parse(excel_path)