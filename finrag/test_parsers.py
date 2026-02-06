#!/usr/bin/env python3
"""
Document Parser Testing Script
==============================

This script tests the document parsing functionality for PDF, Excel, and PowerPoint files.
Run this script to verify that all parsers are working correctly.

Usage:
    python test_parsers.py

Requirements:
    - All dependencies from requirements.txt must be installed
    - Test files should be placed in the data/uploads/ directory
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from document.parser import parse_document, ParsedDocument
    from document.parser import PDFParser, ExcelParser, PowerPointParser
    print("✓ Successfully imported all parser modules")
    PARSERS_AVAILABLE = True
except ImportError as e:
    print(f"✗ Failed to import parser modules: {e}")
    print("Make sure you're running this from the project root and all dependencies are installed.")
    print("Run: pip install -r requirements.txt")
    PARSERS_AVAILABLE = False
    # Try to import just the data structures for demo
    try:
        from document.parser import ParsedDocument
        DATA_STRUCTURES_AVAILABLE = True
    except ImportError:
        DATA_STRUCTURES_AVAILABLE = False


def test_parser_imports():
    """Test that all parser classes can be instantiated."""
    print("\n=== Testing Parser Imports ===")

    if not PARSERS_AVAILABLE:
        print("⚠ Skipping parser instantiation tests (dependencies not installed)")
        return

    try:
        pdf_parser = PDFParser()
        print("✓ PDFParser instantiated successfully")
    except Exception as e:
        print(f"✗ PDFParser failed: {e}")

    try:
        excel_parser = ExcelParser()
        print("✓ ExcelParser instantiated successfully")
    except Exception as e:
        print(f"✗ ExcelParser failed: {e}")

    try:
        ppt_parser = PowerPointParser()
        print("✓ PowerPointParser instantiated successfully")
    except Exception as e:
        print(f"✗ PowerPointParser failed: {e}")


def test_file_type_detection():
    """Test file type detection logic."""
    print("\n=== Testing File Type Detection ===")

    if not PARSERS_AVAILABLE:
        print("⚠ Skipping file type detection tests (dependencies not installed)")
        return

    test_cases = [
        ("document.pdf", "pdf"),
        ("spreadsheet.xlsx", "excel"),
        ("spreadsheet.xls", "excel"),
        ("presentation.pptx", "powerpoint"),
        ("presentation.ppt", "powerpoint"),
        ("unknown.txt", "unsupported")
    ]

    for filename, expected in test_cases:
        try:
            # This will test the parse_document function's file type detection
            # We expect it to fail with unsupported types, succeed with supported ones
            if expected == "unsupported":
                try:
                    parse_document(filename)
                    print(f"✗ {filename}: Should have failed for unsupported type")
                except ValueError as e:
                    print(f"✓ {filename}: Correctly rejected unsupported type")
                except Exception as e:
                    print(f"? {filename}: Unexpected error: {e}")
            else:
                print(f"? {filename}: Would test parsing (file not present)")
        except Exception as e:
            print(f"✗ {filename}: Unexpected error: {e}")


def find_test_files() -> List[Path]:
    """Find test files in the data/uploads directory."""
    uploads_dir = Path("data/uploads")
    if not uploads_dir.exists():
        print(f"Uploads directory not found: {uploads_dir}")
        return []

    supported_extensions = ['.pdf', '.xlsx', '.xls', '.pptx', '.ppt']
    test_files = []

    for ext in supported_extensions:
        files = list(uploads_dir.glob(f"*{ext}"))
        test_files.extend(files)

    return test_files


def test_document_parsing():
    """Test parsing actual documents if they exist."""
    print("\n=== Testing Document Parsing ===")

    if not PARSERS_AVAILABLE:
        print("⚠ Skipping document parsing tests (dependencies not installed)")
        print("\nDemonstrating expected behavior with mock data...")
        demo_parsed_doc()
        return

    test_files = find_test_files()

    if not test_files:
        print("No test files found in data/uploads/")
        print("To test with real files, place PDF, Excel, or PowerPoint files in data/uploads/")
        print("\nDemonstrating expected behavior with mock data...")

        # Show what the parsing structure looks like
        demo_parsed_doc()
        return

    print(f"Found {len(test_files)} test file(s):")

    for file_path in test_files:
        print(f"\n--- Testing {file_path.name} ---")
        try:
            doc = parse_document(file_path)
            print(f"✓ Successfully parsed: {doc.filename}")
            print(f"  - Document ID: {doc.doc_id}")
            print(f"  - Title: {doc.title or 'No title'}")
            print(f"  - Total pages/sections: {doc.total_pages}")
            print(f"  - Sections: {len(doc.sections)}")
            print(f"  - Elements: {len(doc.elements)}")
            print(f"  - Tables: {len(doc.tables)}")

            # Show first section if available
            if doc.sections:
                first_section = doc.sections[0]
                print(f"  - First section: '{first_section.title}' ({len(first_section.elements)} elements)")

            # Show sample content
            if doc.elements:
                first_element = doc.elements[0]
                content_preview = first_element.content[:100] + "..." if len(first_element.content) > 100 else first_element.content
                print(f"  - Sample content: {content_preview}")

        except Exception as e:
            print(f"✗ Failed to parse {file_path.name}: {e}")


def demo_parsed_doc():
    """Demonstrate the structure of a parsed document."""
    print("\n--- Demonstrating ParsedDocument Structure ---")

    if not DATA_STRUCTURES_AVAILABLE:
        print("⚠ Cannot demonstrate data structures (imports failed)")
        print("Install dependencies and run again to see the demo.")
        return

    # Create a mock parsed document to show the structure
    mock_doc = ParsedDocument(
        doc_id="demo_12345678",
        filename="demo.pdf",
        title="Sample Financial Report",
        total_pages=5,
        sections=[],
        elements=[],
        tables=[],
        metadata={"source_path": "/path/to/demo.pdf", "file_size": 1024000}
    )

    print("ParsedDocument structure:")
    print(f"  - doc_id: {mock_doc.doc_id}")
    print(f"  - filename: {mock_doc.filename}")
    print(f"  - title: {mock_doc.title}")
    print(f"  - total_pages: {mock_doc.total_pages}")
    print(f"  - sections: {len(mock_doc.sections)} (list of Section objects)")
    print(f"  - elements: {len(mock_doc.elements)} (list of DocumentElement objects)")
    print(f"  - tables: {len(mock_doc.tables)} (list of table DocumentElement objects)")
    print(f"  - metadata: {mock_doc.metadata}")

    print("\nEach Section contains:")
    print("  - section_id: unique identifier")
    print("  - title: section title")
    print("  - level: hierarchy level")
    print("  - elements: list of DocumentElement objects")

    print("\nEach DocumentElement contains:")
    print("  - element_type: HEADING, PARAGRAPH, TABLE, etc.")
    print("  - content: the actual text or table content")
    print("  - page_number: source page/slide/sheet number")
    print("  - metadata: additional information")


def test_error_handling():
    """Test error handling for invalid inputs."""
    print("\n=== Testing Error Handling ===")

    if not PARSERS_AVAILABLE:
        print("⚠ Skipping error handling tests (dependencies not installed)")
        return

    # Test with non-existent file
    try:
        parse_document("nonexistent.pdf")
        print("✗ Should have failed for non-existent file")
    except FileNotFoundError:
        print("✓ Correctly handled non-existent file")
    except Exception as e:
        print(f"? Unexpected error for non-existent file: {e}")

    # Test with unsupported file type
    try:
        parse_document("test.txt")
        print("✗ Should have failed for unsupported file type")
    except ValueError as e:
        print("✓ Correctly rejected unsupported file type")
    except Exception as e:
        print(f"? Unexpected error for unsupported type: {e}")


def main():
    """Run all tests."""
    print("Document Parser Testing Script")
    print("=" * 40)

    # Check if we're in the right directory
    if not Path("src/document/parser.py").exists():
        print("✗ Error: Please run this script from the project root directory")
        print("   Current directory should contain 'src/' folder")
        sys.exit(1)

    test_parser_imports()
    test_file_type_detection()
    test_document_parsing()
    test_error_handling()

    print("\n" + "=" * 40)
    print("Testing complete!")
    print("\nTo test with real files:")
    print("1. Place PDF (.pdf), Excel (.xlsx/.xls), or PowerPoint (.pptx/.ppt) files in data/uploads/")
    print("2. Run this script again")
    print("\nTo install dependencies:")
    print("   pip install -r requirements.txt")


if __name__ == "__main__":
    main()