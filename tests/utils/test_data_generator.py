import fitz  # PyMuPDF
import tempfile
from pathlib import Path

def create_test_racecard_pdf():
    """Create a sample race card PDF for testing."""
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
        # Create new PDF document
        doc = fitz.open()
        page = doc.new_page()
        
        # Sample race card content
        content = """
        Race 1 - The Test Stakes
        
        Horse A (5)
        Jockey: John Smith
        Weight: 500 kg
        Odds: 4/1
        Form: 1-2-3
        
        Horse B (6)
        Jockey: Mike Jones
        Weight: 520 kg
        Odds: 5/1
        Form: 2-1-4
        
        Horse C (4)
        Jockey: Dave Brown
        Weight: 490 kg
        Odds: 3/1
        Form: 3-3-1
        """
        
        # Insert text into PDF
        page.insert_text(
            (50, 50),  # position
            content,
            fontsize=12
        )
        
        # Save PDF
        doc.save(tmp_file.name)
        doc.close()
        
        return Path(tmp_file.name)

def cleanup_test_files(file_path):
    """Clean up temporary test files."""
    try:
        Path(file_path).unlink()
    except Exception:
        pass  # Ignore errors during cleanup