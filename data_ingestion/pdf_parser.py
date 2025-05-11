# pdf_parser.py

import fitz  # PyMuPDF
import re
from utils.logger import get_logger
import pdfplumber
from typing import Dict, Any

logger = get_logger(__name__)

class RaceCardParser:
    def __init__(self):
        self.patterns = {
            'horse_name': r'([A-Z][A-Za-z\s\-\']+)\s+\(\d+\)',
            'age': r'\((\d+)\)',
            'weight': r'(\d+\.\d+|\d+)\s*(?:kg|KG)',
            'odds': r'(\d+\/\d+|\d+\.\d+)',
            'form': r'Form:?\s*([0-9\-P]+)',
            'jockey': r'Jockey:\s*([A-Za-z\s\-\']+)'
        }
    
    def _extract_pattern(self, text, pattern, group=1):
        """Extract data using regex pattern."""
        match = re.search(pattern, text)
        return match.group(group) if match else None

    def _parse_odds(self, odds_str):
        """Convert fractional or decimal odds to decimal format."""
        try:
            if '/' in odds_str:
                num, den = map(int, odds_str.split('/'))
                return round(num/den + 1, 2)
            return float(odds_str)
        except (ValueError, ZeroDivisionError):
            return None

    def _parse_form(self, form_str):
        """Clean and standardize form string."""
        if not form_str:
            return '0-0-0'
        # Replace non-numeric characters except hyphen
        clean_form = re.sub(r'[^0-9\-]', '0', form_str)
        # Ensure we have exactly 3 results
        parts = clean_form.split('-')[:3]
        while len(parts) < 3:
            parts.append('0')
        return '-'.join(parts)

    def parse_racecard(self, pdf_path):
        """
        Parse race card PDF into structured data.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing structured race data
        """
        try:
            logger.info(f"Starting to parse PDF: {pdf_path}")
            doc = fitz.open(pdf_path)
            
            # Extract text from all pages
            full_text = ""
            for page in doc:
                full_text += page.get_text()
            
            # Split text into sections for each horse
            horse_sections = re.split(r'\n{2,}', full_text)
            
            horses = []
            for section in horse_sections:
                try:
                    # Extract horse details
                    horse_name = self._extract_pattern(section, self.patterns['horse_name'])
                    if not horse_name:
                        continue  # Skip if no horse name found
                        
                    horse_data = {
                        'horse': horse_name,
                        'age': self._extract_pattern(section, self.patterns['age']),
                        'weight': self._extract_pattern(section, self.patterns['weight']),
                        'odds_string': self._extract_pattern(section, self.patterns['odds']),
                        'form': self._extract_pattern(section, self.patterns['form']),
                        'jockey': self._extract_pattern(section, self.patterns['jockey'])
                    }
                    
                    # Convert types and clean data
                    if horse_data['age']:
                        horse_data['age'] = int(horse_data['age'])
                    
                    if horse_data['weight']:
                        horse_data['weight'] = float(horse_data['weight'])
                    
                    if horse_data['odds_string']:
                        horse_data['odds'] = self._parse_odds(horse_data['odds_string'])
                        del horse_data['odds_string']
                    
                    if horse_data['form']:
                        horse_data['form'] = self._parse_form(horse_data['form'])
                    
                    # Validate required fields
                    if horse_data['horse'] and horse_data.get('odds'):
                        horses.append(horse_data)
                    else:
                        logger.warning(f"Skipping horse due to missing required fields: {horse_data}")
                
                except Exception as e:
                    logger.error(f"Error parsing horse section: {str(e)}")
                    continue
            
            if not horses:
                raise ValueError("No valid horse data found in PDF")
            
            logger.info(f"Successfully parsed data for {len(horses)} horses")
            return {'horses': horses}
            
        except Exception as e:
            logger.error(f"Error parsing PDF: {str(e)}")
            raise
        finally:
            if 'doc' in locals():
                doc.close()

def parse_racecard(pdf_path: str) -> Dict[str, Any]:
    """Parse race card PDF into structured data.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Dictionary containing both raw text and structured data
    """
    try:
        logger.info(f"Starting to parse PDF: {pdf_path}")
        
        # Read PDF content using pdfplumber for better text extraction
        with pdfplumber.open(pdf_path) as pdf:
            raw_text = ""
            horses = []
            current_horse = {}
            
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    raw_text += text + "\n"
                    
                    # Process lines
                    for line in text.split('\n'):
                        line = line.strip()
                        if not line:
                            if current_horse:
                                horses.append(current_horse)
                                current_horse = {}
                            continue
                            
                        if ':' in line:
                            key, value = line.split(':', 1)
                            key = key.strip().lower()
                            value = value.strip()
                            
                            if key == 'horse' or 'name' in key.lower():
                                if current_horse:
                                    horses.append(current_horse)
                                current_horse = {'horse': value}
                            else:
                                if not current_horse:
                                    current_horse = {}
                                current_horse[key] = value
                                
            # Add last horse if any
            if current_horse:
                horses.append(current_horse)
                
        logger.info(f"Successfully parsed data for {len(horses)} horses")
        
        # Clean horse data entries
        cleaned_horses = []
        for horse in horses:
            try:
                cleaned = clean_horse_data(horse)
                # Ensure horse name is preserved
                if 'horse' not in cleaned and 'horse' in horse:
                    cleaned['horse'] = horse['horse']
                cleaned_horses.append(cleaned)
            except ValueError as e:
                logger.warning(f"Skipping invalid horse data: {str(e)}")
                
        return {
            'raw_text': raw_text,  # For test compatibility
            'horses': cleaned_horses  # For structured processing
        }
        
    except Exception as e:
        logger.error(f"Error parsing PDF: {str(e)}")
        raise
