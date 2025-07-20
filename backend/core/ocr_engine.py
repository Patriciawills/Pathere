"""
OCR Engine for processing dictionary and grammar book PDFs
Uses open-source OCR solutions with specialized text processing
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import re
import json
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class OCREngine:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)  # Limit for memory efficiency
        
        # OCR configuration for different scripts
        self.ocr_configs = {
            'english': '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?()-\'" ',
            'hindi': '--oem 3 --psm 6 -l eng+hin',
            'sanskrit': '--oem 3 --psm 6 -l eng+san'
        }
        
        # Text preprocessing patterns
        self.patterns = {
            'dictionary_entry': re.compile(r'^([a-zA-Z]+)\s*(?:\[([^\]]+)\])?\s*(?:\(([^)]+)\))?\s*(.+)$'),
            'grammar_rule': re.compile(r'^([A-Z][^:]+):\s*(.+)$'),
            'example_sentence': re.compile(r'^\d+\.\s*(.+)$'),
            'phonetic': re.compile(r'\[([^\]]+)\]'),
            'pos_tag': re.compile(r'\(([^)]+)\)')
        }
    
    async def process_pdf(self, pdf_path: str, page_number: Optional[int] = None, processing_type: str = "dictionary") -> Dict[str, Any]:
        """
        Process PDF and extract structured text using OCR
        """
        try:
            loop = asyncio.get_event_loop()
            
            if page_number is not None:
                # Process single page
                result = await loop.run_in_executor(
                    self.executor, 
                    self._process_single_page, 
                    pdf_path, page_number, processing_type
                )
            else:
                # Process entire PDF (in chunks to manage memory)
                result = await loop.run_in_executor(
                    self.executor, 
                    self._process_entire_pdf, 
                    pdf_path, processing_type
                )
            
            return result
            
        except Exception as e:
            logger.error(f"OCR processing error: {str(e)}")
            raise Exception(f"OCR processing failed: {str(e)}")
    
    def _process_single_page(self, pdf_path: str, page_number: int, processing_type: str) -> Dict[str, Any]:
        """Process a single PDF page"""
        try:
            # Open PDF
            pdf_document = fitz.open(pdf_path)
            
            if page_number >= len(pdf_document):
                raise ValueError(f"Page {page_number} does not exist in PDF")
            
            page = pdf_document.load_page(page_number)
            
            # Convert page to image
            mat = fitz.Matrix(2.0, 2.0)  # High resolution for better OCR
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Preprocess image for better OCR
            processed_img = self._preprocess_image(np.array(img))
            
            # Extract text using OCR
            extracted_text = self._extract_text_from_image(processed_img, 'english')
            
            # Structure the extracted data
            structured_data = self._structure_text(extracted_text, processing_type)
            
            pdf_document.close()
            
            return {
                'page_number': page_number,
                'raw_text': extracted_text,
                'structured_data': structured_data,
                'processing_type': processing_type,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Single page processing error: {str(e)}")
            return {
                'page_number': page_number,
                'error': str(e),
                'success': False
            }
    
    def _process_entire_pdf(self, pdf_path: str, processing_type: str) -> Dict[str, Any]:
        """Process entire PDF in memory-efficient chunks"""
        try:
            pdf_document = fitz.open(pdf_path)
            total_pages = len(pdf_document)
            
            all_structured_data = []
            failed_pages = []
            
            # Process in chunks of 5 pages to manage memory
            chunk_size = 5
            for start_page in range(0, total_pages, chunk_size):
                end_page = min(start_page + chunk_size, total_pages)
                
                for page_num in range(start_page, end_page):
                    try:
                        page = pdf_document.load_page(page_num)
                        
                        # Convert to image
                        mat = fitz.Matrix(1.5, 1.5)  # Balanced resolution
                        pix = page.get_pixmap(matrix=mat)
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        
                        # OCR processing
                        processed_img = self._preprocess_image(np.array(img))
                        extracted_text = self._extract_text_from_image(processed_img, 'english')
                        structured_data = self._structure_text(extracted_text, processing_type)
                        
                        if structured_data and structured_data.get('entries'):
                            all_structured_data.extend(structured_data['entries'])
                        
                        # Clean up
                        del img, processed_img, pix
                        
                    except Exception as e:
                        logger.warning(f"Failed to process page {page_num}: {str(e)}")
                        failed_pages.append(page_num)
                        continue
            
            pdf_document.close()
            
            return {
                'total_pages': total_pages,
                'processed_pages': total_pages - len(failed_pages),
                'failed_pages': failed_pages,
                'structured_data': {'entries': all_structured_data},
                'processing_type': processing_type,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Full PDF processing error: {str(e)}")
            return {
                'error': str(e),
                'success': False
            }
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR accuracy
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Noise reduction
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Binarization (adaptive threshold)
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def _extract_text_from_image(self, image: np.ndarray, language: str = 'english') -> str:
        """
        Extract text from preprocessed image using Tesseract OCR
        """
        try:
            config = self.ocr_configs.get(language, self.ocr_configs['english'])
            text = pytesseract.image_to_string(image, config=config)
            return text.strip()
        except Exception as e:
            logger.warning(f"OCR extraction error: {str(e)}")
            return ""
    
    def _structure_text(self, text: str, processing_type: str) -> Dict[str, Any]:
        """
        Structure extracted text based on processing type
        """
        if processing_type == "dictionary":
            return self._structure_dictionary_text(text)
        elif processing_type == "grammar":
            return self._structure_grammar_text(text)
        else:
            return {"entries": [], "type": "unknown"}
    
    def _structure_dictionary_text(self, text: str) -> Dict[str, Any]:
        """
        Structure dictionary text into entries
        """
        entries = []
        lines = text.split('\n')
        
        current_entry = None
        
        for line in lines:
            line = line.strip()
            if not line or len(line) < 2:
                continue
            
            # Try to match dictionary entry pattern
            match = self.patterns['dictionary_entry'].match(line)
            if match:
                # Save previous entry if exists
                if current_entry:
                    entries.append(current_entry)
                
                word = match.group(1).strip()
                phonetic = match.group(2) if match.group(2) else ""
                pos = match.group(3) if match.group(3) else ""
                definition = match.group(4).strip() if match.group(4) else ""
                
                current_entry = {
                    "word": word,
                    "phonetic": phonetic,
                    "part_of_speech": pos,
                    "definition": definition,
                    "examples": [],
                    "synonyms": [],
                    "antonyms": []
                }
            elif current_entry and line.startswith(('e.g.', 'Ex:', 'Example:')):
                # Example sentence
                example = line.replace('e.g.', '').replace('Ex:', '').replace('Example:', '').strip()
                if example:
                    current_entry["examples"].append(example)
            elif current_entry and line.startswith(('Syn:', 'Synonyms:')):
                # Synonyms
                syns = line.replace('Syn:', '').replace('Synonyms:', '').strip()
                current_entry["synonyms"] = [s.strip() for s in syns.split(',') if s.strip()]
            elif current_entry and line.startswith(('Ant:', 'Antonyms:')):
                # Antonyms
                ants = line.replace('Ant:', '').replace('Antonyms:', '').strip()
                current_entry["antonyms"] = [a.strip() for a in ants.split(',') if a.strip()]
            elif current_entry and not line.startswith(tuple('ABCDEFGHIJKLMNOPQRSTUVWXYZ')):
                # Continuation of definition
                current_entry["definition"] += " " + line
        
        # Add last entry
        if current_entry:
            entries.append(current_entry)
        
        return {
            "entries": entries,
            "type": "dictionary",
            "entry_count": len(entries)
        }
    
    def _structure_grammar_text(self, text: str) -> Dict[str, Any]:
        """
        Structure grammar text into rules and examples
        """
        rules = []
        lines = text.split('\n')
        
        current_rule = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to match grammar rule pattern
            match = self.patterns['grammar_rule'].match(line)
            if match:
                # Save previous rule if exists
                if current_rule:
                    rules.append(current_rule)
                
                rule_name = match.group(1).strip()
                rule_description = match.group(2).strip()
                
                current_rule = {
                    "rule_name": rule_name,
                    "description": rule_description,
                    "examples": [],
                    "exceptions": [],
                    "category": self._categorize_grammar_rule(rule_name)
                }
            elif current_rule and self.patterns['example_sentence'].match(line):
                # Numbered example
                example_match = self.patterns['example_sentence'].match(line)
                if example_match:
                    current_rule["examples"].append(example_match.group(1))
            elif current_rule and line.startswith(('Exception:', 'Note:')):
                # Exception or note
                exception = line.replace('Exception:', '').replace('Note:', '').strip()
                current_rule["exceptions"].append(exception)
            elif current_rule and line.startswith('-'):
                # Bullet point example
                current_rule["examples"].append(line[1:].strip())
        
        # Add last rule
        if current_rule:
            rules.append(current_rule)
        
        return {
            "entries": rules,
            "type": "grammar",
            "rule_count": len(rules)
        }
    
    def _categorize_grammar_rule(self, rule_name: str) -> str:
        """
        Categorize grammar rule based on name
        """
        rule_lower = rule_name.lower()
        
        if any(word in rule_lower for word in ['tense', 'past', 'present', 'future']):
            return "tense"
        elif any(word in rule_lower for word in ['noun', 'plural', 'singular']):
            return "noun"
        elif any(word in rule_lower for word in ['verb', 'auxiliary', 'modal']):
            return "verb"
        elif any(word in rule_lower for word in ['adjective', 'comparative', 'superlative']):
            return "adjective"
        elif any(word in rule_lower for word in ['sentence', 'clause', 'syntax']):
            return "syntax"
        elif any(word in rule_lower for word in ['punctuation', 'comma', 'period']):
            return "punctuation"
        else:
            return "general"
    
    async def get_pdf_info(self, pdf_path: str) -> Dict[str, Any]:
        """
        Get basic information about the PDF file
        """
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor, 
                self._extract_pdf_info, 
                pdf_path
            )
            return result
        except Exception as e:
            logger.error(f"PDF info extraction error: {str(e)}")
            return {"error": str(e)}
    
    def _extract_pdf_info(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract basic PDF information
        """
        try:
            pdf_document = fitz.open(pdf_path)
            metadata = pdf_document.metadata
            
            info = {
                "total_pages": len(pdf_document),
                "title": metadata.get('title', 'Unknown'),
                "author": metadata.get('author', 'Unknown'),
                "subject": metadata.get('subject', ''),
                "creator": metadata.get('creator', ''),
                "file_size": Path(pdf_path).stat().st_size,
                "success": True
            }
            
            pdf_document.close()
            return info
            
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }