from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import json
import logging

logger = logging.getLogger(__name__)

class BaseClassifier(ABC):
    """
    Abstract Base Class for all document section classifiers.
    """

    def __init__(self, rules_file_path: Optional[str] = None, rules_dict: Optional[Dict[str, Any]] = None):
        """
        Initializes the classifier, optionally loading rules from a file or a dictionary.

        """
        self.rules: Dict[str, Any] = {}
        self.max_score_exemplar: float = 1.0

        if rules_file_path:
            self.rules = self._load_rules_from_file(rules_file_path)
        elif rules_dict:
            self.rules = rules_dict
        else:
            logger.warning(f"No rules provided for {self.__class__.__name__}. Classifier may not function correctly.")

        # Check if rules were loaded
        if self.rules:
            self.max_score_exemplar = float(self.rules.get("max_score_exemplar", 1.0))
            if self.max_score_exemplar <= 0:
                logger.warning(f"max_score_exemplar for {self.__class__.__name__} is {self.max_score_exemplar}. Setting to 1.0 to avoid errors.")
                self.max_score_exemplar = 1.0


    def _load_rules_from_file(self, file_path: str) -> Dict[str, Any]:
        """Loads rules from a JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded_rules = json.load(f)
                logger.info(f"Successfully loaded rules for {self.__class__.__name__} from {file_path}")
                return loaded_rules
        except FileNotFoundError:
            logger.error(f"Rules file not found: {file_path} for {self.__class__.__name__}")
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from rules file: {file_path} for {self.__class__.__name__}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading rules from {file_path} for {self.__class__.__name__}: {e}")
        return {}

    @abstractmethod
    def classify(self, doc_blocks: List[Dict[str, Any]], **kwargs) -> Optional[Dict[str, Any]]:
        """
        Main classification method. Identifies and returns information about the classified section.
        """
        pass

    @abstractmethod
    def _calculate_score(self, combined_text: str, first_block_index: int, **kwargs) -> Dict[str, Any]:
        """
        Calculates the raw score for a given text content.
        Returns a dictionary with "total" score and "breakdown".
        """
        pass

    def display_results(self, classification_result: Optional[Dict[str, Any]], all_doc_blocks: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Displays the classification results.
        """
        section_name = classification_result.get("section_name", "N/A") if classification_result else "N/A"
        logger.info(f"\n{section_name.upper()} CLASSIFICATION RESULTS:")
        logger.info("-" * 40)

        if classification_result:
            logger.info(f" Section identified: {classification_result.get('section_name', 'N/A')}")
            confidence = classification_result.get('confidence', 0.0)
            raw_score = classification_result.get('raw_score', 0)
            num_blocks = classification_result.get('num_blocks', 0)
            start_idx = classification_result.get('start_block_index', 'N/A')
            end_idx = classification_result.get('end_block_index', 'N/A')

            logger.info(f"  Confidence: {confidence:.3f}")
            logger.info(f"  Raw score: {raw_score}")
            logger.info(f"  Number of blocks: {num_blocks}")
            logger.info(f"  Start Block Index: {start_idx}")
            logger.info(f"  End Block Index: {end_idx}")

            if logger.isEnabledFor(logging.DEBUG):
                breakdown = classification_result.get('breakdown', {})
                if breakdown:
                    logger.debug(f"\n  Score breakdown:")
                    for category, score in breakdown.items():
                        if isinstance(score, (int, float)) and score != 0: # Only show contributing scores
                            logger.debug(f"    • {category.replace('_', ' ').title()}: {score:+.2f}")
                        elif isinstance(score, list) and score and "keywords" in category.lower(): # Show lists of keywords
                            logger.debug(f"    • {category.replace('_', ' ').title()}: {', '.join(str(s) for s in score[:5])}{'...' if len(score) > 5 else ''}")


                content_preview = classification_result.get("content_preview") or classification_result.get("content", "")[:200]
                logger.debug(f"\n  Content Preview (first 200 chars):")
                logger.debug(f"  '{content_preview.strip().replace(chr(10), ' ')}...'")

        else:
            logger.info(f"\n  No {section_name.lower()} section identified with sufficient confidence.")
