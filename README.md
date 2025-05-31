# financial-statement-parser
Financial Statement Converter from Docx to Json
# Financial Statement Parser

## Project Overview

The Financial Statement Parser is a Python-based tool designed to analyze `.docx` financial documents. It parses these documents, identifies key sections such as the Cover Page, Statement of Financial Position (SoFP), Statement of Comprehensive Income (SoCI), and Financial Notes, and then outputs a structured JSON representation of these sections.

The core functionality relies on a series of rule-based classifiers, each tailored to identify specific financial statement components. These classifiers use configurable JSON rules to define keywords, patterns, and scoring heuristics.

## Features

* Parses `.docx` files to extract textual and structural content into meaningful blocks.
* Identifies the following financial statement sections:
    * Cover Page
    * Statement of Financial Position (Balance Sheet)
    * Statement of Comprehensive Income (Income Statement/P&L)
    * Financial Notes (individually identified and titled)
* Rule-based classification using external JSON configuration files for flexibility.
* Generates a structured JSON output detailing the identified sections, their block ranges, and confidence scores.
* Supports debug logging for detailed analysis of the parsing and classification process.

## Project Structure

The project is organized into the following key components:

* **`main.py`**: The main entry point for the script. It handles command-line arguments, orchestrates the parsing and classification workflow, and outputs the final results.
* **`docx_parser.py`**: Responsible for reading and parsing the content of `.docx` files into a list of structured blocks (e.g., paragraphs, tables).
* **`classifiers/`**: This directory contains the individual classifier modules:
    * `base_classifier.py`: An abstract base class for all section classifiers, providing common functionality like loading rules and displaying results.
    * `cover_page_classifier.py`: Identifies the cover page of the financial document.
    * `sofp_classifier.py`: Identifies the Statement of Financial Position.
    * `soci_classifier.py`: Identifies the Statement of Comprehensive Income.
    * `financial_notes_classifier.py`: Identifies and segments individual financial notes.
* **`rules/`**: This directory stores the JSON configuration files that define the heuristics and keywords for each classifier:
    * `cover_page_rules.json`
    * `sofp_rules.json`
    * `soci_rules.json`
    * `financial_notes_rules.json`
* **`utils/`**: Contains utility modules:
    * `section_generator.py`: A utility class to accumulate and format the identified sections into the final JSON output.
    * `text_utils.py` (Assumed): Likely contains functions for text normalization, phrase finding, and regex pattern matching used by the classifiers. *(You'll want to confirm and detail this)*

## Prerequisites

* Python 3.x
* The `python-docx` library (or a similar library used by `DocxParser` for reading `.docx` files). *(You'll need to specify this based on your `DocxParser` implementation)*
* Any other libraries used by the utility or classifier modules (e.g., `nltk` if you implement stemming/lemmatization).

## Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <your-repository-url>
    cd financial-statement-parser
    ```

2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    Create a `requirements.txt` file listing all necessary packages (e.g., `python-docx`). Then run:
    ```bash
    pip install -r requirements.txt
    ```
    *(Alternatively, list manual pip install commands here if `requirements.txt` isn't set up yet).*

## Usage

The script is run from the command line.

**Basic command:**

```bash
python main.py <path_to_your_docx_file.docx>

Optional arguments:

--debug: Enables detailed debug logging output to the console.

python main.py <path_to_your_docx_file.docx> --debug

--rules_dir <directory_path>: Specifies the directory containing the JSON rule files. If not provided, it defaults to a rules/ subdirectory relative to main.py.

python main.py <path_to_your_docx_file.docx> --rules_dir /custom/path/to/rules

Example:

python main.py "Financial Report Q4.docx" --debug

The script will output the identified financial sections in JSON format to the console (primarily via logger info messages).

How it Works
Setup Logging: Initializes logging based on the --debug flag.

Load and Parse DOCX (DocxParser): The input .docx file is read, and its content is broken down into a list of "meaningful blocks." Each block typically has an index, type (e.g., 'paragraph', 'table'), and its textual content.

Sequential Classification: The script then attempts to identify sections in a predefined order:

Cover Page

Statement of Financial Position (SoFP)

Statement of Comprehensive Income (SoCI)

Financial Notes

Rule-Based Classifiers: Each classifier (CoverPageClassifier, SoFPClassifier, etc.) loads its specific rules from the corresponding JSON file in the rules/ directory. It then iterates through the document blocks (starting from where the previous section likely ended) and applies its heuristics (keyword matching, pattern recognition, structural cues, scoring logic) to identify its target section.

Section Generation (SectionGenerator): As sections are identified and their confidence scores meet predefined thresholds, they are added to a SectionGenerator instance. This utility helps in organizing the final output.

Output: The main.py script logs the final structured list of identified sections in JSON format.

Rules Configuration
The accuracy and behavior of the classifiers are heavily dependent on the JSON rule files located in the rules/ directory. These files define:

Keywords: Specific phrases and terms indicative of a section or its components.

Regex Patterns: Regular expressions for more flexible pattern matching.

Scoring Logic: Points awarded for finding keywords, matching patterns, or satisfying structural heuristics.

Thresholds: Minimum scores or confidence levels required for a section to be considered identified.

Termination Logic: Rules that help a classifier determine where its section ends and another might begin.

Modifying these JSON files is the primary way to tune the parser's performance for different styles or specific requirements of financial documents.

Future Enhancements / To-Do
Implement more advanced NLP techniques (e.g., stemming, lemmatization, synonym handling) in text_utils.py for more robust keyword matching.

Develop a more sophisticated DocxParser to extract richer structural information (e.g., font styles, heading levels, table cell details).

Add a dedicated output module to save the JSON results to a file instead of just logging.

Implement a more formal evaluation framework and test suite for measuring classifier accuracy.

Consider machine learning approaches as an alternative or supplement to the rule-based system for certain classification tasks.

Add classifiers for other common financial statement sections (e.g., Statement of Cash Flows, Statement of Changes in Equity).

Contributing
(Details on how others can contribute, if applicable: e.g., pull requests, issue reporting, coding standards).

License
*(Specify the license for your