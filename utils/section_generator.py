import json

class SectionGenerator:
    """
    A class for creating financial statement section objects and arrays.
    """

    def __init__(self):
        """
        Initialize the SectionGenerator.
        """
        self.sections = []

    def create_normalized_section_object(self, section_header, normalized_section, start_block, end_block, confidence_rate):
        """
        Creates a normalized section object with 5 fields.

        Args:
            section_header (string): The display name of the section
            normalized_section (string): The normalized/standardized name
            start_block (int): Starting block number
            end_block (int): Ending block number
            confidence_rate (float): Confidence rate (uses default if None)

        Returns:
            dict: Section object with 5 fields
        """

        return {
            "section_header": section_header,
            "normalized_section": normalized_section,
            "start_block": start_block,
            "end_block": end_block,
            "confidence_rate": confidence_rate
        }

    def create_note_section(self, section_header, note_number, note_title, start_block, end_block, confidence_rate):
        """
        Creates a notes section object with 6 fields.

        Args:
            section_header (string): The display name of the section
            note_number (int): The note number
            note_title (string): The title of the note
            start_block (int): Starting block number
            end_block (int): Ending block number
            confidence_rate (float): Confidence rate (uses default if None)

        Returns:
            dict: Notes section object with 6 fields
        """

        return {
            "section_header": section_header,
            "note_number": note_number,
            "note_title": note_title,
            "start_block": start_block,
            "end_block": end_block,
            "confidence_rate": confidence_rate
        }

    def add_normalized_section(self, section_header, normalized_section, start_block, end_block, confidence_rate):
        """
        Creates and adds a normalized section to the internal sections list.

        Args:
            section_header (string): The display name of the section
            normalized_section (str): The normalized/standardized name
            start_block (int): Starting block number
            end_block (int): Ending block number
            confidence_rate (float): Confidence rate (uses default if None)

        Returns:
            SectionGenerator: Self for method chaining
        """
        section = self.create_normalized_section_object(
            section_header, normalized_section, start_block, end_block, confidence_rate
        )
        self.sections.append(section)
        return self

    def add_note_section(self, section_header, note_number, note_title, start_block, end_block, confidence_rate):
        """
        Creates and adds a note section to the internal sections list.

        Args:
            section_header (string): The display name of the section
            note_number (int): The note number
            note_title (string): The title of the note
            start_block (int): Starting block number
            end_block (int): Ending block number
            confidence_rate (float): Confidence rate (uses default if None)

        Returns:
            SectionGenerator: Self for method chaining
        """
        section = self.create_note_section(
            section_header, note_number, note_title, start_block, end_block, confidence_rate
        )
        self.sections.append(section)
        return self

    def get_sections(self):
        """
        Returns the current list of sections.

        Returns:
            list: List of section dictionaries
        """
        json_string = json.dumps(self.sections, indent=2)
        return json_string

    def clear_sections(self):
        """
        Clears the internal sections list.

        Returns:
            SectionGenerator: Self for method chaining
        """
        self.sections.clear()
        return self
