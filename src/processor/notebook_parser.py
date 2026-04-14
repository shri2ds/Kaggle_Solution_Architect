import json
import os

class NotebookProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def parse(self):
        """Extracts text from a .ipynb file, separating code and markdown."""
        if not os.path.exists(self.file_path):
            return "File not found."

        with open(self.file_path, 'r', encoding='utf-8') as f:
            nb_data = json.load(f)

        # structured_text = []
        # for cell in nb_data['cells']:
        #     cell_type = cell['cell_type']
        #     # Join the list of strings into one block
        #     source = "".join(cell['source'])
        #
        #     if cell_type == 'markdown':
        #         structured_text.append(f"### STRATEGY/EXPLANATION ###\n{source}")
        #     elif cell_type == 'code':
        #         structured_text.append(f"### CODE IMPLEMENTATION ###\n{source}")
        #
        # return "\n\n".join(structured_text)

        processed_cells = []
        for cell in nb_data['cells']:
            content = "".join(cell['source'])
            if not content.strip(): continue

            processed_cells.append({
                "text": content,
                "metadata": {
                    "type": cell['cell_type'],
                    "source": self.file_path
                }
            })

        return processed_cells


if __name__ == "__main__":
    # Test it on any .ipynb file you have in your notebooks folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    notebook_path = os.path.join(current_dir, "../../notebooks/lmsys-kerasnlp-starter.ipynb")
    processor = NotebookProcessor(notebook_path)
    clean_text = processor.parse()

    if "File not found" in clean_text:
        print(f"❌ Still failing. Tried looking at: {os.path.abspath(notebook_path)}")
    else:
        print("✅ Success! Preview:")
        print(clean_text[:500])