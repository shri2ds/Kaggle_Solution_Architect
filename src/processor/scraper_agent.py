import os
import subprocess

class KaggleScraper:
    def __init__(self, competition_id):
        self.comp_id = competition_id

        #   Get the absolute path to the root 'notebooks' folder
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_path = os.path.join(current_dir, "../../notebooks")
        os.makedirs(self.base_path, exist_ok=True)


    def fetch_top_notebooks(self, limit=10):

        print(f"🚀 Scout Agent: Searching for top {limit} notebooks in {self.comp_id}...")

        cmd = f"kaggle kernels list --competition {self.comp_id} --sort-by voteCount --page-size {limit}"

        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"❌ CLI Error: {result.stdout.strip()} \n {result.stderr.strip()}")
                return []

            # Parsing the table output
            lines = result.stdout.strip().split('\n')
            if len(lines) < 2:
                print("⚠️ No notebooks found.")
                return []

            refs = []
            for line in lines[2:]:
                parts = line.split()
                if parts:
                    refs.append(parts[0])

            print(f"✅ Found {len(refs)} notebooks. Starting downloads...")

            for ref in refs:
                print(f"📥 Pulling: {ref}...")
                pull_cmd = f"kaggle kernels pull {ref} -p {self.base_path}"
                subprocess.run(pull_cmd, shell=True)

            return refs

        except Exception as e:
            print(f"❌ Scraper Crash: {str(e)}")
            return []

