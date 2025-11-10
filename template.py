from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "VideoCaptioning"


# List of files to be created
list_of_files = [
    ".github/workflows/.gitkeep",
    "src/__init__.py",
    "src/components/__init__.py",
    "src/utils/__init__.py",
    "src/config/__init__.py",
    "src/config/configuration.py",
    "src/pipeline/__init__.py",
    "src/entity/__init__.py",
    "src/constants/__init__.py",
    'main.py',
    'CLI.py',
    
    # MLflow files
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    
    # setup files
    "requirements.txt",
    "setup.py",
    "README.md",

    # Jupyter Notebook files
    "notebook/",

    # Data files
    "dataset"
]

# Function to create directories and files
def create_directories_and_files(project_name, list_of_files):
    for file_path in list_of_files:
        file_path = Path(file_path)
        # Create directories if they don't exist
        if file_path.suffix == "":
            if not file_path.exists():
                file_path.mkdir(parents=True, exist_ok=True)
                logging.info(f"Created directory: {file_path}")
        else:
            # Create files if they don't exist
            if not file_path.exists():
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.touch(exist_ok=True)
                logging.info(f"Created file: {file_path}")


if __name__ == "__main__":
    # Create directories and files
    create_directories_and_files(project_name, list_of_files)
    logging.info(f"Project structure for '{project_name}' created successfully.")