import os

# ----------------------------
# Helper functions
# ----------------------------
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        print(f"⚠️ Directory already exists: {path}")

def create_file(path):
    if not os.path.exists(path):
        with open(path, "w") as f:
            pass
        print(f"Created file: {path}")
    else:
        print(f"⚠️ File already exists: {path}")

# ----------------------------
# Folder structure
# ----------------------------
directories = [
    ".streamlit",
    "pages",
    "src/ui",
    "src/shared",
    "src/projects/diabetes",
    "src/projects/netflix",
    "src/projects/spotify",
    "src/projects/churn",
    "artifacts/raw",
    "artifacts/processed",
    "data",
    "tests"
]

# ----------------------------
# Files to create
# ----------------------------
files = [
    "src/__init__.py",
    "src/config.py",

    "src/ui/__init__.py",
    "src/ui/theme.py",
    "src/ui/layout.py",
    "src/ui/components.py",

    "src/shared/__init__.py",
    "src/shared/paths.py",
    "src/shared/data_loader.py",
    "src/shared/plotting.py",
    "src/shared/utils.py",

    "src/projects/__init__.py",

    "src/projects/diabetes/__init__.py",
    "src/projects/netflix/__init__.py",
    "src/projects/spotify/__init__.py",
    "src/projects/churn/__init__.py",

    "tests/__init__.py",
    "tests/test_smoke_imports.py",

    ".streamlit/config.toml",
    "requirements.txt",
    "README.md",

    "pages/1_🩺_Diabetes_Prediction.py",
    "pages/2_🎬_Netflix_Clustering.py",
    "pages/3_🎵_Spotify_Popularity.py",
    "pages/4_📉_Customer_Churn.py"
]

# ----------------------------
# Execution
# ----------------------------
def main():
    print("Setting up project structure...\n")

    # Create directories
    for directory in directories:
        create_dir(directory)

    # Create files
    for file in files:
        create_file(file)

    print("\n Project setup complete!")

if __name__ == "__main__":
    main()