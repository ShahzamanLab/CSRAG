import os
from pathlib import Path

list_of_files = [
    "README.md",
    "requirements.txt",
    ".env",
    "data.txt",
    "Dockerfile",
    "docker-compose.yml",

    "app.py",

    "src/",
    "src/__init__.py",

    "src/config.py",
    "src/logger.py",
    "src/metrics.py",
    "src/utils.py",

    # Ingestion
    "src/ingestion/",
    "src/ingestion/__init__.py",
    "src/ingestion/loader.py",
    "src/ingestion/chunker.py",
    "src/ingestion/embedder.py",
    "src/ingestion/evaluator_finetune_data_generation.py",

    # Retrieval / CRAG
    "src/retrieval/",
    "src/retrieval/__init__.py",
    "src/retrieval/retriever.py",
    "src/retrieval/evaluator.py",
    "src/retrieval/evaluator_test.py",
    "src/retrieval/evaluator_class.py",
    "src/retrieval/route_picker.py",


    # Database / Cache
    "src/db/",
    "src/db/__init__.py",
    "src/db/vector_db.py",

    # API / Deployment
    "src/api/",
    "src/api/__init__.py",
    "src/api/main.py",
    "src/api/routes.py",

    # Notebooks & experiments
    "notebooks/",
    "notebooks/data_exploration.ipynb",
    "notebooks/model_testing.ipynb",

    # Tests
    "tests/",
    "tests/__init__.py",
    "tests/test_ingestion.py",
    "tests/test_retrieval.py",
    "tests/test_generation.py",
    "templates/index.html",
    "static/css/style.css",
    "static/js/script.js",
    "test.py"
]

for file_path in list_of_files:
    path = Path(file_path)
    folder = path.parent
    if folder != Path("."):
        os.makedirs(folder, exist_ok=True)
    if path.suffix:  
        path.touch(exist_ok=True)