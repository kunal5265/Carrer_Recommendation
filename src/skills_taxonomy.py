"""Canonical skill taxonomy with aliases used for resume skill extraction.

Each entry maps a canonical skill name to the surface forms (aliases) that
should resolve to it. Matching is case-insensitive and word-boundary aware.
"""

SKILL_TAXONOMY = {
    "Python": ["python", "python3", "python programming"],
    "JavaScript": ["javascript", "js", "es6", "ecmascript"],
    "TypeScript": ["typescript", "ts"],
    "React": ["react", "reactjs", "react.js", "react js"],
    "Node.js": ["nodejs", "node.js", "node js", "node"],
    "HTML": ["html", "html5"],
    "CSS": ["css", "css3", "sass", "tailwind", "bootstrap"],
    "SQL": ["sql", "mysql", "postgresql", "postgres", "sqlite", "t-sql", "bigquery"],
    "Java": ["java", "spring boot"],
    "Git": ["git", "github", "gitlab", "version control"],
    "REST APIs": ["rest api", "restful", "rest apis", "http api", "api design"],
    "Pandas": ["pandas", "dataframe manipulation"],
    "NumPy": ["numpy", "np.random"],
    "Scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "PyTorch": ["pytorch", "torch"],
    "TensorFlow": ["tensorflow", "keras"],
    "Deep Learning": ["deep learning", "neural network", "neural networks", "cnn", "cnn's"],
    "Machine Learning": ["machine learning", "ml models", "supervised learning", "unsupervised learning"],
    "NLP": ["nlp", "natural language processing", "text classification", "named entity recognition", "tokenization"],
    "spaCy": ["spacy", "spacynlp"],
    "Transformers": ["transformers", "bert", "gpt", "hugging face", "huggingface", "llm", "large language model"],
    "Sentence Embeddings": ["sentence embeddings", "word embeddings", "embeddings", "semantic search", "vector search", "faiss"],
    "LLM APIs": ["llm api", "openai api", "langchain", "prompt engineering", "rag", "retrieval-augmented"],
    "Statistics": ["statistics", "statistical analysis", "probability", "hypothesis testing", "regression"],
    "A/B Testing": ["a/b testing", "ab testing", "experiment design", "ab tests"],
    "Data Visualization": ["data visualization", "visualization", "matplotlib", "seaborn", "plotly", "visual storytelling"],
    "Dashboarding": ["dashboard", "dashboards", "dashboarding", "tableau", "power bi", "looker"],
    "Excel": ["excel", "vlookup", "pivot tables"],
    "Spark": ["spark", "pyspark", "apache spark"],
    "Airflow": ["airflow", "apache airflow"],
    "Data Modeling": ["data modeling", "data modelling", "data warehouse", "dbt", "star schema"],
    "MLOps": ["mlops", "mlflow", "model serving", "model monitoring", "feature store", "kubeflow"],
    "Docker": ["docker", "dockerfile", "containerization", "containers"],
    "Kubernetes": ["kubernetes", "k8s"],
    "AWS": ["aws", "amazon web services", "s3", "ec2", "lambda", "sagemaker"],
    "Terraform": ["terraform", "infrastructure as code", "iac"],
    "CI/CD": ["ci/cd", "cicd", "continuous integration", "continuous delivery", "jenkins", "github actions"],
    "Linux": ["linux", "bash", "shell scripting", "unix"],
    "Redis": ["redis", "caching", "cache"],
    "System Design": ["system design", "system architecture", "distributed systems", "scalability", "microservices"],
    "OpenCV": ["opencv", "cv2"],
    "Computer Vision": ["computer vision", "image processing", "object detection", "image segmentation", "yolo"],
    "Research": ["research", "paper reproduction", "literature review", "publications"],
    "Communication": ["communication", "stakeholder", "presentation skills", "cross-functional", "teamwork", "leadership"],
    "Snowflake": ["snowflake"],
    "GraphQL": ["graphql"],
}

# Skills that appear in job postings but no course in our catalog directly
# teaches; they are still tracked for gap analysis.
DIFFICULT_SKILLS = set()
