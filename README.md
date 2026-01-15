[![Python](https://img.shields.io/badge/python-3.9+-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/harmydiano/yorubastopwords/blob/main/LICENSE)
[![Demo](https://img.shields.io/badge/demo-online-brightgreen)](https://yorubastopwords.onrender.com/)

# yorubastopwords
...

# Yoruba Stopwords Generator

A web application for automatically extracting stopwords from Yoruba language documents using statistical natural language processing techniques.

## Overview

This tool analyzes Yoruba text documents and identifies common words (stopwords) that should typically be filtered out in NLP tasks such as information retrieval, text mining, and document classification. The system uses **6 complementary methods** with intelligent weighted voting:

### Statistical Methods
- **Frequency Analysis**: Identifies highly frequent words (top 2% by default)
- **TF-IDF (Term Frequency-Inverse Document Frequency)**: Finds words with low TF-IDF scores (common across all documents)
- **Entropy Calculation**: Measures document-level distribution uniformity (high entropy = evenly distributed)
- **Variance Analysis**: Identifies words with consistent relative frequency across documents (low variance = consistent)
- **Dispersion Analysis**: Uses Gries' DP metric to measure even distribution (low dispersion = evenly spread)

### Linguistic Methods
- **Function Word Patterns**: Matches against known Yoruba function words (pronouns, conjunctions, prepositions, etc.)

## Features

- ✨ **Multi-document processing**: Upload and process multiple Yoruba documents
- 📊 **Advanced statistical analysis**: Uses 5 statistical metrics + linguistic patterns
- 🧠 **Intelligent combination**: Weighted voting system with linguistic priority
- 🌐 **Web interface**: User-friendly drag-and-drop file upload
- 📁 **Multiple output formats**: TXT, JSON, CSV, and DOCX
- 🔤 **Yoruba language support**: Properly handles tone marks and special characters (àáèéẹọìíòóùúṣ)
- 📈 **Detailed statistics**: Provides comprehensive analysis metrics and filter breakdown
- 🎯 **Scientifically sound**: Corrected implementations based on NLP research
- 🚀 **RESTful API**: Programmatic access to stopword generation
- 🐳 **Docker support**: Easy deployment with containers

## Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/harmydiano/yorubastopwords.git
   cd yorubastopwords
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the application**
   Open your browser and navigate to `http://localhost:5001`

### Docker Setup

1. **Build the Docker image**
   ```bash
   docker build -t yoruba-stopwords .
   ```

2. **Run the container**
   ```bash
   docker run -p 5001:5001 yoruba-stopwords
   ```

## Getting Yoruba Documents

For testing and development, you can use Yoruba text corpora from the **Niger-Volta Language Technology Initiative**:

🔗 **[Yoruba Text Corpus Repository](https://github.com/Niger-Volta-LTI/yoruba-text)**

This repository provides:
- Various Yoruba text documents and corpora
- Religious texts (Bible translations)
- News articles
- Literature samples
- Pre-processed text files ready for NLP tasks

### Quick Start with Sample Data

```bash
# Clone the Yoruba text repository
git clone https://github.com/Niger-Volta-LTI/yoruba-text.git

# Copy sample documents to your uploads folder
cp yoruba-text/samples/*.txt ./uploads/

# Process the documents
python app.py
```

## Usage

### Web Interface

1. Navigate to `http://localhost:5001`
2. Upload one or more Yoruba documents (`.docx`, `.doc`, or `.txt`)
3. Click "Upload and Process"
4. Download the generated stopwords in your preferred format


### Demo Url
https://yorubastopwords.onrender.com/

### API Usage

#### Get Stopwords (JSON)

```bash
GET /api/stopwords
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "stopwords": ["ni", "ti", "ló", "wà", ...],
    "statistics": {
      "total_documents": 2,
      "total_words": 15420,
      "unique_words": 3240,
      "final_stopwords": 42
    }
  }
}
```

#### Health Check

```bash
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-10T12:00:00"
}
```

### Python API

```python
from tfidf_analyzer import TFIDFAnalyzer

# Initialize analyzer
analyzer = TFIDFAnalyzer(
    filenames=['document1.docx', 'document2.docx'],
    min_word_length=2,
    max_word_length=5,
    min_frequency=5
)

# Process documents
stopwords = analyzer.process()

# Get statistics
stats = analyzer.get_statistics()
print(f"Found {len(stopwords)} stopwords")

# Save results
analyzer.save_stopwords(
    output_path='output/stopwords',
    formats=['txt', 'json', 'csv']
)
```

### Default Stopword List

A pre-generated **default stopword list** is included for a quick use.  

It was generated using the tool's default settings (combined statistical + linguistic methods) on a religious texts from the Niger-Volta Yoruba corpus.

- **TXT format** (simple, one word per line): [`data/yoruba_stopwords_default.txt`](./data/yoruba_stopwords.txt)  
- **JSON format** (with metadata like generation date and thresholds): [`data/yoruba_stopwords_default.json`](./data/yoruba_stopwords.json) 

**Note**: This is a list generated from a religious corpus. For specialized use cases (e.g., only news or literature), generate a custom list via the web app, API, or Python class.

### Quick Usage Example (Python)

```python
# Load the default stopwords into a set (recommended for fast lookups)
with open('data/yoruba_stopwords.txt', encoding='utf-8') as f:
    yoruba_stopwords = set(line.strip() for line in f if line.strip())

# Example: Filter stopwords from a sentence
text = "Mo fẹ́ ra ọkọ̀ ayọ́kẹ́lẹ́ lọ́wọ́ ọ̀rẹ́ mi"
words = text.split()
filtered = [word for word in words if word.lower() not in yoruba_stopwords]
print("Filtered:", filtered)

## Configuration

Configuration is managed through environment variables and the `config.py` file.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FLASK_ENV` | Environment (development/production) | `development` |
| `SECRET_KEY` | Flask secret key | Auto-generated |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `5001` |
| `DEBUG` | Debug mode | `True` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `MIN_WORD_LENGTH` | Minimum word length | `2` |
| `MAX_WORD_LENGTH` | Maximum word length | `5` |
| `MIN_FREQUENCY` | Minimum word frequency | `5` |

### Statistical Thresholds (Percentile-based)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `FREQUENCY_PERCENTILE` | Top N% most frequent words | `98` (top 2%) |
| `TFIDF_PERCENTILE` | Bottom N% (low TF-IDF) | `25` (bottom 25%) |
| `ENTROPY_PERCENTILE` | Top N% (high entropy) | `75` (top 25%) |
| `VARIANCE_PERCENTILE` | Bottom N% (low variance) | `25` (bottom 25%) |
| `DISPERSION_PERCENTILE` | Bottom N% (low dispersion) | `75` (adjustable) |
| `SINGLE_DOC_CHUNKS` | Number of chunks for single docs | `4` |

### Configuration Parameters

Edit `config.py` to customize:

- **Upload settings**: File size limits, allowed extensions
- **Processing parameters**: Word length, frequency thresholds
- **Output formats**: Available export formats
- **Yoruba characters**: Character set for text processing

## Algorithm

The stopword extraction process uses a **scientifically validated multi-stage pipeline**:

### Pipeline Stages

1. **Text Extraction**: Convert .docx documents to plain text using docx2txt

2. **Preprocessing**:
   - Convert to lowercase
   - Remove punctuation and non-Yoruba characters
   - Filter by word length (2-5 characters by default)
   - Split single documents into random chunks for statistical independence

3. **Statistical Analysis** (5 metrics):
   - **Frequency**: Count word occurrences across corpus
   - **TF-IDF**: Compute term frequency-inverse document frequency using scikit-learn
   - **Entropy**: Calculate Shannon entropy over document distribution
   - **Variance**: Measure consistency of normalized frequencies across documents
   - **Dispersion**: Calculate Gries' Deviation of Proportions (DP)

4. **Linguistic Analysis**:
   - Match words against Yoruba function word dictionary (90+ patterns)

5. **Filtering**: Apply percentile-based thresholds to each statistical metric

6. **Intelligent Combination**:
   - **Weighted voting system** (not hard intersection)
   - Words pass if: **(Linguistic match + 2+ statistical filters) OR (4+ statistical filters)**
   - Prioritizes linguistic patterns over pure statistics

### Mathematical Details

**Frequency Analysis (Percentile-based)**
```
threshold = percentile(frequencies, 98)  # Top 2%
stopword if: freq(w) ≥ threshold
```

**TF-IDF (Low scores indicate stopwords)**
```
tfidf(w,d) = tf(w,d) × log(N/df(w))
threshold = percentile(tfidf_scores, 25)  # Bottom 25%
stopword if: tfidf(w) ≤ threshold
```

**Entropy (Document-level distribution)**
```
H(w) = -Σ[p(w,d) × log₂(p(w,d))]  for all documents d
normalized_H(w) = H(w) / log₂(N)
threshold = percentile(entropy_scores, 75)  # Top 25%
stopword if: H(w) ≥ threshold
```

**Variance (Consistency across documents)**
```
Var(w) = variance([freq(w,d₁)/|d₁|, freq(w,d₂)/|d₂|, ...])
threshold = percentile(variance_scores, 25)  # Bottom 25%
stopword if: Var(w) ≤ threshold
```

**Dispersion (Gries' DP - Even distribution)**
```
DP(w) = 0.5 × Σ|observed_proportion(w,dᵢ) - expected_proportion(dᵢ)|
threshold = percentile(dispersion_scores, 75)
stopword if: DP(w) ≤ threshold
```

**Combination Logic**
```
votes = count of filters passed
stopword if:
  (word in YORUBA_FUNCTION_WORDS AND votes ≥ 2) OR
  (votes ≥ 4)
```

## Output Formats

### TXT Format
```
ilẹ̀
di
ló
ni
ti
...
```

### JSON Format
```json
{
  "stopwords": ["ilẹ̀", "di", "ló", ...],
  "statistics": {
    "total_documents": 2,
    "total_words": 15420,
    "unique_words": 3240
  },
  "metadata": {
    "min_word_length": 2,
    "max_word_length": 5
  }
}
```

### CSV Format
```csv
word,frequency,tfidf_score
ilẹ̀,245,0.012340
di,198,0.010234
...
```

## Project Structure

```
yorubastopwords/
├── app.py                 # Main Flask application
├── tfidf_analyzer.py      # Core stopword extraction logic
├── config.py              # Configuration management
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose setup
├── templates/            # HTML templates
│   └── index.html       # Main page
├── static/              # Static assets (CSS, JS, images)
├── uploads/             # Uploaded files (gitignored)
├── logs/                # Application logs
└── tests/               # Unit tests
```

## Development

### Running Tests

```bash
pytest tests/ -v --cov=.
```

### Code Formatting

```bash
black *.py
flake8 *.py
mypy *.py
```

### Adding New Features

1. Create a new branch
2. Implement your feature
3. Add tests
4. Submit a pull request

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Yoruba Language Support

This tool properly handles Yoruba orthography including:

- Tone marks: à, á, è, é, ẹ̀, ẹ́, ì, í, ò, ó, ọ̀, ọ́, ù, ú
- Special characters: ṣ, Ṣ
- Both uppercase and lowercase variants

## Use Cases

- **Information Retrieval**: Improve search relevance in Yoruba documents
- **Text Mining**: Preprocess Yoruba text for analysis
- **Machine Learning**: Feature engineering for NLP models
- **Academic Research**: Linguistic analysis of Yoruba language
- **Digital Libraries**: Index and catalog Yoruba literature

## Performance

- Processes ~10,000 words per second
- Handles documents up to 16MB
- Supports batch processing of multiple files
- Efficient memory usage with streaming

## License

MIT License - see [LICENSE](LICENSE) file for details

## Acknowledgments

- Built with Flask, scikit-learn, and python-docx
- Inspired by research in low-resource NLP
- Yoruba language support based on standard orthography

## Support

For questions or issues, please open an issue on GitHub.

---

Made by Diano with ❤️ for the Yoruba NLP community
