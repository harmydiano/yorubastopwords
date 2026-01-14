"""
Enhanced TF-IDF Analyzer for Yoruba Stopword Extraction

This module provides improved statistical methods for identifying stopwords
in Yoruba text using multiple metrics: TF-IDF, frequency, entropy, and variance.
"""

from typing import List, Dict, Set, Tuple, Optional
import re
import math
import logging
import random
from pathlib import Path
from collections import Counter
import json

import docx2txt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from config import Config

# Setup logging
logger = logging.getLogger(__name__)


class TFIDFAnalyzer:
    """
    Advanced TF-IDF based stopword analyzer for Yoruba language.

    This class processes Yoruba documents to extract stopwords using
    statistical methods including TF-IDF, frequency analysis, entropy,
    and variance calculations.
    """

    def __init__(
        self,
        filenames: List[str],
        min_word_length: int = Config.MIN_WORD_LENGTH,
        max_word_length: int = Config.MAX_WORD_LENGTH,
        min_frequency: int = Config.MIN_FREQUENCY
    ):
        """
        Initialize the TF-IDF analyzer.

        Args:
            filenames: List of document filenames to process
            min_word_length: Minimum word length to consider
            max_word_length: Maximum word length to consider
            min_frequency: Minimum frequency threshold for stopwords
        """
        self.filenames = filenames
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        self.min_frequency = min_frequency

        # Data storage
        self.documents: List[List[str]] = []
        self.merged_words: List[str] = []
        self.word_frequencies: Counter = Counter()

        # Statistical results
        self.tfidf_scores: Dict[str, float] = {}
        self.entropy_scores: Dict[str, float] = {}
        self.variance_scores: Dict[str, float] = {}
        self.dispersion_scores: Dict[str, float] = {}

        # Filtered word sets
        self.high_frequency_words: Set[str] = set()
        self.low_tfidf_words: Set[str] = set()
        self.high_entropy_words: Set[str] = set()
        self.low_variance_words: Set[str] = set()
        self.high_dispersion_words: Set[str] = set()
        self.function_word_candidates: Set[str] = set()

        # Final stopwords
        self.stopwords: Set[str] = set()

        logger.info(f"Initialized TFIDFAnalyzer with {len(filenames)} files")

    def process(self) -> Set[str]:
        """
        Execute the complete stopword extraction pipeline.

        Returns:
            Set of identified stopwords
        """
        logger.info("Starting stopword extraction pipeline")

        # Step 1: Extract and preprocess text
        self.extract_text_from_documents()
        self.preprocess_documents()

        # Step 2: Calculate statistics
        self.calculate_frequencies()
        self.calculate_tfidf()
        self.calculate_entropy()
        self.calculate_variance()
        self.calculate_dispersion()

        # Step 3: Apply linguistic filtering
        self.filter_by_function_words()

        # Step 4: Apply statistical filters
        self.filter_by_frequency()
        self.filter_by_tfidf()
        self.filter_by_entropy()
        self.filter_by_variance()
        self.filter_by_dispersion()

        # Step 5: Combine filters to get final stopwords
        self.combine_filters()

        logger.info(f"Extracted {len(self.stopwords)} stopwords")
        return self.stopwords

    def extract_text_from_documents(self) -> None:
        """Extract text from documents (.docx, .doc, .txt) and save as .txt files."""
        logger.info("Extracting text from documents")

        for idx, filename in enumerate(self.filenames):
            try:
                # Handle different file paths
                file_path = Path(filename)
                if not file_path.exists():
                    file_path = Path(Config.UPLOAD_FOLDER) / filename

                # Get file extension
                file_ext = file_path.suffix.lower()

                # Extract text based on file type
                if file_ext in ['.docx', '.doc']:
                    # Extract text from Word documents
                    text = docx2txt.process(str(file_path))
                elif file_ext == '.txt':
                    # Read text files directly
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                else:
                    logger.warning(f"Unsupported file type: {file_ext} for {filename}")
                    # Try to read as text anyway
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()

                # Save to output file
                output_file = Path(f"output_{idx}.txt")
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(text)

                logger.debug(f"Extracted text from {filename} ({file_ext}) to {output_file}")

            except Exception as e:
                logger.error(f"Error extracting text from {filename}: {e}")
                raise

    def preprocess_documents(self) -> None:
        """
        Preprocess documents by removing punctuation and normalizing text.
        For single documents, splits into random chunks for better independence.
        """
        logger.info("Preprocessing documents")

        num_files = len(self.filenames)

        if num_files == 1:
            # Single file: split into random chunks for better independence
            with open("output_0.txt", 'r', encoding='utf-8') as f:
                text = f.read().lower()
                words = self._clean_text(text)

                # Shuffle and split into chunks for better statistical independence
                chunk_size = len(words) // Config.SINGLE_DOC_CHUNKS

                if chunk_size < 100:
                    # If chunks would be too small, just split in half
                    logger.warning("Single document too small for chunking, using simple split")
                    mid = len(words) // 2
                    self.documents = [words[:mid], words[mid:]]
                else:
                    # Create random chunks (shuffle indices, then split)
                    indices = list(range(len(words)))
                    random.shuffle(indices)

                    self.documents = []
                    for i in range(Config.SINGLE_DOC_CHUNKS):
                        start_idx = i * chunk_size
                        end_idx = start_idx + chunk_size if i < Config.SINGLE_DOC_CHUNKS - 1 else len(indices)
                        chunk_indices = sorted(indices[start_idx:end_idx])
                        chunk = [words[idx] for idx in chunk_indices]
                        self.documents.append(chunk)

                self.merged_words = words
                logger.info(f"Split single document into {len(self.documents)} chunks")

        elif num_files >= 2:
            # Multiple files: process each separately
            for idx in range(num_files):
                with open(f"output_{idx}.txt", 'r', encoding='utf-8') as f:
                    text = f.read().lower()
                    words = self._clean_text(text)
                    self.documents.append(words)

            # Merge all words
            self.merged_words = [word for doc in self.documents for word in doc]

        logger.info(f"Preprocessed {len(self.documents)} documents with {len(self.merged_words)} total words")

    def _clean_text(self, text: str) -> List[str]:
        """
        Clean text by removing punctuation and filtering words.

        Args:
            text: Input text string

        Returns:
            List of cleaned words
        """
        # Remove punctuation
        translator = str.maketrans('', '', Config.PUNCTUATION)
        text = text.translate(translator)

        # Remove non-Yoruba characters (keep only valid Yoruba characters)
        pattern = f"[^{Config.YORUBA_CHARACTERS}\\s]+"
        text = re.sub(pattern, '', text)

        # Split into words and filter
        words = text.split()

        # Filter by length and non-digit
        filtered_words = [
            word for word in words
            if self.min_word_length <= len(word) <= self.max_word_length
            and not word.isdigit()
        ]

        return filtered_words

    def calculate_frequencies(self) -> None:
        """Calculate word frequencies across all documents."""
        logger.info("Calculating word frequencies")

        self.word_frequencies = Counter(self.merged_words)
        logger.debug(f"Found {len(self.word_frequencies)} unique words")

    def calculate_tfidf(self) -> None:
        """
        Calculate TF-IDF scores using scikit-learn.
        For stopword detection, we want LOW TF-IDF scores (common words).
        """
        logger.info("Calculating TF-IDF scores")

        try:
            # Prepare documents as strings
            doc_strings = [' '.join(doc) for doc in self.documents]

            # Create TF-IDF vectorizer - NO max_df to include all words
            vectorizer = TfidfVectorizer(
                token_pattern=f"[{Config.YORUBA_CHARACTERS}]+",
                min_df=1,
                # Removed max_df to include high-frequency words
            )

            # Fit and transform
            tfidf_matrix = vectorizer.fit_transform(doc_strings)
            feature_names = vectorizer.get_feature_names_out()

            # Get average TF-IDF scores across documents
            avg_tfidf = np.mean(tfidf_matrix.toarray(), axis=0)

            # Store scores
            self.tfidf_scores = dict(zip(feature_names, avg_tfidf))

            logger.debug(f"Calculated TF-IDF for {len(self.tfidf_scores)} words")

        except Exception as e:
            logger.warning(f"Scikit-learn TF-IDF failed, using manual calculation: {e}")
            self._calculate_tfidf_manual()

    def _calculate_tfidf_manual(self) -> None:
        """Manual TF-IDF calculation as fallback."""
        # Calculate TF for each document
        doc_tfs = []
        for doc in self.documents:
            doc_counter = Counter(doc)
            doc_length = len(doc)
            tf = {word: count / doc_length for word, count in doc_counter.items()}
            doc_tfs.append(tf)

        # Calculate IDF
        num_docs = len(self.documents)
        idf = {}

        for word in self.word_frequencies:
            # Count documents containing the word
            doc_count = sum(1 for doc_tf in doc_tfs if word in doc_tf)
            if doc_count > 0:
                idf[word] = math.log(num_docs / doc_count)

        # Calculate TF-IDF (average across documents)
        for word in self.word_frequencies:
            if word in idf:
                tfidf_sum = sum(
                    doc_tf.get(word, 0) * idf[word]
                    for doc_tf in doc_tfs
                )
                self.tfidf_scores[word] = tfidf_sum / num_docs

    def calculate_entropy(self) -> None:
        """
        Calculate Shannon entropy for each word across documents.
        Higher entropy indicates more uniform distribution (typical of stopwords).
        Entropy measures how evenly a word is distributed across documents.
        """
        logger.info("Calculating entropy scores")

        for word, freq in self.word_frequencies.items():
            if freq >= self.min_frequency:
                # Count occurrences in each document
                doc_counts = [doc.count(word) for doc in self.documents]
                total = sum(doc_counts)

                if total > 0:
                    # Calculate Shannon entropy over document distribution
                    entropy = 0
                    for count in doc_counts:
                        if count > 0:
                            p = count / total
                            entropy -= p * math.log2(p)

                    # Normalize by max possible entropy (log2 of num documents)
                    max_entropy = math.log2(len(self.documents)) if len(self.documents) > 1 else 1
                    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

                    self.entropy_scores[word] = normalized_entropy

        logger.debug(f"Calculated entropy for {len(self.entropy_scores)} words")

    def calculate_variance(self) -> None:
        """
        Calculate variance of normalized word frequencies across documents.
        Low variance indicates consistent usage (typical of stopwords).
        Variance measures how much a word's relative frequency varies across documents.
        """
        logger.info("Calculating variance scores")

        for word, freq in self.word_frequencies.items():
            if freq >= self.min_frequency:
                # Get normalized frequency in each document
                doc_freqs = []
                for doc in self.documents:
                    doc_counter = Counter(doc)
                    doc_length = len(doc)
                    # Normalize by document length to get relative frequency
                    normalized_freq = (doc_counter.get(word, 0) / doc_length) if doc_length > 0 else 0
                    doc_freqs.append(normalized_freq)

                # Calculate variance across documents
                if len(doc_freqs) > 1:
                    variance = np.var(doc_freqs)
                    self.variance_scores[word] = variance
                else:
                    # Single document case
                    self.variance_scores[word] = 0

        logger.debug(f"Calculated variance for {len(self.variance_scores)} words")

    def calculate_dispersion(self) -> None:
        """
        Calculate Gries' Deviation of Proportions (DP) for each word.
        Low DP indicates even distribution across documents (typical of stopwords).
        DP ranges from 0 (perfectly even) to 1 (maximally uneven).
        """
        logger.info("Calculating dispersion scores")

        num_docs = len(self.documents)
        total_words = len(self.merged_words)

        for word, freq in self.word_frequencies.items():
            if freq >= self.min_frequency:
                # Calculate expected proportion in each document
                doc_lengths = [len(doc) for doc in self.documents]

                # Count actual occurrences in each document
                doc_counts = [doc.count(word) for doc in self.documents]

                # Calculate dispersion (Gries' DP)
                if total_words > 0 and freq > 0:
                    dp = 0
                    for i in range(num_docs):
                        # Expected proportion if evenly distributed
                        expected_prop = doc_lengths[i] / total_words
                        # Actual proportion
                        actual_prop = doc_counts[i] / freq
                        # Sum of absolute differences
                        dp += abs(actual_prop - expected_prop)

                    # Normalize by dividing by 2 (max possible value)
                    dp = dp / 2
                    self.dispersion_scores[word] = dp
                else:
                    self.dispersion_scores[word] = 1.0  # Max dispersion for edge cases

        logger.debug(f"Calculated dispersion for {len(self.dispersion_scores)} words")

    def filter_by_function_words(self) -> None:
        """
        Filter words based on Yoruba linguistic patterns.
        Words matching known function word patterns are candidates for stopwords.
        """
        logger.info("Applying linguistic filtering")

        for word in self.word_frequencies:
            if word in Config.YORUBA_FUNCTION_WORDS:
                self.function_word_candidates.add(word)

        logger.debug(f"Found {len(self.function_word_candidates)} function word candidates")

    def filter_by_frequency(self) -> None:
        """
        Filter words by high frequency using percentile-based thresholding.
        Selects words in the top N% by frequency (typical stopwords).
        """
        logger.info("Filtering by frequency")

        if not self.word_frequencies:
            logger.warning("No word frequencies available")
            return

        # Use percentile-based threshold (more robust across different corpus sizes)
        frequencies = list(self.word_frequencies.values())
        threshold_freq = np.percentile(frequencies, Config.FREQUENCY_PERCENTILE)

        for word, freq in self.word_frequencies.items():
            if freq >= self.min_frequency and freq >= threshold_freq:
                self.high_frequency_words.add(word)

        logger.debug(f"Found {len(self.high_frequency_words)} high-frequency words "
                    f"(threshold: {threshold_freq:.2f}, {Config.FREQUENCY_PERCENTILE}th percentile)")

    def filter_by_tfidf(self) -> None:
        """
        Filter words by TF-IDF scores.
        Stopwords have LOW TF-IDF scores because they appear frequently
        across all documents (high TF, low IDF).

        Note: For single documents split into chunks, TF-IDF is less reliable
        and we use frequency as a proxy.
        """
        logger.info("Filtering by TF-IDF")

        if not self.tfidf_scores:
            logger.warning("No TF-IDF scores available")
            return

        # Special handling for single documents split into chunks
        # TF-IDF doesn't work well when all chunks come from same document
        if len(self.filenames) == 1 and len(self.tfidf_scores) > 0:
            # Check if TF-IDF is actually useful (not all zeros or uniform)
            tfidf_values = list(self.tfidf_scores.values())
            tfidf_range = max(tfidf_values) - min(tfidf_values)

            if tfidf_range < 0.001:  # Nearly uniform TF-IDF scores
                logger.warning("Single document detected: TF-IDF scores are uniform, using frequency-based proxy")
                # Use high-frequency words as proxy for low TF-IDF
                self.low_tfidf_words = self.high_frequency_words.copy()
                logger.debug(f"Using {len(self.low_tfidf_words)} high-frequency words as TF-IDF proxy")
                return

        # Normal TF-IDF filtering for multiple documents
        tfidf_values = list(self.tfidf_scores.values())
        threshold_tfidf = np.percentile(tfidf_values, Config.TFIDF_PERCENTILE)

        for word, score in self.tfidf_scores.items():
            # Include words with LOW TF-IDF (bottom percentile)
            if score <= threshold_tfidf and self.word_frequencies.get(word, 0) >= self.min_frequency:
                self.low_tfidf_words.add(word)

        logger.debug(f"Found {len(self.low_tfidf_words)} low TF-IDF words "
                    f"(threshold: {threshold_tfidf:.6f}, {Config.TFIDF_PERCENTILE}th percentile)")

    def filter_by_entropy(self) -> None:
        """
        Filter words by entropy.
        High entropy indicates uniform distribution across documents (typical of stopwords).
        """
        logger.info("Filtering by entropy")

        if not self.entropy_scores:
            logger.warning("No entropy scores available")
            return

        # Stopwords have HIGH entropy (evenly distributed across documents)
        entropy_values = list(self.entropy_scores.values())
        threshold_entropy = np.percentile(entropy_values, Config.ENTROPY_PERCENTILE)

        for word, entropy in self.entropy_scores.items():
            # Include words with HIGH entropy (top percentile)
            if entropy >= threshold_entropy and self.word_frequencies.get(word, 0) >= self.min_frequency:
                self.high_entropy_words.add(word)

        logger.debug(f"Found {len(self.high_entropy_words)} high-entropy words "
                    f"(threshold: {threshold_entropy:.4f}, {Config.ENTROPY_PERCENTILE}th percentile)")

    def filter_by_variance(self) -> None:
        """
        Filter words with low variance.
        Low variance indicates consistent usage across documents (typical of stopwords).
        """
        logger.info("Filtering by variance")

        if not self.variance_scores:
            logger.warning("No variance scores available")
            return

        # Stopwords have LOW variance (consistent relative frequency across documents)
        variance_values = list(self.variance_scores.values())
        threshold_variance = np.percentile(variance_values, Config.VARIANCE_PERCENTILE)

        for word, variance in self.variance_scores.items():
            # Include words with LOW variance (bottom percentile)
            if variance <= threshold_variance:
                self.low_variance_words.add(word)

        logger.debug(f"Found {len(self.low_variance_words)} low-variance words "
                    f"(threshold: {threshold_variance:.6f}, {Config.VARIANCE_PERCENTILE}th percentile)")

    def filter_by_dispersion(self) -> None:
        """
        Filter words by dispersion.
        Low dispersion indicates even distribution across documents (typical of stopwords).
        """
        logger.info("Filtering by dispersion")

        if not self.dispersion_scores:
            logger.warning("No dispersion scores available")
            return

        # Stopwords have LOW dispersion (evenly distributed across documents)
        dispersion_values = list(self.dispersion_scores.values())
        threshold_dispersion = np.percentile(dispersion_values, Config.DISPERSION_PERCENTILE)

        for word, dispersion in self.dispersion_scores.items():
            # Include words with LOW dispersion (bottom percentile)
            # Note: Lower DP = more even distribution
            if dispersion <= threshold_dispersion:
                self.high_dispersion_words.add(word)

        logger.debug(f"Found {len(self.high_dispersion_words)} low-dispersion words "
                    f"(threshold: {threshold_dispersion:.4f}, {Config.DISPERSION_PERCENTILE}th percentile)")

    def combine_filters(self) -> None:
        """
        Combine filters using weighted voting system.
        Words must pass multiple filters to be considered stopwords.
        Priority: Linguistic patterns > Statistical consensus
        """
        logger.info("Combining filters")

        # Statistical filters (each word gets a vote for each filter it passes)
        statistical_votes = {}

        for word in self.word_frequencies:
            votes = 0

            if word in self.high_frequency_words:
                votes += 1
            if word in self.low_tfidf_words:
                votes += 1
            if word in self.high_entropy_words:
                votes += 1
            if word in self.low_variance_words:
                votes += 1
            if word in self.high_dispersion_words:
                votes += 1

            if votes > 0:
                statistical_votes[word] = votes

        # Combine with linguistic filter
        # REVISED STRATEGY (prioritizes precision over recall):
        # 1. Linguistic match with ANY statistical support (1+ votes), OR
        # 2. Very strong statistical consensus (4+ out of 5 filters)
        #
        # Rationale: Statistical methods alone cannot distinguish frequent content words
        # from stopwords. Linguistic knowledge is essential for high precision.

        candidates = set()

        for word, votes in statistical_votes.items():
            # Priority 1: Known function words with minimal statistical validation
            if word in self.function_word_candidates and votes >= 1:
                candidates.add(word)
            # Priority 2: Overwhelming statistical evidence (4+ filters agree)
            elif votes >= 4:
                candidates.add(word)

        self.stopwords = candidates

        logger.info(f"Final stopwords: {len(self.stopwords)} words")
        logger.info(f"  - High frequency words: {len(self.high_frequency_words)}")
        logger.info(f"  - Low TF-IDF words: {len(self.low_tfidf_words)}")
        logger.info(f"  - High entropy words: {len(self.high_entropy_words)}")
        logger.info(f"  - Low variance words: {len(self.low_variance_words)}")
        logger.info(f"  - Low dispersion words: {len(self.high_dispersion_words)}")
        logger.info(f"  - Function word candidates: {len(self.function_word_candidates)}")
        logger.info(f"  - Final stopwords: {len(self.stopwords)}")

    def get_stopwords(self) -> Set[str]:
        """Return the set of identified stopwords."""
        return self.stopwords

    def get_statistics(self) -> Dict:
        """
        Get detailed statistics about the analysis.

        Returns:
            Dictionary containing analysis statistics
        """
        return {
            'total_documents': len(self.documents),
            'total_words': len(self.merged_words),
            'unique_words': len(self.word_frequencies),
            'filter_results': {
                'high_frequency_words': len(self.high_frequency_words),
                'low_tfidf_words': len(self.low_tfidf_words),
                'high_entropy_words': len(self.high_entropy_words),
                'low_variance_words': len(self.low_variance_words),
                'low_dispersion_words': len(self.high_dispersion_words),
                'function_word_candidates': len(self.function_word_candidates),
            },
            'final_stopwords': len(self.stopwords),
            'top_10_frequencies': dict(self.word_frequencies.most_common(10)),
            'thresholds': {
                'frequency_percentile': Config.FREQUENCY_PERCENTILE,
                'tfidf_percentile': Config.TFIDF_PERCENTILE,
                'entropy_percentile': Config.ENTROPY_PERCENTILE,
                'variance_percentile': Config.VARIANCE_PERCENTILE,
                'dispersion_percentile': Config.DISPERSION_PERCENTILE,
            }
        }

    def save_stopwords(
        self,
        output_path: Path,
        formats: List[str] = ['txt', 'json']
    ) -> Dict[str, Path]:
        """
        Save stopwords in multiple formats.

        Args:
            output_path: Base path for output files
            formats: List of formats to save ('txt', 'json', 'csv')

        Returns:
            Dictionary mapping format to file path
        """
        saved_files = {}

        output_path = Path(output_path)
        base_name = output_path.stem
        output_dir = output_path.parent

        # Save as TXT
        if 'txt' in formats:
            txt_path = output_dir / f"{base_name}.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                for word in sorted(self.stopwords):
                    f.write(f"{word}\n")
            saved_files['txt'] = txt_path
            logger.info(f"Saved stopwords to {txt_path}")

        # Save as JSON with statistics
        if 'json' in formats:
            json_path = output_dir / f"{base_name}.json"
            data = {
                'stopwords': sorted(list(self.stopwords)),
                'statistics': self.get_statistics(),
                'metadata': {
                    'min_word_length': self.min_word_length,
                    'max_word_length': self.max_word_length,
                    'min_frequency': self.min_frequency
                }
            }
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            saved_files['json'] = json_path
            logger.info(f"Saved stopwords with stats to {json_path}")

        # Save as CSV
        if 'csv' in formats:
            csv_path = output_dir / f"{base_name}.csv"
            with open(csv_path, 'w', encoding='utf-8') as f:
                f.write("word,frequency,tfidf_score\n")
                for word in sorted(self.stopwords):
                    freq = self.word_frequencies.get(word, 0)
                    tfidf = self.tfidf_scores.get(word, 0.0)
                    f.write(f"{word},{freq},{tfidf:.6f}\n")
            saved_files['csv'] = csv_path
            logger.info(f"Saved stopwords with scores to {csv_path}")

        return saved_files
