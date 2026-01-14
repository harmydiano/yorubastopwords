"""
Unit tests for TFIDFAnalyzer class
"""

import unittest
from pathlib import Path
import tempfile
import shutil
from collections import Counter

from tfidf_analyzer import TFIDFAnalyzer


class TestTFIDFAnalyzer(unittest.TestCase):
    """Test cases for TFIDFAnalyzer"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.sample_text = """
        Àwọn ẹranko tí wọ́n wà ní ilẹ̀ Áfríkà púpọ̀.
        Ilẹ̀ Áfríkà ni ilẹ̀ tí ó tóbi jù lọ.
        Ní ilẹ̀ yìí, a lè rí ẹranko bí i ekùn àti kìnnìún.
        """

    def tearDown(self):
        """Clean up test fixtures"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Test TFIDFAnalyzer initialization"""
        analyzer = TFIDFAnalyzer(
            filenames=['test1.docx'],
            min_word_length=2,
            max_word_length=5,
            min_frequency=3
        )

        self.assertEqual(len(analyzer.filenames), 1)
        self.assertEqual(analyzer.min_word_length, 2)
        self.assertEqual(analyzer.max_word_length, 5)
        self.assertEqual(analyzer.min_frequency, 3)

    def test_clean_text(self):
        """Test text cleaning functionality"""
        analyzer = TFIDFAnalyzer(filenames=['test.docx'])

        text = "Àwọn ẹranko!! Ní ilẹ̀ 123 yìí."
        cleaned = analyzer._clean_text(text)

        # Should remove punctuation and numbers
        self.assertNotIn('!!', ' '.join(cleaned))
        self.assertNotIn('123', cleaned)

        # Should preserve Yoruba characters
        self.assertIn('àwọn', cleaned)
        self.assertIn('ẹranko', cleaned)

    def test_word_length_filter(self):
        """Test word length filtering"""
        analyzer = TFIDFAnalyzer(
            filenames=['test.docx'],
            min_word_length=3,
            max_word_length=5
        )

        text = "a bb ccc dddd eeeee ffffff"
        cleaned = analyzer._clean_text(text)

        # Should only include words of length 3-5
        self.assertNotIn('a', cleaned)
        self.assertNotIn('bb', cleaned)
        self.assertIn('ccc', cleaned)
        self.assertIn('dddd', cleaned)
        self.assertIn('eeeee', cleaned)
        self.assertNotIn('ffffff', cleaned)

    def test_calculate_frequencies(self):
        """Test frequency calculation"""
        analyzer = TFIDFAnalyzer(filenames=['test.docx'])
        analyzer.merged_words = ['ilẹ̀', 'ilẹ̀', 'ilẹ̀', 'ní', 'ní', 'wà']

        analyzer.calculate_frequencies()

        self.assertEqual(analyzer.word_frequencies['ilẹ̀'], 3)
        self.assertEqual(analyzer.word_frequencies['ní'], 2)
        self.assertEqual(analyzer.word_frequencies['wà'], 1)

    def test_calculate_entropy(self):
        """Test entropy calculation"""
        analyzer = TFIDFAnalyzer(
            filenames=['test.docx'],
            min_frequency=1
        )
        analyzer.merged_words = ['ilẹ̀'] * 10 + ['ní'] * 5
        analyzer.word_frequencies = Counter(analyzer.merged_words)

        analyzer.calculate_entropy()

        self.assertIn('ilẹ̀', analyzer.entropy_scores)
        self.assertIn('ní', analyzer.entropy_scores)
        self.assertGreater(analyzer.entropy_scores['ilẹ̀'], 0)

    def test_calculate_variance(self):
        """Test variance calculation"""
        analyzer = TFIDFAnalyzer(
            filenames=['test.docx'],
            min_frequency=1
        )
        analyzer.merged_words = ['ilẹ̀'] * 10 + ['ní'] * 5 + ['wà'] * 2
        analyzer.word_frequencies = Counter(analyzer.merged_words)

        analyzer.calculate_variance()

        self.assertIn('ilẹ̀', analyzer.variance_scores)
        self.assertGreater(len(analyzer.variance_scores), 0)

    def test_get_statistics(self):
        """Test statistics generation"""
        analyzer = TFIDFAnalyzer(filenames=['test.docx'])
        analyzer.merged_words = ['ilẹ̀', 'ní', 'wà'] * 10
        analyzer.word_frequencies = Counter(analyzer.merged_words)
        analyzer.documents = [['ilẹ̀', 'ní'], ['wà', 'ilẹ̀']]
        analyzer.stopwords = {'ilẹ̀', 'ní'}

        stats = analyzer.get_statistics()

        self.assertEqual(stats['total_documents'], 2)
        self.assertEqual(stats['total_words'], 30)
        self.assertEqual(stats['unique_words'], 3)
        self.assertEqual(stats['final_stopwords'], 2)

    def test_filter_by_frequency(self):
        """Test frequency filtering"""
        analyzer = TFIDFAnalyzer(
            filenames=['test.docx'],
            min_frequency=5
        )
        analyzer.word_frequencies = Counter({
            'ilẹ̀': 100,
            'ní': 50,
            'wà': 3
        })

        analyzer.filter_by_frequency()

        # Should include high frequency words
        self.assertIn('ilẹ̀', analyzer.high_frequency_words)

        # Should not include low frequency words
        self.assertNotIn('wà', analyzer.high_frequency_words)

    def test_combine_filters(self):
        """Test filter combination logic with weighted voting"""
        analyzer = TFIDFAnalyzer(filenames=['test.docx'])
        analyzer.word_frequencies = Counter({'ilẹ̀': 10, 'ní': 8, 'wà': 5, 'tí': 4, 'di': 3})

        # Set up different filter results
        analyzer.high_frequency_words = {'ilẹ̀', 'ní', 'wà', 'tí'}
        analyzer.low_tfidf_words = {'ilẹ̀', 'ní', 'wà'}
        analyzer.high_entropy_words = {'ilẹ̀', 'ní', 'wà'}
        analyzer.low_variance_words = {'ilẹ̀', 'ní', 'di'}
        analyzer.high_dispersion_words = {'ilẹ̀', 'ní'}
        analyzer.function_word_candidates = {'ní', 'wà'}

        analyzer.combine_filters()

        # 'ilẹ̀' should pass with 5/5 votes (strong statistical consensus)
        # 'ní' should pass with linguistic match + 5/5 votes
        # 'wà' should pass with linguistic match + 4/5 votes
        self.assertGreater(len(analyzer.stopwords), 0)
        self.assertIn('ilẹ̀', analyzer.stopwords)  # 5/5 statistical votes
        self.assertIn('ní', analyzer.stopwords)    # linguistic + 5/5 votes

    def test_save_stopwords_txt(self):
        """Test saving stopwords as TXT"""
        analyzer = TFIDFAnalyzer(filenames=['test.docx'])
        analyzer.stopwords = {'ilẹ̀', 'ní', 'wà'}
        analyzer.word_frequencies = Counter({'ilẹ̀': 10, 'ní': 5, 'wà': 3})
        analyzer.tfidf_scores = {'ilẹ̀': 0.5, 'ní': 0.3, 'wà': 0.2}

        output_path = self.test_dir / 'test_stopwords'
        saved_files = analyzer.save_stopwords(output_path, formats=['txt'])

        self.assertIn('txt', saved_files)
        self.assertTrue(saved_files['txt'].exists())

        # Read and verify content
        with open(saved_files['txt'], 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn('ilẹ̀', content)
            self.assertIn('ní', content)
            self.assertIn('wà', content)

    def test_save_stopwords_json(self):
        """Test saving stopwords as JSON"""
        import json

        analyzer = TFIDFAnalyzer(filenames=['test.docx'])
        analyzer.stopwords = {'ilẹ̀', 'ní'}
        analyzer.word_frequencies = Counter({'ilẹ̀': 10, 'ní': 5})
        analyzer.tfidf_scores = {'ilẹ̀': 0.5, 'ní': 0.3}
        analyzer.documents = [['ilẹ̀'], ['ní']]
        analyzer.merged_words = ['ilẹ̀', 'ní']

        output_path = self.test_dir / 'test_stopwords'
        saved_files = analyzer.save_stopwords(output_path, formats=['json'])

        self.assertIn('json', saved_files)
        self.assertTrue(saved_files['json'].exists())

        # Read and verify JSON structure
        with open(saved_files['json'], 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.assertIn('stopwords', data)
            self.assertIn('statistics', data)
            self.assertIn('metadata', data)


if __name__ == '__main__':
    unittest.main()
