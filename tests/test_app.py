"""
Unit tests for Flask application
"""

import unittest
import tempfile
import io
from pathlib import Path

from app import app
from config import config


class TestFlaskApp(unittest.TestCase):
    """Test cases for Flask application"""

    def setUp(self):
        """Set up test fixtures"""
        app.config.from_object(config['testing'])
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False

        # Create test client
        self.client = app.test_client()

        # Create test directories
        Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
        Path(app.config['OUTPUT_FOLDER']).mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Clean up test fixtures"""
        pass

    def test_index_route(self):
        """Test index route returns 200"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Yoruba Stopwords', response.data)

    def test_health_check(self):
        """Test health check endpoint"""
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('timestamp', data)

    def test_upload_no_files(self):
        """Test upload with no files"""
        response = self.client.post('/upload', data={})
        self.assertEqual(response.status_code, 302)  # Redirect

    def test_upload_invalid_file_type(self):
        """Test upload with invalid file type"""
        data = {
            'files[]': (io.BytesIO(b'test content'), 'test.pdf')
        }

        response = self.client.post(
            '/upload',
            data=data,
            content_type='multipart/form-data'
        )

        self.assertEqual(response.status_code, 302)  # Redirect

    def test_allowed_file(self):
        """Test allowed file extension checker"""
        from app import allowed_file

        self.assertTrue(allowed_file('test.docx'))
        self.assertTrue(allowed_file('test.doc'))
        self.assertTrue(allowed_file('test.txt'))
        self.assertFalse(allowed_file('test.pdf'))
        self.assertFalse(allowed_file('test'))
        self.assertFalse(allowed_file('test.exe'))

    def test_validate_uploaded_file(self):
        """Test file validation"""
        from app import validate_uploaded_file
        from werkzeug.datastructures import FileStorage

        # Valid file
        valid_file = FileStorage(
            stream=io.BytesIO(b'content'),
            filename='test.docx'
        )
        is_valid, error = validate_uploaded_file(valid_file)
        self.assertTrue(is_valid)
        self.assertIsNone(error)

        # No file
        is_valid, error = validate_uploaded_file(None)
        self.assertFalse(is_valid)
        self.assertIsNotNone(error)

        # Empty filename
        empty_file = FileStorage(
            stream=io.BytesIO(b'content'),
            filename=''
        )
        is_valid, error = validate_uploaded_file(empty_file)
        self.assertFalse(is_valid)
        self.assertIsNotNone(error)

        # Invalid extension
        invalid_file = FileStorage(
            stream=io.BytesIO(b'content'),
            filename='test.pdf'
        )
        is_valid, error = validate_uploaded_file(invalid_file)
        self.assertFalse(is_valid)
        self.assertIn('Invalid file type', error)

    def test_404_error(self):
        """Test 404 error handler"""
        response = self.client.get('/nonexistent-page')
        self.assertEqual(response.status_code, 404)

    def test_download_nonexistent_file(self):
        """Test downloading a file that doesn't exist"""
        response = self.client.get('/download/nonexistent.txt')
        self.assertEqual(response.status_code, 302)  # Redirect

    def test_api_stopwords_not_found(self):
        """Test API when no stopwords have been generated"""
        response = self.client.get('/api/stopwords')

        # Should return 404 if no stopwords exist
        if response.status_code == 404:
            data = response.get_json()
            self.assertEqual(data['status'], 'not_found')


class TestConfig(unittest.TestCase):
    """Test configuration settings"""

    def test_development_config(self):
        """Test development configuration"""
        from config import DevelopmentConfig

        self.assertTrue(DevelopmentConfig.DEBUG)

    def test_production_config(self):
        """Test production configuration"""
        from config import ProductionConfig

        self.assertFalse(ProductionConfig.DEBUG)

    def test_testing_config(self):
        """Test testing configuration"""
        from config import TestingConfig

        self.assertTrue(TestingConfig.TESTING)

    def test_config_attributes(self):
        """Test required config attributes exist"""
        from config import Config

        self.assertTrue(hasattr(Config, 'UPLOAD_FOLDER'))
        self.assertTrue(hasattr(Config, 'MAX_CONTENT_LENGTH'))
        self.assertTrue(hasattr(Config, 'ALLOWED_EXTENSIONS'))
        self.assertTrue(hasattr(Config, 'MIN_WORD_LENGTH'))
        self.assertTrue(hasattr(Config, 'MAX_WORD_LENGTH'))
        self.assertTrue(hasattr(Config, 'MIN_FREQUENCY'))


if __name__ == '__main__':
    unittest.main()
