"""
Configuration settings for Yoruba Stopwords Generator
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.absolute()

# Flask settings
class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'

    # Upload settings
    UPLOAD_FOLDER = BASE_DIR / 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'docx', 'doc', 'txt'}

    # Processing settings
    MIN_WORD_LENGTH = 2
    MAX_WORD_LENGTH = 5
    MIN_FREQUENCY = 5

    # Statistical thresholds (percentile-based)
    FREQUENCY_PERCENTILE = 98  # Top 2% most frequent words
    TFIDF_PERCENTILE = 25      # Bottom 25% (low TF-IDF = stopwords)
    ENTROPY_PERCENTILE = 75    # Top 25% (high entropy = uniform distribution)
    VARIANCE_PERCENTILE = 25   # Bottom 25% (low variance = consistent usage)
    DISPERSION_PERCENTILE = 75 # Top 25% (high dispersion = evenly distributed)

    # Document splitting for single file
    MIN_DOCUMENTS_REQUIRED = 2
    SINGLE_DOC_CHUNKS = 4      # Split single doc into N chunks

    # Output settings
    OUTPUT_FOLDER = BASE_DIR / 'uploads'
    OUTPUT_FORMAT = ['txt', 'docx', 'json', 'csv']

    # Server settings
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', 5001))
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'

    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = BASE_DIR / 'logs' / 'app.log'

    # Yoruba language specific
    YORUBA_CHARACTERS = 'a-zA-ZÀÁÈÉẸẸ̀Ẹ́Ọ̀ỌỌ́ÙÓÚÒṢàùÌÍèọ̀èáéọòẹẹ́ìíọ́óúẹ̀ṣ'
    PUNCTUATION = '!.,?);(:{}[]*0123456789-—–\'\"'

    # Yoruba function words (linguistic filtering)
    # Core grammatical words that function as stopwords across all domains
    YORUBA_FUNCTION_WORDS = {
        # Pronouns (subject and object forms)
        'mo', 'mi', 'o', 'ó', 'a', 'ẹ', 'wọn', 'wọ́n',
        'ẹmi', 'èmi', 'emi',           # I/me
        'iwo', 'ìwọ', 'iwọ',            # you
        'òun', 'oun',                   # he/she/it
        'àwa', 'awa',                   # we
        'ẹyin',                         # you (plural)
        'àwọn', 'awọn',                 # they/them

        # Conjunctions
        'àti', 'ati',                   # and
        'tàbí', 'tabi',                 # or
        'ṣùgbọ́n', 'sugbon',             # but
        'bí', 'bi',                     # if/like
        'pé', 'pe',                     # that
        'tí', 'ti',                     # that/which
        'nítorí', 'nitori', 'torí', 'tori',  # because

        # Prepositions
        'ní', 'ni',                     # in/at/on
        'sí', 'si',                     # to/at
        'láti', 'lati',                 # from
        'fún', 'fun',                   # for
        'nínú', 'ninu',                 # inside/in
        'lórí', 'lori',                 # on/about
        'nípa', 'nipa',                 # about/concerning
        'pẹ̀lú', 'pelu',                 # with
        'láìsí', 'laisi',               # without

        # Auxiliary verbs and modal particles
        'ti', 'tí',                     # have/perfective
        'máa', 'maa',                   # will/habitual
        'yóò', 'yoo',                   # will/future
        'lè', 'le',                     # can/able
        'kò', 'ko',                     # not/no
        'kó', 'kí', 'ki',               # should/let
        'kì', 'kií', 'kii',             # never
        'ń', 'n',                       # progressive marker
        'sì', 'si',                     # and/also (very common!)

        # Copula verbs (be/is)
        'jẹ́', 'je', 'jé',               # be/is
        'ṣe', 'se',                     # do/is
        'ni',                           # is/copula

        # Highly grammaticalized movement verbs
        'wà', 'wa',                     # be/exist/be at
        'wá',                           # come
        'ló', 'lo', 'lọ',               # go

        # Essential particles
        'fi', 'fí',                     # with/use (instrumental)
        'di',                           # become
        'un', 'ún',                     # particle (emphasis)

        # Demonstratives
        'yìí', 'yii',                   # this
        'yẹn', 'yen',                   # that
        'náà', 'naa',                   # the/that
        'èyí', 'eyi',                   # this one

        # Determiners and quantifiers
        'kan',                          # one/a/an
        'gbogbo',                       # all/every
        'púpọ̀', 'pupo',                 # many/much

        # Question words
        'kí', 'ki',                     # what
        'kíni', 'kini',                 # what is
        'tani',                         # who
        'níbo', 'nibo',                 # where
        'báwo', 'bawo',                 # how
        'ìgbà', 'igba',                 # when/time

        # Possessive markers
        'rẹ', 'rẹ̀',                     # your/his/her/its
        'tèmi', 'tirẹ̀', 'tàwa', 'tiwọn',  # mine/yours/ours/theirs

        # Negation and affirmation
        'rárá', 'rara',                 # no/not at all
        'bẹ́ẹ̀', 'beee',                 # yes
        'béè', 'bee',                   # yes
        'bẹ́ẹ̀ni', 'bẹẹni',              # yes (formal)
    }

    @staticmethod
    def init_app(app):
        """Initialize application directories"""
        # Create necessary directories
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.OUTPUT_FOLDER, exist_ok=True)
        os.makedirs(Config.LOG_FILE.parent, exist_ok=True)


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    # Add production-specific settings


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    UPLOAD_FOLDER = BASE_DIR / 'tests' / 'uploads'
    OUTPUT_FOLDER = BASE_DIR / 'tests' / 'outputs'


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
