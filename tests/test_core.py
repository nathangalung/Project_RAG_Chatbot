import unittest
from core.llm import get_llm
from core.embeddings import get_embeddings

class TestCore(unittest.TestCase):
    def test_llm_initialization(self):
        try:
            llm = get_llm()
            self.assertIsNotNone(llm)
        except Exception as e:
            self.fail(f"LLM initialization failed with error: {str(e)}")
            
    def test_embeddings_initialization(self):
        try:
            embeddings = get_embeddings()
            self.assertIsNotNone(embeddings)
        except Exception as e:
            self.fail(f"Embeddings initialization failed with error: {str(e)}")

if __name__ == '__main__':
    unittest.main()