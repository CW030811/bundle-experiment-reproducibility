import importlib.util
from pathlib import Path
import unittest


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/verify_reference_results.py'
        self.assertTrue(script.exists(), 'Public reference verifier is not implemented')
        spec = importlib.util.spec_from_file_location('verify_reference_results', script)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_numeric_comparison_rejects_nonfinite_values(self):
        self.assertFalse(self.module.equal_value('nan', 'nan'))
        self.assertFalse(self.module.equal_value('inf', 'inf'))
        self.assertTrue(self.module.equal_value('0.25', '0.250000'))


if __name__ == '__main__':
    unittest.main()
