import unittest

from segwork.registry import ConfigurableRegistry

class Testregistry(unittest.TestCase):

    def setUp(self):
        # load test data
        self.registry = ConfigurableRegistry('model', )
        self.registry['test_entry'] = dict( model=str)

    def test_getitem(self):
        self.assertEqual(self.registry['test_entry'].get('model'), str, "Should be 6")

if __name__ == '__main__':
    unittest.main()