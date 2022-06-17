import unittest

from segwork.registry import ConfigurableRegistry

class Testregistry(unittest.TestCase):

    def setUp(self):
        # load test data
        self.registry = ConfigurableRegistry('model', )
        self.registry['test_entry'] = dict( model=str)

    def test_sum(self):
        self.assertEqual(type(self.registry[test_entry])), 6, "Should be 6")

    def test_sum_tuple(self):
        self.assertEqual(sum((1, 2, 2)), 6, "Should be 6")

if __name__ == '__main__':
    unittest.main()