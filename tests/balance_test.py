import unittest

import numpy as np


from segwork import NumpyPixelCounter

class Testregistry(unittest.TestCase):

    def setUp(self):
        self.num_classes = 3
        self.pixel_counter = NumpyPixelCounter(num_classes = self.num_classes)

    def test_initial_pixel_count(self):
        self.assertTrue(np.all(self.pixel_counter.pixel_count == 0), "Should be True")
        self.assertEqual(self.pixel_counter.pixel_count.size, self.num_classes, f"Should be {self.num_classes}")
    
    def test_initial_class_count(self):
        self.assertTrue(np.all(self.pixel_counter.class_count == 0), "Should be True")
        self.assertEqual(self.pixel_counter.class_count.size, self.num_classes, f"Should be {self.num_classes}")

    def test_update(self):

        label = np.array([[1,1],[2,0]], dtype=self.pixel_counter._dtype)
        self.pixel_counter.update(label)

        self.assertEqual(self.pixel_counter.pixel_count[1], 2, f"Should be 2")
        self.assertTrue(np.all(self.pixel_counter.class_count == 4), f"Should be True")

if __name__ == '__main__':
    unittest.main(verbosity=2)