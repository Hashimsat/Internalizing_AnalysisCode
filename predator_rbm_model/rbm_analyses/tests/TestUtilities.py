import unittest

from rbm_analyses.utilities import compute_persprob


class TestUtilities(unittest.TestCase):
    """This class implements the unit test in order to test the utilities"""

    def test_compute_pers_prob(self):
        """This function test the perseveration function of the RBM"""

        # Test parameters that bring down perseveration to zero
        persprob = compute_persprob(-30, -1.5, 1)
        self.assertTrue(persprob < 1.0e-10)

        # Test parameters that lead to 0.5
        persprob = compute_persprob(0, 0, 10)
        self.assertEqual(persprob, 0.5)

        # Test parameters that lead to low perseveration probability
        persprob = compute_persprob(0, -0.1, 10)
        self.assertEqual(persprob, 0.2689414213699951, 6)

        # Test parameters that lead to low perseveration probability
        persprob = compute_persprob(20, -0.1, 10)
        self.assertEqual(persprob, 0.7310585786300049, 6)


# Run unit test
if __name__ == "__main__":
    unittest.main()
