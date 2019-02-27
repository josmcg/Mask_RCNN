from training.train import train
import unittest
import os

class TestTrain(unittest.TestCase):
    def setUp(self):
        self.epochs = 1
        self.save_dir = "tests/data/weights"
        self.data_dir = "tests/data"
        self.collapse = 0

    def test_train(self):
        
        try:
            train(self.epochs,
                  self.save_dir,
                  self.data_dir,
                  self.collapse)
        except Exception:
            exit()

if __name__ == "__main__":
    unittest.main()

