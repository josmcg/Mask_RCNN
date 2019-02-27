from training.train import train
import unittest

class TestTrain(unittest.TestCase):
    def setUp(self):
        self.epochs = 1
        self.save_dir = "data/weights"
        self.data_dir = "data"
        self.collapse = 0

    def test_train(self):
        train(self.epochs,
              self.dave_dir,
              self.data_dir,
              self.collapse)
