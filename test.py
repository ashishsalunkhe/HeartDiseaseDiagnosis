import unittest
from sklearn.externals import joblib
import os
import numpy as np
import pickle
import main_file 
class BasicTestCase(unittest.TestCase):

    def test_index(self):
        tester = app.test_client(self)
        response = tester.get('/', content_type='html/text')
        self.assertEqual(response.status_code, 404)

    def test_database(self):
        tester = os.path.exists("flaskr.db")
        self.assertTrue(tester)
        

import tempfile



class BasicTestCase(unittest.TestCase):

    def test_index(self):
        """Initial test: Ensure flask was set up correctly."""
        tester = main_file.app.test_client(self)
        response = tester.get('/', content_type='html/text')
        self.assertEqual(response.status_code, 200)

    def test_database(self):
        """Initial test: Ensure that the database exists."""
        tester = os.path.exists("flaskr.db")
        self.assertEqual(tester, True)




class TestModelResponse(unittest.TestCase):
    def test_true(self):
        x = np.array([67,1,2,152,212,0,0,150,0,0.8,1,0,3]).reshape(1, -1)

        scaler_path = os.path.join(os.path.dirname(__file__), 'models/scaler.pkl')
        scaler = None
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        x = scaler.transform(x)

        model_path = os.path.join(os.path.dirname(__file__), 'models/rfc.sav')
        clf = joblib.load(model_path)

        y = clf.predict(x)
        print(y)
        self.assertEqual(y, 1)

    def test_false(self):
        x = np.array([63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]).reshape(1, -1)

        scaler_path = os.path.join(os.path.dirname(__file__), 'models/scaler.pkl')
        scaler = None
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        x = scaler.transform(x)

        model_path = os.path.join(os.path.dirname(__file__), 'models/rfc.sav')
        clf = joblib.load(model_path)

        y = clf.predict(x)
        print(y)
        self.assertEqual(y, 1)


if __name__ == '__main__':
    unittest.main()
