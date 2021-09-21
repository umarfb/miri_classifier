import unittest
import rnntools
import numpy as np
import pandas as pd

# Create test data
src = pd.read_csv('../data/goto_train_srcs_2.csv').iloc[0]
lc = pd.read_csv('../data/goto_dataset_phot_2020-02-23.csv')
lc = lc[lc['source_id'] == src['goto_id']]

#print(src)
#print(list(lc))
print(lc['ra'].values[0])

class TestCreateIoVectors(unittest.TestCase):

    # Test vector output without time scaling
    def test_vector_noscale(self):

        vector_out, _ = rnntools.create_io_vectors(src, lc)
        vector_exp = np.array([
            [
             lc['mag'].values[i],
             lc['mag_err'].values[i],
             lc['jd'].values[i],
             src['ra'],
             src['dec'],
             1.0
            ]
             for i in range(len(lc))
        ])

        self.assertSequenceEqual(vector_out.tolist(), vector_exp.tolist())
    
    # Test label output without time scaling
    def test_label_noscale(self):

        _, label_out = rnntools.create_io_vectors(src, lc)
        label_exp = 'EB'

        self.assertEqual(label_out, label_exp)
    
    # Test vector output with time scaling
    def test_vector_scaled(self):

        vector_out, _ = rnntools.create_io_vectors(src, lc, scale_time=True)
        vector_exp = np.array([
            [
             lc['mag'].values[i],
             lc['mag_err'].values[i],
             lc['jd'].values[i] - min(lc['jd'].values),
             src['ra'],
             src['dec'],
             1.0
            ]
             for i in range(len(lc))
        ])

        self.assertSequenceEqual(vector_out.tolist(), vector_exp.tolist())
    
    # Test label output with time scaling
    def test_label_scaled(self):

        _, label_out = rnntools.create_io_vectors(src, lc, scale_time=True)
        label_exp = 'EB'

        self.assertEqual(label_out, label_exp)
    
class TestCatToNum(unittest.TestCase):

    def test_num_label(self):

        data = ['B', 'A', 'C', 'A']
        cat_labels = ['A', 'B', 'C']
        num_labels = np.array([1, 0, 2, 0])

        result = rnntools.cat_to_num(data, cat_labels)

        self.assertSequenceEqual(result.tolist(), num_labels.tolist())

class TestOneHotEncode(unittest.TestCase):

    def test_onehotencode(self):

        data = ['B', 'A', 'C', 'A']
        cat_labels = ['A', 'B', 'C']

        ohe_data = np.array([
            [0., 1., 0.],
            [1., 0., 0.],
            [0., 0., 1.],
            [1., 0., 0.]
        ])

        result = rnntools.one_hot_encode(data, cat_labels)

        self.assertSequenceEqual(result.tolist(), ohe_data.tolist())

class TestF1(unittest.TestCase):

    def test_f1(self):

        y_true = np.array([1., 0., 0., 1.])
        y_pred = np.array([1., 1., 0., 1.])

        result = rnntools.f1(y_true, y_pred)

        self.assertEqual(result.numpy().round(3), 0.8)

if __name__ == '__main__':
    unittest.main()