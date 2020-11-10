import unittest
import mockito
import builtins
import io
import numpy as np

from Implementation.classifiers.custom_classifier import CustomClassifier


class MyTestCase(unittest.TestCase):

    @staticmethod
    def construct_open_result(str_list):
        open_result = io.TextIOWrapper(io.BytesIO(), line_buffering=True)

        for elem in str_list:
            open_result.write(elem + "\n")

        open_result.seek(0, 0)

        return open_result

    def test_load_data_when_having_valid_data(self):
        # INIT
        open_result = self.construct_open_result([
            '{"id":"0", "label":"1", "text":"some text 1"}',
            '{"id":"1", "label":"0", "text":"some text 2"}',
            '{"id":"2", "label":"1", "text":"some text 3"}'
        ])

        # when calling open return a bytes representation of a json files
        mockito.when(builtins).open(mockito.any(object), mockito.any(object)).thenReturn(open_result)

        expected_data = {"id": np.array([0, 1, 2]),
                         "label": np.array([1, 0, 1]),
                         "text": np.array(["some text 1", "some text 2", "some text 3"])}

        # EXECUTE
        obj = CustomClassifier()
        obj.load_data("random string")
        actual_data = obj.data

        # VERIFY
        self.assertEqual(type(actual_data), dict)
        self.assertEqual(actual_data.keys(), expected_data.keys())
        self.assertTrue((actual_data["id"] == expected_data["id"]).all())
        self.assertTrue((actual_data["label"] == expected_data["label"]).all())
        self.assertTrue((actual_data["text"] == expected_data["text"]).all())

    def test_load_data_when_having_invalid_data(self):
        # INIT
        open_result = self.construct_open_result([
            'random text',
            'more random text'
        ])

        # when calling open return a bytes representation of a json files
        mockito.when(builtins).open(mockito.any(object), mockito.any(object)).thenReturn(open_result)

        # EXECUTE and VERIFY
        obj = CustomClassifier()
        try:
            obj.load_data("random string")
        except Exception as e:
            return None

        self.fail("Exception not thrown when using invalid data!")


if __name__ == '__main__':
    unittest.main()
