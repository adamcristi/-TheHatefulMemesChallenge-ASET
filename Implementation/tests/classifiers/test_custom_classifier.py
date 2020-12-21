import unittest
import mockito
import builtins
import io
import numpy as np
from transformers import BertTokenizer
from mockito import ANY

from Implementation.classifiers.custom_classifier import CustomClassifier


class CustomClassifierTest(unittest.TestCase):

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
            '{"id":"0", "label":"1", "text":"some text 1", "img":"some text 1"}',
            '{"id":"1", "label":"0", "text":"some text 2", "img":"some text 2"}',
            '{"id":"2", "label":"1", "text":"some text 3", "img":"some text 3"}'
        ])

        # when calling open return a bytes representation of a json files
        mockito.when(builtins).open(ANY, ANY).thenReturn(open_result)

        expected_data = {"id": np.array([0, 1, 2]),
                         "label": np.array([1, 0, 1]),
                         "text": np.array(["some text 1", "some text 2", "some text 3"]),
                         "img": np.array(["some text 1", "some text 2", "some text 3"])}

        obj = CustomClassifier()

        # EXECUTE
        obj.load_data("random string")
        actual_data = obj.data

        # VERIFY
        self.assertEqual(type(actual_data), dict)
        self.assertEqual(actual_data.keys(), expected_data.keys())
        self.assertTrue((actual_data["id"] == expected_data["id"]).all())
        self.assertTrue((actual_data["label"] == expected_data["label"]).all())
        self.assertTrue((actual_data["text"] == expected_data["text"]).all())
        self.assertTrue((actual_data["img"] == expected_data["img"]).all())

    def test_load_data_when_having_invalid_data(self):
        # INIT
        open_result = self.construct_open_result([
            'random text',
            'more random text'
        ])

        # when calling open return a bytes representation of a json files
        mockito.when(builtins).open(ANY, ANY).thenReturn(open_result)

        obj = CustomClassifier()

        # EXECUTE and VERIFY
        try:
            obj.load_data("random string")
        except Exception as e:
            return None

        self.fail("Exception not thrown when using invalid data!")

    def test_preprocess_when_having_valid_data(self):
        # INIT
        obj = CustomClassifier()
        obj.data = {
            "id": [0, 1, 2],
            "label": [0, 1, 0],
            "text": ["text1", "text2", "text3"],
            "img": ["img1", "img2", "img3"]
        }

        tokenizer_mock = mockito.mock(BertTokenizer)
        mockito.when(BertTokenizer).from_pretrained(...).thenReturn(tokenizer_mock)
        mockito.when(tokenizer_mock).encode("text1", ...).thenReturn([])
        mockito.when(tokenizer_mock).encode("text2", ...).thenReturn([0, 2, 3])
        mockito.when(tokenizer_mock).encode("text3", ...).thenReturn([4])

        temp_dict1 = {"input": 0, "attention_mask": 1}
        temp_dict2 = {"input": 2, "attention_mask": 3}
        temp_dict3 = {"input": 4, "attention_mask": 5}

        mockito.when(tokenizer_mock).encode_plus("text1", ...).thenReturn(temp_dict1)
        mockito.when(tokenizer_mock).encode_plus("text2", ...).thenReturn(temp_dict2)
        mockito.when(tokenizer_mock).encode_plus("text3", ...).thenReturn(temp_dict3)

        mockito.when(torch).cat([0, 2, 4], ...).thenReturn("interaction1")
        mockito.when(torch).cat([1, 3, 5], ...).thenReturn("interaction2")
        mockito.when(torch).tensor(obj.data["label"], ...).thenReturn("interaction3")

        # EXECUTE
        obj.preprocess()
        actual_data = obj.data

        # VERIFY
        self.assertEqual(actual_data["encoded_text"], "interaction1")
        self.assertEqual(actual_data["attention_masks"], "interaction2")
        self.assertEqual(actual_data["label"], "interaction3")

    def test_preprocess_when_having_invalid_data(self):
        test_cases = [
            ["list"],

            None,

            {
                "id": [0, 1],
                "label": [0, 1, 0],
                "text": ["different", "lengths"],
                "img": ["img1", "img2"]
            },

            {
                "id": [0, 1],
                "text": ["missing", "key"]
            },
        ]

        for tc in test_cases:
            # INIT
            obj = CustomClassifier()
            obj.data = tc

            # EXECUTE and VERIFY
            try:
                obj.preprocess()

            except Exception as e:
                if str(e) != "Cannot preprocess data with that format!":
                    self.fail("Unexpected exception thrown : " + str(e))
                return None

            self.fail("Exception not thrown when using invalid data! Failed test case: " + str(tc))


if __name__ == '__main__':
    unittest.main()
