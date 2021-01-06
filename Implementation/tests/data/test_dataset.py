import unittest

from Implementation.data.dataset import Dataset


class DatasetTest(unittest.TestCase):

    def setUp(self) -> None:
        self.dataset_object = Dataset()

    def test_received_train_dataset(self):
        # verify read dataframe columns
        official_columns = ["id", "img", "label", "text"]
        columns_from_dataset = self.dataset_object.train_dataset.columns
        self.assertEqual(set(columns_from_dataset), set(official_columns))

        # verify if dataframe contains NaN (None) values
        self.assertFalse(self.dataset_object.train_dataset.isnull().any().any(), 'Dataframe contains NaN values')

    def test_received_dev_seen_dataset(self):
        # verify read dataframe columns
        official_columns = ["id", "img", "label", "text"]
        columns_from_dataset = self.dataset_object.dev_seen_dataset.columns
        self.assertEqual(set(columns_from_dataset), set(official_columns))

        # verify if dataframe contains NaN (None) values
        self.assertFalse(self.dataset_object.dev_seen_dataset.isnull().any().any(), 'Dataframe contains NaN values')

    def test_received_dev_unseen_dataset(self):
        # verify read dataframe columns
        official_columns = ["id", "img", "label", "text"]
        columns_from_dataset = self.dataset_object.dev_unseen_dataset.columns
        self.assertEqual(set(columns_from_dataset), set(official_columns))

        # verify if dataframe contains NaN (None) values
        self.assertFalse(self.dataset_object.dev_unseen_dataset.isnull().any().any(), 'Dataframe contains NaN values')

    def test_received_test_seen_dataset(self):
        # verify read dataframe columns
        official_columns = ["id", "img", "text"]
        columns_from_dataset = self.dataset_object.test_seen_dataset.columns
        self.assertEqual(set(columns_from_dataset), set(official_columns))

        # verify if dataframe contains NaN (None) values
        self.assertFalse(self.dataset_object.test_seen_dataset.isnull().any().any(), 'Dataframe contains NaN values')

    def test_received_test_unseen_dataset(self):
        # verify read dataframe columns
        official_columns = ["id", "img", "text"]
        columns_from_dataset = self.dataset_object.test_unseen_dataset.columns
        self.assertEqual(set(columns_from_dataset), set(official_columns))

        # verify if dataframe contains NaN (None) values
        self.assertFalse(self.dataset_object.test_unseen_dataset.isnull().any().any(), 'Dataframe contains NaN values')

    def test_obtained_texts_from_train_dataset(self):
        dictionary_texts = self.dataset_object.get_texts_from_train_dataset()

        # verify obtained dictionary keys
        offical_keys = ["id", "label", "text"]
        keys_from_dictionary = dictionary_texts.keys()
        self.assertEqual(set(offical_keys), set(keys_from_dictionary))

        # verify the number of entries
        self.assertEqual(self.dataset_object.train_dataset.shape[0], len(dictionary_texts.get("id")))

        # verify if the obtained texts are correct
        self.assertTrue(self.dataset_object.train_dataset["id"].isin(dictionary_texts.get("id")).all(),
                        'Not all texts ids obtained')
        self.assertTrue(self.dataset_object.train_dataset["label"].isin(dictionary_texts.get("label")).all(),
                        'Not all texts labels obtained')
        self.assertTrue(self.dataset_object.train_dataset["text"].isin(dictionary_texts.get("text")).all(),
                        'Not all texts obtained')

    def test_obtained_texts_from_dev_seen_dataset(self):
        dictionary_texts = self.dataset_object.get_texts_from_dev_seen_dataset()

        # verify obtained dictionary keys
        offical_keys = ["id", "label", "text"]
        keys_from_dictionary = dictionary_texts.keys()
        self.assertEqual(set(offical_keys), set(keys_from_dictionary))

        # verify the number of entries
        self.assertEqual(self.dataset_object.dev_seen_dataset.shape[0], len(dictionary_texts.get("id")))

        # verify if the obtained texts are correct
        self.assertTrue(self.dataset_object.dev_seen_dataset["id"].isin(dictionary_texts.get("id")).all(),
                        'Not all texts ids obtained')
        self.assertTrue(self.dataset_object.dev_seen_dataset["label"].isin(dictionary_texts.get("label")).all(),
                        'Not all texts labels obtained')
        self.assertTrue(self.dataset_object.dev_seen_dataset["text"].isin(dictionary_texts.get("text")).all(),
                        'Not all texts obtained')

    def test_obtained_texts_from_dev_unseen_dataset(self):
        dictionary_texts = self.dataset_object.get_texts_from_dev_unseen_dataset()

        # verify obtained dictionary keys
        offical_keys = ["id", "label", "text"]
        keys_from_dictionary = dictionary_texts.keys()
        self.assertEqual(set(offical_keys), set(keys_from_dictionary))

        # verify the number of entries
        self.assertEqual(self.dataset_object.dev_unseen_dataset.shape[0], len(dictionary_texts.get("id")))

        # verify if the obtained texts are correct
        self.assertTrue(self.dataset_object.dev_unseen_dataset["id"].isin(dictionary_texts.get("id")).all(),
                        'Not all texts ids obtained')
        self.assertTrue(self.dataset_object.dev_unseen_dataset["label"].isin(dictionary_texts.get("label")).all(),
                        'Not all texts labels obtained')
        self.assertTrue(self.dataset_object.dev_unseen_dataset["text"].isin(dictionary_texts.get("text")).all(),
                        'Not all texts obtained')

    def test_obtained_texts_from_test_seen_dataset(self):
        dictionary_texts = self.dataset_object.get_texts_from_test_seen_dataset()

        # verify obtained dictionary keys
        offical_keys = ["id", "label", "text"]
        keys_from_dictionary = dictionary_texts.keys()
        self.assertEqual(set(offical_keys), set(keys_from_dictionary))

        # verify the number of entries
        self.assertEqual(self.dataset_object.test_seen_dataset.shape[0], len(dictionary_texts.get("id")))

        # verify if the obtained texts are correct
        self.assertTrue(self.dataset_object.test_seen_dataset["id"].isin(dictionary_texts.get("id")).all(),
                        'Not all texts ids obtained')
        self.assertTrue(len(dictionary_texts.get("label")) == 0, 'All texts labels obtained')
        self.assertTrue(self.dataset_object.test_seen_dataset["text"].isin(dictionary_texts.get("text")).all(),
                        'Not all texts obtained')

    def test_obtained_texts_from_test_unseen_dataset(self):
        dictionary_texts = self.dataset_object.get_texts_from_test_unseen_dataset()

        # verify obtained dictionary keys
        offical_keys = ["id", "label", "text"]
        keys_from_dictionary = dictionary_texts.keys()
        self.assertEqual(set(offical_keys), set(keys_from_dictionary))

        # verify the number of entries
        self.assertEqual(self.dataset_object.test_unseen_dataset.shape[0], len(dictionary_texts.get("id")))

        # verify if the obtained texts are correct
        self.assertTrue(self.dataset_object.test_unseen_dataset["id"].isin(dictionary_texts.get("id")).all(),
                        'Not all texts ids obtained')
        self.assertTrue(len(dictionary_texts.get("label")) == 0, 'All texts labels obtained')
        self.assertTrue(self.dataset_object.test_unseen_dataset["text"].isin(dictionary_texts.get("text")).all(),
                        'Not all texts obtained')

    def test_obtained_images_from_train_dataset(self):
        list_images = self.dataset_object.get_images_from_train_dataset()

        # verify the number of entries
        self.assertEqual(self.dataset_object.train_dataset.shape[0], len(list_images))

        # verify if the obtained images are correct
        images = [image.split("/", 2)[-1] for image in list_images]
        self.assertTrue(self.dataset_object.train_dataset["img"].isin(images).all(), 'Not all images obtained')

    def test_obtained_images_from_dev_seen_dataset(self):
        list_images = self.dataset_object.get_images_from_dev_seen_dataset()

        # verify the number of entries
        self.assertEqual(self.dataset_object.dev_seen_dataset.shape[0], len(list_images))

        # verify if the obtained images are correct
        images = [image.split("/", 2)[-1] for image in list_images]
        self.assertTrue(self.dataset_object.dev_seen_dataset["img"].isin(images).all(), 'Not all images obtained')

    def test_obtained_images_from_dev_unseen_dataset(self):
        list_images = self.dataset_object.get_images_from_dev_unseen_dataset()

        # verify the number of entries
        self.assertEqual(self.dataset_object.dev_unseen_dataset.shape[0], len(list_images))

        # verify if the obtained images are correct
        images = [image.split("/", 2)[-1] for image in list_images]
        self.assertTrue(self.dataset_object.dev_unseen_dataset["img"].isin(images).all(), 'Not all images obtained')

    def test_obtained_images_from_test_seen_dataset(self):
        list_images = self.dataset_object.get_images_from_test_seen_dataset()

        # verify the number of entries
        self.assertEqual(self.dataset_object.test_seen_dataset.shape[0], len(list_images))

        # verify if the obtained images are correct
        images = [image.split("/", 2)[-1] for image in list_images]
        self.assertTrue(self.dataset_object.test_seen_dataset["img"].isin(images).all(), 'Not all images obtained')

    def test_obtained_images_from_test_unseen_dataset(self):
        list_images = self.dataset_object.get_images_from_test_unseen_dataset()

        # verify the number of entries
        self.assertEqual(self.dataset_object.test_unseen_dataset.shape[0], len(list_images))

        # verify if the obtained images are correct
        images = [image.split("/", 2)[-1] for image in list_images]
        self.assertTrue(self.dataset_object.test_unseen_dataset["img"].isin(images).all(), 'Not all images obtained')


if __name__ == "__main__":
    unittest.main()

# Comenzi rulare coverage pentru aceste unit teste
# -> coverage run -m unittest Implementation/tests/data/test_dataset.py
# -> coverage report sau coverage report -m (arata si numarul liniilor unde nu s-a facut coverage)
