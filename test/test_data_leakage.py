import unittest
from collections import defaultdict
import torchreid
from torchreid.utils import set_random_seed


class TestLeak(unittest.TestCase):
    """ Test whether there is data overlapping between training, test and validation sets.
        It tests both with and without the validation set option.
        TODO: do test for different target domain.
    """

    @classmethod
    def setUpClass(cls) -> None:

        set_random_seed(0)

        cls.datamanager = torchreid.data.ImageDataManager(
            root='/mnt/data2/defonte_data/PersonReid_datasets/',
            sources='gta_synthreid',
            targets='gta_synthreid',
            height=256,
            width=128,
            verbose=False,
            batch_size_train=32,
            batch_size_test=100,
            transforms=['random_flip', 'random_crop'],
            relabel=False
        )

        cls.datamanager_val = torchreid.data.ImageDataManager(
            root='/mnt/data2/defonte_data/PersonReid_datasets/',
            sources='gta_synthreid',
            targets='gta_synthreid',
            height=256,
            width=128,
            verbose=False,
            batch_size_train=32,
            batch_size_test=100,
            transforms=['random_flip', 'random_crop'],
            val=True,
            relabel=False
        )

    def test_TrainTest(self) -> None:
        train_loader = self.datamanager.train_loader
        test_loader = self.datamanager.test_loader
        train_pids = set()

        for data in train_loader:
            train_pids.update(data['pid'].numpy())

        for target in test_loader.keys():
            test_pids_all = set()
            test_pids = {
                'query': set(),
                'gallery': set()
            }

            for mode in ['query', 'gallery']:
                for data in test_loader[target][mode]:
                    test_pids[mode].update(data['pid'].numpy())
                    test_pids_all.update(data['pid'].numpy())

            self.assertEqual(bool(test_pids['query'] & test_pids['gallery']), True)
            self.assertEqual(bool(test_pids_all & train_pids), False)

    def test_TrainTestVal(self) -> None:
        train_loader = self.datamanager_val.train_loader
        val_loader = self.datamanager_val.val_loader
        test_loader = self.datamanager_val.test_loader
        train_pids = set()
        val_pids_all = defaultdict(set)
        test_pids_all = defaultdict(set)

        for data in train_loader:
            train_pids.update(data['pid'].numpy())

        for target in val_loader.keys():
            val_pids_all[target] = set()
            val_pids = {
                'query': set(),
                'gallery': set()
            }
            for mode in ['query', 'gallery']:
                for data in val_loader[target][mode]:
                    val_pids[mode].update(data['pid'].numpy())
                    val_pids_all[target].update(data['pid'].numpy())

            self.assertEqual(bool(val_pids['query'] & val_pids['gallery']), True)
            self.assertEqual(bool(val_pids_all[target] & train_pids), False)

        for target in test_loader.keys():
            test_pids_all[target] = set()
            test_pids = {
                'query': set(),
                'gallery': set()
            }
            for mode in ['query', 'gallery']:
                for data in test_loader[target][mode]:
                    test_pids[mode].update(data['pid'].numpy())
                    test_pids_all[target].update(data['pid'].numpy())

            self.assertEqual(bool(test_pids['query'] & test_pids['gallery']), True)
            self.assertEqual(bool(test_pids_all[target] & train_pids), False)

        for pids_val in val_pids_all.values():
            for pids_test in test_pids_all.values():
                self.assertEqual(bool(pids_val & pids_test), False)
        print(val_pids_all)
        print(test_pids_all)


if __name__ == '__main__':
    unittest.main()
