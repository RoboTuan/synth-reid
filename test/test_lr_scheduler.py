import sys
import unittest

# sys.path.append('.')
from torchreid.models import build_model
from torchreid.optim import build_optimizer, build_lr_scheduler
from torchreid.utils import set_random_seed
# from solver.build import make_optimizer
# from config import cfg


class TestScheduler(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        set_random_seed()

        model_name = 'resnet34'

        cls.model = build_model(
            name=model_name,
            num_classes=388,
            loss='softmax',
            pretrained=False
        )
        cls.optimizer = build_optimizer(
            model_name,
            cls.model,
            optim='adam',
            lr=0.01,
            new_layers='classifier',
            staged_lr=True,
            base_lr_mult=0.1
        )
        cls.scheduler = build_lr_scheduler(
            cls.optimizer,
            lr_scheduler='warmup_multi_step',
            stepsize=[20, 40],
            warmup_iters=10,
            warmup_factor=1 / 10,
            warmup_method='linear',
        )

    def test_something(self):
        for i in range(50):
            for j in range(3):
                print(i, self.scheduler.get_lr()[0])
                self.optimizer.step()
            self.scheduler.step()


if __name__ == '__main__':
    unittest.main()
