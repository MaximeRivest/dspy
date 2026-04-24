from dspy.teleprompt.mipro_optimizer_v2 import _candidate_set_counts
from dspy.teleprompt.teleprompt import Teleprompter


class DummyTeleprompter(Teleprompter):
    def __init__(self, param1: int, param2: str):
        super().__init__()
        self.param1 = param1
        self.param2 = param2

    def compile(self, student, *, trainset, teacher=None, valset=None, **kwargs):
        return student


def test_get_params():
    teleprompter = DummyTeleprompter(param1=1, param2="test")
    params = teleprompter.get_params()
    assert params == {"param1": 1, "param2": "test"}


def test_candidate_set_counts_handles_mipro_demo_candidate_mapping():
    assert _candidate_set_counts({0: ["zero_shot", "labels"], 1: ["zero_shot"]}) == [2, 1]
