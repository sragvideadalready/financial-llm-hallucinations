import json
from pathlib import Path
import pytest

from ..nli import NLIModel, nli_entailment_check


# ---------------- load test cases ----------------

TEST_FILE = Path(__file__).parent / "tests.json"

with open(TEST_FILE, "r", encoding="utf-8") as f:
    TEST_CASES = json.load(f)


# ---------------- pytest fixture ----------------

@pytest.fixture(scope="session")
def nli_model():
    """
    This USES YOUR CLASS NLIModel().
    It is created ONCE and reused.
    """
    return NLIModel()


# ---------------- actual test ----------------

@pytest.mark.parametrize("case", TEST_CASES)
def test_nli_cases(case, nli_model):
    result = nli_entailment_check(
        answer=case["answer"],
        context=case["context"],
        nli_model=nli_model
    )

    assert result.label == case["expected_label"], case["name"]
