import pytest

from src.models.train_utils import get_model, get_optimizer

"""
Using pytest.raises, we check that the ValueError for the optimizer choice works correctly.
"""

def test_optimizer_choice():
    model = get_model()
    with pytest.raises(ValueError, match="Illegal optimizer! Specify optimizer as 'sgd' or 'adam'"):
        get_optimizer(model, lr = 1e-3, weight_decay=1e-5, optimizer='AdamW')

if __name__ == "__main__":
    test_optimizer_choice()