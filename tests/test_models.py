import pytest
from pydantic import ValidationError

from app.models import SupportResponse


def test_support_response_validates_expected_contract() -> None:
    payload = {
        "category": "TRANSFER_ISSUE",
        "priority": "HIGH",
        "sentiment": "NEGATIVE",
        "requires_human": True,
        "suggested_response": "We apologize for the delay and will investigate promptly.",
    }

    response = SupportResponse.model_validate(payload)

    assert response.category == "TRANSFER_ISSUE"
    assert response.priority == "HIGH"
    assert response.sentiment == "NEGATIVE"
    assert response.requires_human is True
    assert "apologize" in response.suggested_response.lower()


def test_support_response_rejects_invalid_values() -> None:
    with pytest.raises(ValidationError):
        SupportResponse.model_validate(
            {
                "category": "UNKNOWN",
                "priority": "SEVERE",
                "sentiment": "CONFUSED",
                "requires_human": "yes",
                "suggested_response": "",
            }
        )
