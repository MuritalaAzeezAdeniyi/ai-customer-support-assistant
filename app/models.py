from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, StrictBool, field_validator

Category = Literal[
    "TRANSFER_ISSUE",
    "ACCOUNT_ISSUE",
    "CARD_ISSUE",
    "PAYMENT_ISSUE",
    "BILLING_ISSUE",
    "PASSWORD_RESET",
    "OTHER",
]

Priority = Literal["LOW", "MEDIUM", "HIGH", "URGENT"]
Sentiment = Literal["POSITIVE", "NEGATIVE", "NEUTRAL"]


class SupportResponse(BaseModel):
    """Structured response contract expected from the LLM classifier."""

    category: Category = Field(..., description="Primary issue category for the customer request.")
    priority: Priority = Field(..., description="Priority level assigned to the ticket.")
    sentiment: Sentiment = Field(..., description="Customer sentiment inferred from the message.")
    requires_human: StrictBool = Field(..., description="Whether a human agent should review the case.")
    suggested_response: str = Field(
        ...,
        min_length=10,
        description="Professional support message that addresses the customer concern.",
    )

    @field_validator("category", "priority", "sentiment", mode="before")
    @classmethod
    def normalize_enum_values(cls, value: str) -> str:
        if isinstance(value, str):
            return value.strip().upper()
        return value

    @field_validator("suggested_response")
    @classmethod
    def sanitize_response(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("suggested_response cannot be blank")
        return cleaned
