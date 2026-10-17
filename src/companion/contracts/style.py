from enum import StrEnum
from typing import Annotated, Self

from pydantic import Field, StringConstraints, model_validator

from companion.contracts.base import ContractModel


class StyleIntent(StrEnum):
    CATEGORY_DEFAULT = "category_default"
    SLEEP_FAREWELL = "sleep_farewell"
    SLEEP_REMINDER = "sleep_reminder"
    CONVERSATION_REOPEN = "conversation_reopen"
    COURTESY_THANKS = "courtesy_thanks"
    COURTESY_APOLOGY = "courtesy_apology"
    WEATHER_COLD_REMINDER = "weather_cold_reminder"
    WORK_STUDY_COMPLETED = "work_study_completed"
    MORNING_GREETING = "morning_greeting"
    MISSED_MEAL_CARE = "missed_meal_care"
    UNFINISHED_HOUSEHOLD_REASSURANCE = "unfinished_household_reassurance"
    SHOPPING_COMPARISON_REQUEST = "shopping_comparison_request"
    WORK_FATIGUE_CARE = "work_fatigue_care"
    POSITIVE_EMOTION_SHARE = "positive_emotion_share"
    NEGATIVE_EMOTION_CHECK_IN = "negative_emotion_check_in"
    MISSED_CONTACT_AFFECTION = "missed_contact_affection"
    NOT_EATEN_YET_CARE = "not_eaten_yet_care"
    RAIN_REMINDER = "rain_reminder"
    HOUSEHOLD_COMPLETED_ACKNOWLEDGEMENT = "household_completed_acknowledgement"
    GOING_OUT_CARE = "going_out_care"
    SHOPPING_DESIRE_ACKNOWLEDGEMENT = "shopping_desire_acknowledgement"


StyleVersion = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]
StyleCategory = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=40)
]
StyleExample = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)
]
StyleSoftLock = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=32)
]


class StyleGuidanceStatus(StrEnum):
    MATCHED = "matched"
    NO_MATCH = "no_match"
    UNAVAILABLE = "unavailable"


class StyleReplyGuidance(ContractModel):
    status: StyleGuidanceStatus
    style_version: StyleVersion | None = None
    category: StyleCategory | None = None
    intent: StyleIntent | None = None
    examples: list[StyleExample] = Field(default_factory=list, max_length=2)
    soft_lock_expression: StyleSoftLock | None = None

    @model_validator(mode="after")
    def require_consistent_state(self) -> Self:
        if len(self.examples) != len(set(self.examples)):
            raise ValueError("style guidance examples must be unique")
        if self.soft_lock_expression in self.examples:
            raise ValueError("style guidance soft lock must not duplicate an example")
        content = (self.category, self.intent, self.examples, self.soft_lock_expression)
        if self.status is StyleGuidanceStatus.MATCHED:
            if (
                self.style_version is None
                or self.category is None
                or self.intent is None
                or not (self.examples or self.soft_lock_expression)
            ):
                raise ValueError("matched style guidance is incomplete")
        elif self.status is StyleGuidanceStatus.NO_MATCH:
            if self.style_version is None or any(content):
                raise ValueError("no-match style guidance must contain only a version")
        elif self.style_version is not None or any(content):
            raise ValueError("unavailable style guidance must be empty")
        return self


class StylePreference(ContractModel):
    rule_id: str = Field(min_length=1, max_length=80)
    category: str = Field(min_length=1, max_length=40)
    expression: str = Field(min_length=1, max_length=32)
    trigger_tags: list[str] = Field(min_length=1, max_length=8)
    intent_tags: list[StyleIntent] = Field(default_factory=list, max_length=4)
    max_consecutive_uses: int = Field(ge=0, le=3)

    @model_validator(mode="after")
    def require_unique_intent_tags(self) -> "StylePreference":
        if len(set(self.intent_tags)) != len(self.intent_tags):
            raise ValueError("style preference intent tags must be unique")
        return self


class StylePack(ContractModel):
    version: str
    statistics: dict[str, float | int | str]
    scene_tags: list[str]
    forbidden_traits: list[str]
    examples: list[str]
    preferences: list[StylePreference] = Field(default_factory=list)


PartnerTurn = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=60)
]
OwnerReplyLine = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)
]


class StyleSnippet(ContractModel):
    partner_turns: list[PartnerTurn] = Field(min_length=1, max_length=3)
    owner_reply: list[OwnerReplyLine] = Field(min_length=1, max_length=3)


class AddresseeRegister(StrEnum):
    INTIMATE = "intimate"
    NEUTRAL = "neutral"


class StyleEvidence(ContractModel):
    style_version: StyleVersion
    addressee_register: AddresseeRegister
    similar: list[StyleSnippet] = Field(default_factory=list, max_length=6)
    general: list[StyleSnippet] = Field(default_factory=list, max_length=6)

    @model_validator(mode="after")
    def require_unique_snippets(self) -> Self:
        for snippets in (self.similar, self.general):
            for index, snippet in enumerate(snippets):
                if snippet in snippets[:index]:
                    raise ValueError("style evidence snippets must be unique")
        if any(snippet in self.general for snippet in self.similar):
            raise ValueError("style evidence similar and general must not overlap")
        return self
