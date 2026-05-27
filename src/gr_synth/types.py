from pydantic import BaseModel, Field, model_validator

from .prompts import PROMPTS


class Record(BaseModel):
    text: str = Field(
        description="The rephrased output text produced by the model for this prompt/doc pair."
    )
    language_confidence: float | None = Field(
        description="fastText lid.176 probability that ``text`` is Greek (averaged over sampled slices for long texts)."
    )
    source_id: str = Field(
        description="Stable identifier of the source document — the source ``url`` when present, else its ``id``, else the stream index."
    )
    prompt: str = Field(
        description="Name of the rephrasing prompt used (one of the keys in ``PROMPTS``: faq, math, table, tutorial)."
    )
    model: str = Field(
        description="vLLM model id that generated ``text`` (``settings.vllm_model_id`` at run time)."
    )
    source_data: str = Field(
        description="The source dataset this generated sample came from."
    )
    failure_reason: str | None = Field(
        default=None,
        description="If this record was kept by filters, None. Otherwise the reason "
        "(e.g. 'dropped for lang').",
    )

    @model_validator(mode="after")
    def validate_prompt(self):
        if self.prompt not in PROMPTS.keys():
            raise ValueError(
                f"prompt field should be one of {sorted(PROMPTS)}, got {self.prompt}"
            )
        return self
