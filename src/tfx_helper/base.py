import os.path
import re

from pydantic import BaseModel, validator

PIPELINE_NAME_PATTERN = re.compile("^[A-Za-z][A-Za-z0-9]+$")


class BasePipelineHelper(BaseModel):
    pipeline_name: str
    """The name of the pipeline."""

    output_dir: str
    """Where all pipeline artifacts should be stored."""

    @validator("pipeline_name")
    def pipeline_name_must_follow_pattern(cls, v: str) -> str:
        if not PIPELINE_NAME_PATTERN.match(v):
            raise ValueError("Pipeline name must contain letters and digits only")
        return v

    @property
    def pipeline_root(self) -> str:
        return os.path.join(self.output_dir, self.pipeline_name)
