from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

StageHandler = Callable[[Any, "PipelineState"], Any]


@dataclass(slots=True)
class PipelineStage:
    name: str
    handler: StageHandler


@dataclass(slots=True)
class PipelineState:
    initial_input: Any
    current_value: Any
    stage_outputs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_output(self, stage_name: str) -> Any:
        if stage_name not in self.stage_outputs:
            raise KeyError(f"Stage '{stage_name}' has not produced an output yet")
        return self.stage_outputs[stage_name]

    def set_output(self, stage_name: str, value: Any) -> None:
        if stage_name not in self.stage_outputs:
            raise KeyError(f"Stage '{stage_name}' has not produced an output yet")
        self.stage_outputs[stage_name] = value
        self.current_value = value

    def reset_from(self, stage_names: list[str]) -> None:
        for stage_name in stage_names:
            self.stage_outputs.pop(stage_name, None)


class Pipeline:
    def __init__(self, name: str = "pipeline") -> None:
        self.name = name
        self._stages: list[PipelineStage] = []

    @property
    def stage_names(self) -> list[str]:
        return [stage.name for stage in self._stages]

    def add_stage(self, name: str, handler: StageHandler) -> "Pipeline":
        if name in self.stage_names:
            raise ValueError(f"Stage '{name}' already exists in pipeline '{self.name}'")

        self._stages.append(PipelineStage(name=name, handler=handler))
        return self

    def run(self, initial_input: Any, metadata: dict[str, Any] | None = None) -> PipelineState:
        state = PipelineState(
            initial_input=initial_input,
            current_value=initial_input,
            metadata=dict(metadata or {}),
        )
        return self._execute(state, start_index=0)

    def resume(self, state: PipelineState, from_stage: str) -> PipelineState:
        start_index = self._get_stage_index(from_stage)

        if start_index == 0:
            state.current_value = state.initial_input
        else:
            previous_stage_name = self._stages[start_index - 1].name
            if previous_stage_name not in state.stage_outputs:
                raise ValueError(
                    f"Cannot resume from stage '{from_stage}' because the previous stage "
                    f"'{previous_stage_name}' does not have an output"
                )
            state.current_value = state.stage_outputs[previous_stage_name]

        state.reset_from(self.stage_names[start_index:])
        return self._execute(state, start_index=start_index)

    def _execute(self, state: PipelineState, start_index: int) -> PipelineState:
        current_value = state.current_value

        for stage in self._stages[start_index:]:
            current_value = stage.handler(current_value, state)
            state.stage_outputs[stage.name] = current_value
            state.current_value = current_value

        return state

    def _get_stage_index(self, stage_name: str) -> int:
        for index, stage in enumerate(self._stages):
            if stage.name == stage_name:
                return index

        raise ValueError(f"Stage '{stage_name}' does not exist in pipeline '{self.name}'")
