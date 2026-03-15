from common.pipeline_chain import PipelineChain


def test_pipeline_runs_stages_in_order():
    pipeline = PipelineChain("numbers")
    pipeline.add_stage("double", lambda value, state: value * 2)
    pipeline.add_stage("offset", lambda value, state: value + state.metadata["offset"])
    pipeline.add_stage("square", lambda value, state: value**2)

    state = pipeline.run(3, metadata={"offset": 4})

    assert state.stage_outputs == {
        "double": 6,
        "offset": 10,
        "square": 100,
    }
    assert state.current_value == 100


def test_pipeline_fork_sends_same_input_to_each_branch():
    pipeline = PipelineChain("fork")
    pipeline.add_stage("start", lambda value, state: value + 1)
    pipeline.add_fork(
        "geometry",
        {
            "mesh": lambda value, state: value * 2,
            "roads": lambda value, state: value + 5,
        },
    )

    state = pipeline.run(4)

    assert state.get_output("start") == 5
    assert state.get_output("mesh") == 10
    assert state.get_output("roads") == 10
    assert state.get_output("geometry") == {
        "mesh": 10,
        "roads": 10,
    }
    assert state.current_value == {
        "mesh": 10,
        "roads": 10,
    }


def test_pipeline_fork_merge_shapes_next_stage_input():
    pipeline = PipelineChain("merge")
    pipeline.add_stage("start", lambda value, state: value * 3)
    pipeline.add_fork(
        "geometry",
        {
            "mesh": lambda value, state: value + 1,
            "roads": lambda value, state: value + 2,
        },
        merge=lambda outputs, state: outputs["mesh"] + outputs["roads"],
    )
    pipeline.add_stage("finish", lambda value, state: value * 10)

    state = pipeline.run(2)

    assert state.get_output("mesh") == 7
    assert state.get_output("roads") == 8
    assert state.get_output("geometry") == 15
    assert state.get_output("finish") == 150


def test_pipeline_resume_replays_only_downstream_stage():
    pipeline = PipelineChain("resume")
    pipeline.add_stage("double", lambda value, state: value * 2)
    pipeline.add_stage("offset", lambda value, state: value + 4)
    pipeline.add_stage("square", lambda value, state: value**2)

    state = pipeline.run(3)
    state.set_output("offset", 20)

    resumed_state = pipeline.resume(state, from_stage="square")

    assert resumed_state.get_output("double") == 6
    assert resumed_state.get_output("offset") == 20
    assert resumed_state.get_output("square") == 400
    assert resumed_state.current_value == 400


def test_pipeline_resume_clears_fork_outputs_before_replay():
    pipeline = PipelineChain("fork-resume")
    pipeline.add_stage("start", lambda value, state: value + 1)
    pipeline.add_fork(
        "geometry",
        {
            "mesh": lambda value, state: value * 2,
            "roads": lambda value, state: value * 3,
        },
    )
    pipeline.add_stage("finish", lambda value, state: value["mesh"] + value["roads"])

    state = pipeline.run(2)
    state.set_output("start", 10)

    resumed_state = pipeline.resume(state, from_stage="geometry")

    assert resumed_state.get_output("mesh") == 20
    assert resumed_state.get_output("roads") == 30
    assert resumed_state.get_output("geometry") == {
        "mesh": 20,
        "roads": 30,
    }
    assert resumed_state.get_output("finish") == 50