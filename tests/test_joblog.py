from api.joblog import JobReporter
from common import PipelineState
from common.reporter import CONSOLE, Reporter, get_reporter


def make_reporter(tmp_path):
    messages = []
    reporter = JobReporter(
        tmp_path / "pipeline.log", messages.append, progress_interval_s=0
    )
    return reporter, messages


def saved_lines(reporter):
    return [line.split("  ", 1)[1] for line in reporter.path.read_text().splitlines()]


def test_lines_are_saved_and_streamed_with_numbers(tmp_path):
    reporter, messages = make_reporter(tmp_path)

    reporter.info("Loading model")
    reporter.info("two\nlines")
    reporter.warning("Rotation conversion failed")
    reporter.close()

    assert saved_lines(reporter) == [
        "Loading model",
        "two",
        "lines",
        "Warning: Rotation conversion failed",
    ]
    logs = [m for m in messages if m["type"] == "log"]
    assert [m["n"] for m in logs] == [1, 2, 3, 4]


def test_progress_streams_updates_and_saves_only_the_final_state(tmp_path):
    reporter, messages = make_reporter(tmp_path)

    items = list(reporter.progress(range(3), desc="Downloading images"))
    reporter.close()

    assert items == [0, 1, 2]
    progress = [m["text"] for m in messages if m["type"] == "progress"]
    assert progress[0] == "Downloading images: 0/3 (0%)"
    assert any(t.startswith("Downloading images: 1/3 (33%)") for t in progress)
    assert progress[-1] == ""  # cleared when the loop ends
    assert saved_lines(reporter) == ["Downloading images: 3/3 (100%)"]


def test_reporter_comes_from_pipeline_state_or_defaults_to_console(tmp_path):
    reporter, _ = make_reporter(tmp_path)
    with_reporter = PipelineState(None, None, metadata={"reporter": reporter})
    without = PipelineState(None, None)

    assert get_reporter(with_reporter) is reporter
    assert get_reporter(without) is CONSOLE
    assert get_reporter() is CONSOLE
    reporter.close()


def test_console_reporter_prints_like_before(capsys):
    console = Reporter()

    console.info("Saved mesh to out.glb")
    assert list(console.progress([1, 2], desc="Visibility")) == [1, 2]

    out = capsys.readouterr()
    assert "Saved mesh to out.glb" in out.out
    assert "Visibility" in out.err  # tqdm's bar, on stderr as before
