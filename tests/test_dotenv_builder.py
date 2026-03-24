import pytest
from pathlib import Path
from unittest.mock import patch

from dotenv_builder import choose_environment, main


@pytest.mark.parametrize(
    "choice, expected_label, expected_filename",
    [
        ("1", "development", ".env.development"),
        ("2", "production", ".env.production"),
        ("3", "none", ".env"),
    ],
)
def test_choose_environment_valid(choice, expected_label, expected_filename):
    with patch("builtins.input", return_value=choice):
        label, path = choose_environment()

    assert label == expected_label
    assert path == Path(f"./{expected_filename}")


def test_choose_environment_retries_on_invalid_then_succeeds():
    with patch("builtins.input", side_effect=["bad", "x", "1"]):
        label, path = choose_environment()

    assert label == "development"


def test_main_raises_when_example_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError, match=".env.example"):
        main()


def test_main_writes_development_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env.example").write_text("API_KEY=\nSECRET=\n")

    inputs = iter(["1", "mykey", "mysecret"])
    with patch("builtins.input", side_effect=inputs):
        main()

    content = (tmp_path / ".env.development").read_text()
    assert 'API_KEY="mykey"' in content
    assert 'SECRET="mysecret"' in content


def test_main_writes_production_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env.example").write_text("TOKEN=\n")

    with patch("builtins.input", side_effect=["2", "prod-token"]):
        main()

    content = (tmp_path / ".env.production").read_text()
    assert 'TOKEN="prod-token"' in content


def test_main_writes_plain_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env.example").write_text("TOKEN=\n")

    with patch("builtins.input", side_effect=["3", "some-token"]):
        main()

    content = (tmp_path / ".env").read_text()
    assert 'TOKEN="some-token"' in content


def test_main_skips_comments_and_blanks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env.example").write_text(
        "# this is a comment\n\nAPI_KEY=\n"
    )

    with patch("builtins.input", side_effect=["3", "abc"]):
        main()

    content = (tmp_path / ".env").read_text()
    assert 'API_KEY="abc"' in content
    assert "comment" not in content


def test_main_aborts_if_overwrite_declined(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env.example").write_text("KEY=\n")
    (tmp_path / ".env").write_text('KEY="old"\n')

    with patch("builtins.input", side_effect=["3", "n"]):
        main()

    assert (tmp_path / ".env").read_text() == 'KEY="old"\n'


def test_main_overwrites_if_confirmed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env.example").write_text("KEY=\n")
    (tmp_path / ".env").write_text('KEY="old"\n')

    with patch("builtins.input", side_effect=["3", "y", "new-value"]):
        main()

    assert 'KEY="new-value"' in (tmp_path / ".env").read_text()
