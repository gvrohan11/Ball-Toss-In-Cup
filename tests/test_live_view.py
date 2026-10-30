import json

import live_view


def _log(tmp_path, monkeypatch, name="20261030-120000", lines=(), tail=""):
    monkeypatch.setattr(live_view, "ROOT", tmp_path)
    d = tmp_path / name
    d.mkdir()
    (d / "log.jsonl").write_text("".join(json.dumps(l) + "\n" for l in lines) + tail)
    (d / "0001_cup_view_1_det.jpg").write_bytes(b"jpg")
    return d


def test_half_written_line_waits_for_next_poll(tmp_path, monkeypatch):
    _log(tmp_path, monkeypatch, lines=[{"event": "a"}, {"event": "b"}], tail='{"event": "c"')
    assert [l["event"] for l in live_view.read_lines("20261030-120000")] == ["a", "b"]
    assert [l["event"] for l in live_view.read_lines("20261030-120000", 1)] == ["b"]


def test_state_follows_newest_folder_and_reads_only_new_lines(tmp_path, monkeypatch):
    _log(tmp_path, monkeypatch, "20261030-120000", [{"event": "old"}])
    (tmp_path / "20261030-130000").mkdir()
    (tmp_path / "20261030-130000" / "log.jsonl").write_text('{"event": "x"}\n{"event": "y"}\n')
    r = live_view.state({}, None)
    assert r["folder"] == "20261030-130000" and len(r["lines"]) == 2
    r = live_view.state({"have": ["20261030-130000"], "from": ["1"]}, None)
    assert [l["event"] for l in r["lines"]] == ["y"]
    r = live_view.state({"have": ["20261030-120000"], "from": ["5"]}, None)   # page had an older folder
    assert r["start"] == 0 and len(r["lines"]) == 2
    r = live_view.state({"folder": ["20261030-120000"]}, None)                # replay an old attempt
    assert [l["event"] for l in r["lines"]] == ["old"]


def test_only_frames_inside_log_folders_are_served(tmp_path, monkeypatch):
    d = _log(tmp_path, monkeypatch)
    assert live_view.image_path(d.name, "0001_cup_view_1_det.jpg") == d / "0001_cup_view_1_det.jpg"
    assert live_view.image_path(d.name, "../../.env") is None
    assert live_view.image_path(d.name, "log.jsonl") is None
    assert live_view.image_path("..", "x.jpg") is None
    assert live_view.image_path(d.name, "missing.jpg") is None


def test_narration_tail_strips_colours(tmp_path):
    p = tmp_path / "live.txt"
    p.write_text("1) survey\n\x1b[32m  view 1/7\x1b[0m\n\n")
    assert live_view.narration(p)["lines"] == ["1) survey", "  view 1/7"]
    assert live_view.narration(tmp_path / "none.txt") is None
