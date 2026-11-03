import csv
import json
import subprocess
import sys


def test_cli_records_reproducible_episodes_and_versions(tmp_path):
    outputs = [tmp_path / "a.csv", tmp_path / "b.csv"]
    for output in outputs:
        result = subprocess.run([sys.executable, "-m", "thruster_landing.play",
                                 "--episodes", "2", "--seed", "12", "--output", str(output)],
                                text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    with outputs[0].open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert rows[0]["seed"] == "12"
    metadata = json.loads(outputs[0].with_suffix(".metadata.json").read_text())
    assert metadata["completed_episodes"] == 2
    assert metadata["versions"]["numpy"]
    assert metadata["config"]["mass"] == 100
