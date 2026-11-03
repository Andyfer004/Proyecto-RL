import csv
import json
from pathlib import Path
import subprocess
import sys
import pytest
from thruster_landing.evaluation import load_scenarios, summarize

ROOT = Path(__file__).resolve().parents[1]


def test_manifests_are_disjoint_and_seeds_unique():
    validation = load_scenarios(ROOT / 'configs/pid_validation_scenarios.json')
    test = load_scenarios(ROOT / 'configs/pid_test_scenarios.json')
    a = {row['seed'] for row in validation['scenarios']}
    b = {row['seed'] for row in test['scenarios']}
    assert not a & b
    assert len(a) == len(validation['scenarios'])
    assert len(b) == len(test['scenarios'])


def test_pid_manifest_cli_records_contact_absence_and_trace(tmp_path):
    manifest = {'schema_version': 1, 'split': 'validation', 'scenarios': [
        {'id': 'timeout', 'condition': 'timeout', 'seed': 20, 'options': {}}]}
    path = tmp_path / 'scenarios.json'
    path.write_text(json.dumps(manifest))
    config = tmp_path / 'env.json'
    config.write_text(json.dumps({'max_time': 0.013}))
    output = tmp_path / 'results.csv'
    cmd = [sys.executable, '-m', 'thruster_landing.play', '--controller', 'pid',
           '--scenarios', str(path), '--config', str(config), '--trace', '--output', str(output)]
    result = subprocess.run(cmd, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    with output.open() as stream:
        row = next(csv.DictReader(stream))
    assert row['outcome'] == 'time_limit'
    assert row['contact_vy_m_s'] == ''
    assert row['final_center_height_m']
    metadata = json.loads(output.with_suffix('.metadata.json').read_text())
    assert metadata['pid_config_sha256']
    assert metadata['manifest_sha256']
    assert metadata['scenario_manifest'] == manifest
    assert output.with_suffix('.traces.csv').is_file()
    with output.with_suffix('.summary.csv').open() as stream:
        summary = next(csv.DictReader(stream))
    assert summary['n_contact'] == '0'
    assert summary['mean_contact_vy_m_s_contacts'] == ''
    first = output.read_bytes()
    subprocess.run(cmd, check=True, capture_output=True)
    assert output.read_bytes() == first


def test_summary_separates_failures_and_successful_contacts():
    def episode(outcome, consumed, contact):
        return {'condition': 'example', 'outcome': outcome, 'success': outcome == 'success',
                'gas_consumed_Ns': consumed, 'duration_s': 1, 'final_error_x_m': 0,
                'contact_error_x_m': 0 if contact else None,
                'contact_vx_m_s': 0 if contact else None,
                'contact_vy_m_s': -0.5 if contact else None,
                'contact_theta_rad': 0 if contact else None,
                'contact_omega_rad_s': 0 if contact else None}
    summary = summarize([episode('success', 100, True), episode('out_of_bounds', 200, False)])[0]
    assert summary['n'] == 2 and summary['n_contact'] == 1 and summary['n_success'] == 1
    assert summary['mean_gas_consumed_Ns_all'] == 150
    assert summary['mean_gas_consumed_Ns_success'] == 100
    assert summary['mean_contact_vy_m_s_contacts'] == -0.5


def test_duplicate_manifest_rejected(tmp_path):
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps({'schema_version': 1, 'split': 'test',
                               'scenarios': [{'id': 'same'}, {'id': 'same'}]}))
    with pytest.raises(ValueError):
        load_scenarios(path)
