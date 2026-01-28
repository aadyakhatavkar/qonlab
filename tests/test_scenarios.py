import json
from pathlib import Path


def test_example_scenarios_have_task_and_owner():
    p = Path('scenarios/example_scenarios.json')
    assert p.exists(), 'example_scenarios.json missing'
    data = json.loads(p.read_text())
    assert isinstance(data, list) and len(data) > 0
    for sc in data:
        assert 'name' in sc
        assert 'task' in sc
        assert 'owner' in sc
        # variance scenarios should have variance_Tb and variance_sigma1/variance_sigma2
        if sc['task'] == 'variance':
            assert 'variance_Tb' in sc and 'variance_sigma1' in sc and 'variance_sigma2' in sc
        if sc['task'] == 'parameter':
            # parameter scenarios should include Tb and phi1/phi2 or similar
            assert 'Tb' in sc
        if sc['task'] == 'mean':
            # mean scenarios should include Tb and mu0/mu1
            assert 'Tb' in sc and 'mu0' in sc and 'mu1' in sc

