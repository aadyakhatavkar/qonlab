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
        # variance scenarios should have Tb and sigma1/sigma2
        if sc['task'] == 'variance':
            assert 'Tb' in sc and 'sigma1' in sc and 'sigma2' in sc
        if sc['task'] == 'parameter':
            # parameter scenarios should include Tb and phi1/phi2 or similar
            assert 'Tb' in sc

