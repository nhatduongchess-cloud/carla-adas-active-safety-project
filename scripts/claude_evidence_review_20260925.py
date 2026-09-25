"""Read-only offline reproductions for the Claude handoff; never connects to CARLA."""
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.active_safety import ActiveSafetySystem
from modules.ego_control import EgoController
from modules.kpi import KpiRecorder
from modules.l3_report import assess_profile, build_report
from modules.mrm_controller import L3StateMachine
from modules.odd_monitor import ODDMonitor
from modules.scenario_acceptance import assess_scenario
from modules.scenario_library import RunningScenario
from modules.sensor_health import SensorHealthMonitor
from modules.weather_model import estimate_conditions


def main():
    evidence = {}
    evidence['odd_all_nan'] = ODDMonitor().classify(
        {'visibility_m': math.nan, 'mu': math.nan, 'snr': math.nan})['state']
    machine = L3StateMachine()
    sequence = []
    for takeover in (False, True, True):
        sequence.append(machine.update('VIOLATION', takeover, 10., .4, .025)['state'])
    evidence['takeover_while_odd_violation'] = sequence

    cfg = SimpleNamespace(EGO_CONTROL_MODE='custom', LANE_CHANGE_COOLDOWN_S=8.,
                          FIXED_DELTA=.025, TM_DEFAULT_SPEED_DIFF=0.)
    with patch('modules.ego_control.EgoDrivingStack'), \
            patch('modules.ego_control.EgoRoute'), \
            patch('modules.ego_control.set_hazard_lights'), \
            patch('modules.ego_control.carla.VehicleControl', side_effect=lambda **kw: kw):
        ego = MagicMock()
        controller = EgoController(ego, MagicMock(), cfg, world=MagicMock())
        decision = SimpleNamespace(action='BRAKE', brake=1., state='EMERGENCY_BRAKE')
        status = controller.apply(decision, {'override': True, 'state': 'MRM_EXECUTING',
            'target_decel_ms2': 1., 'hazard': True}, 'VIOLATION', 0)
        evidence['mrm_overrides_aeb'] = {
            'aeb_requested_brake': 1., 'applied': ego.apply_control.call_args.args[0],
            'mode': status['mode']}

    for label, delay, clearance in [('nan', math.nan, math.nan), ('negative', -1., .5)]:
        evidence['scenario_invalid_' + label] = assess_scenario(
            'probe', 'crossing', {'collisions': 0, 'min_distance_m': clearance},
            triggered=True, reacted=True, reaction_delay_s=delay, braked=True, evaded=False)
    evidence['scenario_missing_observations'] = assess_scenario(
        'probe', 'crossing', {}, triggered=True, reacted=True,
        reaction_delay_s=.1, braked=True, evaded=False)
    evidence['late_braking_not_a_profile_gate'] = assess_profile(
        'clear', 'NORMAL', 'NORMAL', {'collisions': 0}, False, 0, 0, 5)

    duplicated = [{'name': 'one_case', 'seed': 42, 'weather': 'clear',
                   'triggered': True, 'collisions': 0, 'pass': True} for _ in range(45)]
    evidence['duplicate_case_gate'] = build_report(
        duplicated, meta={'core_scenarios': ['one_case']})['acceptance_gate']
    evidence['empty_suite_gate'] = build_report([])['acceptance_gate']
    evidence['empty_kpi'] = KpiRecorder(.025).summary()

    health = SensorHealthMonitor({'camera': True, 'lidar': True, 'radar': False})
    for i in range(5):
        health.next_frame()
        health.observe('lidar', i, i * .025, False)
    evidence['sole_range_sensor_lost_with_radar_disabled'] = health.summary()

    safety = ActiveSafetySystem(enable_evasion=False)
    obstacle = {'centroid': (4., 0., 0.), 'point_count': 10}
    sequence = [safety.update(1., [], [obstacle], .025).action]
    sequence.extend(safety.update(1., [], [], .025).action for _ in range(5))
    evidence['brake_latch_then_missing_geometry'] = sequence

    ego = SimpleNamespace(get_location=lambda: SimpleNamespace(x=0., y=0.))
    actor = SimpleNamespace(is_alive=True,
        get_location=lambda: SimpleNamespace(x=3., y=0.),
        bounding_box=SimpleNamespace(extent=SimpleNamespace(x=2., y=1.)))
    evidence['reported_clearance_is_center_distance'] = RunningScenario([actor])._min_dist(ego)
    evidence['weather_model_friction_at_extremes'] = [
        estimate_conditions(precipitation=p, fog_density=p, wetness=p,
                            precipitation_deposits=p)['mu'] for p in (0., 100.)]

    benchmarks = []
    for name in ('head_catalog_3seed', 'head_core_5weather', 'head_core_seed42', 'head_doc_probe'):
        report = json.loads((ROOT / 'docs' / 'benchmarks' / (name + '.json')).read_text(encoding='utf-8'))
        benchmarks.append({'file': name, 'summary': report['summary'],
            'acceptance_gate': report.get('acceptance_gate'),
            'run_metadata_keys': list(report.get('run', {})),
            'device': report.get('run', {}).get('inference_device'),
            'collisions': sum(r.get('collisions', 0) for r in report['scenarios']),
            'failures': [{'name': r.get('name'), 'seed': r.get('seed'),
                          'weather': r.get('weather'), 'delay_s': r.get('reaction_delay_s'),
                          'reasons': r.get('reasons')}
                         for r in report['scenarios'] if not r['pass']]})
    evidence['benchmarks'] = benchmarks
    demo = json.loads((ROOT / 'docs/benchmarks/demo_clear.json').read_text(encoding='utf-8'))
    evidence['demo_timing'] = {'status': demo['status'], 'run': demo['run'],
        'frames_div_reported_wall_seconds': demo['run']['frames'] / demo['run']['wall_duration_s'],
        'caveat': 'Reported wall duration is sampled at report creation; not an isolated control window.'}
    output = ROOT / 'output/verification/claude_evidence_review_20260925.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
