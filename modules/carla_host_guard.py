"""Fail-closed identity guard for one launcher-managed local CARLA server.

The guard is intentionally Windows/stdlib-only.  It observes processes and TCP
listeners; it never starts, stops, or kills CARLA.
"""

from datetime import datetime, timezone
import hashlib
import json
import ntpath
import os
from pathlib import Path
import re
import secrets
import subprocess


_LAUNCHER_NAME = 'CarlaUE4.exe'
_SHIPPING_NAME = 'CarlaUE4-Win64-Shipping.exe'
_PORTS = [2000, 2001, 2002]
_MANIFEST_KEYS = {
    'schema_version', 'manifest_type', 'run_id', 'installation_root',
    'endpoint', 'launcher', 'shipping',
}
_PROCESS_KEYS = {'name', 'pid', 'created_utc', 'exe', 'command_line', 'sha256'}
_SHIPPING_KEYS = _PROCESS_KEYS | {'parent_pid'}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _absolute_windows_path(value, name):
    if not isinstance(value, str) or not ntpath.isabs(value):
        raise ValueError(f'{name} must be an absolute Windows path')
    return ntpath.normcase(ntpath.normpath(value))


def _utc_timestamp(value, name):
    if not isinstance(value, str) or not value.endswith('Z'):
        raise ValueError(f'{name} must be an explicit UTC ISO timestamp')
    try:
        parsed = datetime.fromisoformat(value[:-1] + '+00:00')
    except ValueError as error:
        raise ValueError(f'{name} must be an explicit UTC ISO timestamp') from error
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f'{name} must be UTC')
    return value


def _process_rows(snapshot, name):
    if not isinstance(snapshot, dict):
        raise RuntimeError('Windows host snapshot is not an object')
    if snapshot.get('timestamp_format') != 'explicit-iso-8601':
        raise RuntimeError('Windows host snapshot lacks explicit ISO timestamps')
    processes = snapshot.get('processes')
    listeners = snapshot.get('listeners')
    connections = snapshot.get('connections')
    if (not isinstance(processes, list) or not isinstance(listeners, list)
            or not isinstance(connections, list)):
        raise RuntimeError('Windows process/listener query is incomplete')
    return [row for row in processes if isinstance(row, dict)
            and str(row.get('name', '')).casefold() == name.casefold()]


def _exact_carla_pair(snapshot):
    launchers = _process_rows(snapshot, _LAUNCHER_NAME)
    shipping = _process_rows(snapshot, _SHIPPING_NAME)
    if len(launchers) != 1 or len(shipping) != 1:
        raise RuntimeError('Expected exactly one CARLA launcher and one Shipping process')
    launcher, child = launchers[0], shipping[0]
    if _positive_int(child.get('parent_pid'), 'Shipping parent PID') != _positive_int(
            launcher.get('pid'), 'launcher PID'):
        raise RuntimeError('CARLA Shipping process is orphaned or has the wrong parent')
    return launcher, child


def _manifest_process(row, expected_keys, name):
    if not isinstance(row, dict) or set(row) != expected_keys:
        raise ValueError(f'{name} manifest fields are invalid')
    if row.get('name') != name:
        raise ValueError(f'{name} manifest process name is invalid')
    _positive_int(row.get('pid'), f'{name} PID')
    if name == _SHIPPING_NAME:
        _positive_int(row.get('parent_pid'), 'Shipping parent PID')
    _absolute_windows_path(row.get('exe'), f'{name} executable')
    _utc_timestamp(row.get('created_utc'), f'{name} creation time')
    if not isinstance(row.get('command_line'), str) or not row['command_line'].strip():
        raise ValueError(f'{name} command line is missing')
    if not isinstance(row.get('sha256'), str) or not re.fullmatch(r'[0-9a-f]{64}', row['sha256']):
        raise ValueError(f'{name} hash is invalid')


def _validate_manifest_shape(manifest):
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS:
        raise ValueError('Managed-server manifest fields are invalid')
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported managed-server manifest schema')
    if manifest.get('manifest_type') != 'carla_launcher_managed_probe':
        raise ValueError('Unsupported managed-server manifest type')
    run_id = manifest.get('run_id')
    if not isinstance(run_id, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}', run_id):
        raise ValueError('Managed-server run_id is invalid')
    _absolute_windows_path(manifest.get('installation_root'), 'installation_root')
    endpoint = manifest.get('endpoint')
    if (not isinstance(endpoint, dict) or set(endpoint) != {'host', 'ports'}
            or endpoint.get('host') != '127.0.0.1'
            or endpoint.get('ports') != _PORTS):
        raise ValueError('Managed-server endpoint must be 127.0.0.1 ports 2000-2002')
    _manifest_process(manifest.get('launcher'), _PROCESS_KEYS, _LAUNCHER_NAME)
    _manifest_process(manifest.get('shipping'), _SHIPPING_KEYS, _SHIPPING_NAME)
    if manifest['shipping']['parent_pid'] != manifest['launcher']['pid']:
        raise ValueError('Manifest Shipping process is not the launcher child')
    return manifest


def build_managed_manifest(snapshot, install_root, run_id, *, hash_file=_sha256):
    """Build a reviewable manifest from one already-running exact process pair."""
    root = Path(install_root).resolve()
    launcher, shipping = _exact_carla_pair(snapshot)
    expected = {
        _LAUNCHER_NAME: root / _LAUNCHER_NAME,
        _SHIPPING_NAME: root / 'CarlaUE4/Binaries/Win64' / _SHIPPING_NAME,
    }

    def record(row, name, include_parent=False):
        actual = _absolute_windows_path(row.get('exe'), f'{name} executable')
        if actual != _absolute_windows_path(str(expected[name]), f'expected {name} executable'):
            raise RuntimeError(f'{name} executable path is outside the selected installation')
        result = {
            'name': name,
            'pid': _positive_int(row.get('pid'), f'{name} PID'),
            'created_utc': _utc_timestamp(row.get('created_utc'), f'{name} creation time'),
            'exe': str(expected[name]),
            'command_line': row.get('command_line'),
            'sha256': str(hash_file(expected[name])).lower(),
        }
        if include_parent:
            result['parent_pid'] = _positive_int(row.get('parent_pid'), 'Shipping parent PID')
        return result

    manifest = {
        'schema_version': 1,
        'manifest_type': 'carla_launcher_managed_probe',
        'run_id': run_id,
        'installation_root': str(root),
        'endpoint': {'host': '127.0.0.1', 'ports': list(_PORTS)},
        'launcher': record(launcher, _LAUNCHER_NAME),
        'shipping': record(shipping, _SHIPPING_NAME, include_parent=True),
    }
    return _validate_manifest_shape(manifest)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def load_managed_manifest(path):
    """Read one small strict JSON manifest and return it with its file digest."""
    path = Path(path)
    raw = path.read_bytes()
    if not raw or len(raw) > 64 * 1024:
        raise ValueError('Managed-server manifest size is invalid')
    try:
        manifest = json.loads(raw.decode('utf-8'), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError('Managed-server manifest is not valid UTF-8 JSON') from error
    return _validate_manifest_shape(manifest), hashlib.sha256(raw).hexdigest()


def marker_path(manifest_path):
    path = Path(manifest_path)
    return path.with_name(path.stem + '.consumed.json')


def claim_manifest(manifest_path, digest, run_id, *, supervisor_pid=None):
    """Atomically consume a manifest.  The marker is deliberately permanent."""
    if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
        raise ValueError('Manifest digest is invalid')
    if not isinstance(run_id, str) or not run_id:
        raise ValueError('Manifest run_id is invalid')
    pid = os.getpid() if supervisor_pid is None else supervisor_pid
    _positive_int(pid, 'supervisor PID')
    nonce = secrets.token_hex(32)
    marker = {
        'schema_version': 1,
        'run_id': run_id,
        'manifest_sha256': digest,
        'claimed_utc': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        'supervisor_pid': pid,
        'nonce_sha256': hashlib.sha256(nonce.encode('ascii')).hexdigest(),
    }
    path = marker_path(manifest_path)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(marker, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    return path, nonce


def verify_claim(manifest_path, digest, run_id, nonce):
    """Verify that this worker holds the claim created by its supervisor."""
    path = marker_path(manifest_path)
    try:
        marker = json.loads(path.read_text(encoding='utf-8'),
                            object_pairs_hook=_unique_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError('Managed-server claim marker is unreadable') from error
    expected_keys = {'schema_version', 'run_id', 'manifest_sha256', 'claimed_utc',
                     'supervisor_pid', 'nonce_sha256'}
    if not isinstance(marker, dict) or set(marker) != expected_keys:
        raise RuntimeError('Managed-server claim marker fields are invalid')
    expected_nonce = hashlib.sha256(str(nonce).encode('ascii')).hexdigest()
    if (marker.get('schema_version') != 1 or marker.get('run_id') != run_id
            or marker.get('manifest_sha256') != digest
            or not secrets.compare_digest(str(marker.get('nonce_sha256')), expected_nonce)):
        raise RuntimeError('Managed-server claim does not belong to this worker')
    _positive_int(marker.get('supervisor_pid'), 'claim supervisor PID')
    _utc_timestamp(marker.get('claimed_utc'), 'claim timestamp')
    return marker


def _is_beneath(path, root):
    path, root = _absolute_windows_path(path, 'process path'), _absolute_windows_path(root, 'root path')
    return path == root or path.startswith(root.rstrip('\\') + '\\')


def allowed_python_chain(snapshot, current_pid=None):
    """Return only the current Python process and its Python ancestor chain."""
    current = os.getpid() if current_pid is None else current_pid
    rows = {row.get('pid'): row for row in snapshot.get('processes', [])
            if isinstance(row, dict)
            and str(row.get('name', '')).casefold() in {'python.exe', 'pythonw.exe'}}
    allowed = set()
    while current in rows and current not in allowed:
        allowed.add(current)
        current = rows[current].get('parent_pid')
    return allowed


def validate_managed_server(manifest, snapshot, *, expected_install_root,
                            project_root, allowed_python_pids, hash_file=_sha256):
    """Validate current host identity against an immutable reviewed manifest."""
    _validate_manifest_shape(manifest)
    expected_root = _absolute_windows_path(str(Path(expected_install_root).resolve()),
                                           'expected installation_root')
    if _absolute_windows_path(manifest['installation_root'], 'installation_root') != expected_root:
        raise RuntimeError('Managed manifest selects a different CARLA installation')
    launcher, shipping = _exact_carla_pair(snapshot)
    current = {_LAUNCHER_NAME: launcher, _SHIPPING_NAME: shipping}
    expected_paths = {
        _LAUNCHER_NAME: Path(expected_install_root).resolve() / _LAUNCHER_NAME,
        _SHIPPING_NAME: (Path(expected_install_root).resolve()
                         / 'CarlaUE4/Binaries/Win64' / _SHIPPING_NAME),
    }
    for role in (_LAUNCHER_NAME, _SHIPPING_NAME):
        wanted, actual = manifest['launcher' if role == _LAUNCHER_NAME else 'shipping'], current[role]
        actual_pid = _positive_int(actual.get('pid'), f'current {role} PID')
        if actual_pid != wanted['pid']:
            raise RuntimeError(f'{role} PID changed')
        if role == _SHIPPING_NAME:
            parent = _positive_int(actual.get('parent_pid'), 'current Shipping parent PID')
            if parent != wanted['parent_pid'] or parent != manifest['launcher']['pid']:
                raise RuntimeError('CARLA Shipping parent changed')
        actual_time = _utc_timestamp(actual.get('created_utc'), f'current {role} creation time')
        if actual_time != wanted['created_utc']:
            raise RuntimeError(f'{role} creation time changed (possible PID reuse)')
        actual_path = _absolute_windows_path(actual.get('exe'), f'current {role} executable')
        wanted_path = _absolute_windows_path(wanted['exe'], f'manifest {role} executable')
        required_path = _absolute_windows_path(str(expected_paths[role]), f'expected {role} executable')
        if actual_path != wanted_path or actual_path != required_path:
            raise RuntimeError(f'{role} executable path changed')
        if actual.get('command_line') != wanted['command_line']:
            raise RuntimeError(f'{role} command-line arguments changed')
        try:
            digest = str(hash_file(expected_paths[role])).lower()
        except Exception as error:
            raise RuntimeError(f'{role} hash could not be read') from error
        if digest != wanted['sha256']:
            raise RuntimeError(f'{role} executable hash changed')

    listeners = [row for row in snapshot['listeners'] if isinstance(row, dict)
                 and str(row.get('state', '')).casefold() == 'listen'
                 and row.get('local_port') in _PORTS]
    found_ports = {row.get('local_port') for row in listeners}
    if found_ports != set(_PORTS):
        raise RuntimeError('CARLA required listener ports are missing')
    if any(row.get('pid') != manifest['shipping']['pid'] for row in listeners):
        raise RuntimeError('A required CARLA port has a foreign owner')
    established = [row for row in snapshot['connections'] if isinstance(row, dict)
                   and str(row.get('state', '')).casefold() == 'established'
                   and (row.get('local_port') in _PORTS
                        or row.get('remote_port') in _PORTS)]
    if established:
        raise RuntimeError('Another established client connection already uses CARLA')

    project = _absolute_windows_path(str(Path(project_root).resolve()), 'project_root')
    venv = ntpath.join(project, '.venvcarla')
    allowed = set(allowed_python_pids)
    competitors = []
    for row in snapshot['processes']:
        if (not isinstance(row, dict)
                or str(row.get('name', '')).casefold() not in {'python.exe', 'pythonw.exe'}):
            continue
        pid = _positive_int(row.get('pid'), 'Python PID')
        executable = row.get('exe')
        command = str(row.get('command_line') or '')
        from_venv = bool(executable and ntpath.isabs(str(executable))
                         and _is_beneath(str(executable), venv))
        mentions_project = project in ntpath.normcase(command.replace('/', '\\'))
        if (from_venv or mentions_project) and pid not in allowed:
            competitors.append(pid)
    if competitors:
        raise RuntimeError(f'Competing project Python process detected: {sorted(competitors)}')

    return {
        'run_id': manifest['run_id'],
        'launcher_pid': manifest['launcher']['pid'],
        'shipping_pid': manifest['shipping']['pid'],
        'listener_ports': list(_PORTS),
        'powershell_version': snapshot.get('powershell_version'),
        'allowed_python_pids': sorted(allowed),
    }


def query_windows_host(ports=tuple(_PORTS), *, runner=None, timeout_s=10):
    """Query process identity and LISTEN ownership using Windows PowerShell."""
    if list(ports) != _PORTS:
        raise ValueError('Host query ports must be exactly 2000, 2001, 2002')
    runner = subprocess.run if runner is None else runner
    script = r"""
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$names = @('CarlaUE4.exe','CarlaUE4-Win64-Shipping.exe','python.exe','pythonw.exe')
$ports = @(2000,2001,2002)
$processes = @(Get-CimInstance Win32_Process | Where-Object { $names -contains $_.Name } | ForEach-Object {
    $exe = if ($null -eq $_.ExecutablePath) { $null } else { [string]$_.ExecutablePath }
    $cmd = if ($null -eq $_.CommandLine) { $null } else { [string]$_.CommandLine }
    $created = if ($null -eq $_.CreationDate) { $null } else { $_.CreationDate.ToUniversalTime().ToString('o',[Globalization.CultureInfo]::InvariantCulture) }
    [ordered]@{name=[string]$_.Name;pid=[int]$_.ProcessId;parent_pid=[int]$_.ParentProcessId;exe=$exe;created_utc=$created;command_line=$cmd}
})
$tcp = @(Get-NetTCPConnection)
$listeners = @($tcp | Where-Object { $_.State -eq 'Listen' -and $ports -contains [int]$_.LocalPort } | ForEach-Object {
    [ordered]@{local_address=[string]$_.LocalAddress;local_port=[int]$_.LocalPort;pid=[int]$_.OwningProcess;state='Listen'}
})
$connections = @($tcp | Where-Object { $_.State -eq 'Established' -and (($ports -contains [int]$_.LocalPort) -or ($ports -contains [int]$_.RemotePort)) } | ForEach-Object {
    [ordered]@{local_address=[string]$_.LocalAddress;local_port=[int]$_.LocalPort;remote_address=[string]$_.RemoteAddress;remote_port=[int]$_.RemotePort;pid=[int]$_.OwningProcess;state='Established'}
})
[ordered]@{schema_version=1;powershell_version=$PSVersionTable.PSVersion.ToString();timestamp_format='explicit-iso-8601';processes=$processes;listeners=$listeners;connections=$connections} | ConvertTo-Json -Depth 5 -Compress
""".strip()
    try:
        completed = runner(
            ['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', script],
            capture_output=True, text=True, encoding='utf-8', errors='replace',
            timeout=float(timeout_s), shell=False, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f'Windows host query failed: {error}') from error
    if completed.returncode != 0:
        raise RuntimeError(f'Windows host query failed: {completed.stderr.strip()}')
    try:
        snapshot = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError('Windows host query returned malformed JSON') from error
    _process_rows(snapshot, _LAUNCHER_NAME)
    return snapshot
