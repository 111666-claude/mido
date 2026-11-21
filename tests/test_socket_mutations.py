# SPDX-FileCopyrightText: 2026
#
# SPDX-License-Identifier: MIT

"""Prove the loopback suite catches deliberately broken chain links.

Each variant in ``tests/mutation/socket_mutations.py`` is applied inside a
fresh pytest subprocess (via ``-p``) that runs the real
``test_socket_loopback.py`` module.  We assert two things:

1. the suite *fails* under the mutation (a test that can not fail proves
   nothing), and
2. the failing test names localise the defect to the intended link in the
   chain (fragmentation/reassembly, EOF cleanup, or broadcast fan-out).

A baseline subprocess with no mutation must still pass, guarding against
the harness itself breaking the run.
"""

import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOOPBACK_TEST = os.path.join(REPO_ROOT, 'tests', 'test_socket_loopback.py')
PLUGIN = os.path.join(REPO_ROOT, 'tests', 'mutation')

# mutation name -> substrings that must appear among the failing node
# ids.  Each maps the failure to a specific link of the chain.
EXPECTED_FAILURES = {
    'frag_parser_reset': [
        'split_across_segments',
        'message_boundaries_and_order',
        'realtime_message_inside_sysex',
    ],
    'eof_no_close': [
        'client_disconnect',
        'server_close_propagates',
    ],
    'broadcast_first_only': [
        'broadcast_to_multiple_clients',
    ],
}


def _run_loopback(mutation=None):
    """Run the loopback module in a clean subprocess.

    Returns ``(returncode, stdout, stderr)``.
    """
    env = dict(os.environ)
    pythonpath = [REPO_ROOT, PLUGIN]
    if env.get('PYTHONPATH'):
        pythonpath.append(env['PYTHONPATH'])
    env['PYTHONPATH'] = os.pathsep.join(pythonpath)
    if mutation:
        env['MIDO_SOCKET_MUTATION'] = mutation
    else:
        env.pop('MIDO_SOCKET_MUTATION', None)

    cmd = [sys.executable, '-m', 'pytest', LOOPBACK_TEST,
           '-rA', '--no-header', '-p', 'socket_mutations']
    return subprocess.run(cmd, env=env, cwd=REPO_ROOT,
                          capture_output=True, text=True, timeout=180)


def _failing_node_ids(result):
    """Extract the node ids pytest marked FAILED from the -rA summary."""
    nodes = []
    for line in result.stdout.splitlines():
        if line.startswith('FAILED '):
            nodes.append(line.split()[1])
    return nodes


def test_unmutated_loopback_suite_passes():
    """The plugin with no mutation must leave the suite green."""
    result = _run_loopback(mutation=None)
    assert result.returncode == 0, (
        'baseline loopback run failed:\n'
        f'stdout:\n{result.stdout}\nstderr:\n{result.stderr}')


@pytest.mark.parametrize('mutation, expected_nodes',
                         list(EXPECTED_FAILURES.items()),
                         ids=list(EXPECTED_FAILURES))
def test_mutation_makes_suite_fail(mutation, expected_nodes):
    result = _run_loopback(mutation=mutation)

    assert result.returncode != 0, (
        f'mutation {mutation!r} did not break the suite, but it should have\n'
        f'stdout:\n{result.stdout}')

    failed = ' '.join(_failing_node_ids(result))
    assert failed, (
        f'could not parse failing test names for {mutation!r}\n'
        f'stdout:\n{result.stdout}')

    missing = [node for node in expected_nodes if node not in failed]
    assert not missing, (
        f'mutation {mutation!r} should have failed tests locating the '
        f'defect to the affected link, but {missing} did not fail.\n'
        f'failing nodes: {failed}')
