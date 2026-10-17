#!/usr/bin/env python3
"""Check this handoff's internal consistency; not a native Todo or GPU validator."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'planning/volta-bootstrap/'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-manifest', action='store_true')
    args = parser.parse_args()
    errors: list[str] = []
    checks: list[str] = []
    warnings: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    def load(name: str):
        return json.loads((ROOT / name).read_text(encoding='utf-8'))

    def safe(path: str) -> bool:
        p = PurePosixPath(path)
        return bool(path) and not p.is_absolute() and '..' not in p.parts

    def within(path: str, scopes: list[str]) -> bool:
        return any(path == scope or path.startswith(scope.rstrip('/') + '/') for scope in scopes)

    try:
        plan = load('planning/kernelwiki.todo-plan.json')
        local = load('sources/local-inputs.json')['inputs']
        external = load('sources/external-sources.json')['sources']
        adoption = load('seed/ADOPTION.json')
        for p in ROOT.rglob('*.json'):
            json.loads(p.read_text(encoding='utf-8'))
        checks.append('JSON files parse')

        source_ids = [row['id'] for row in local + external]
        require(len(source_ids) == len(set(source_ids)), 'Duplicate source ledger IDs')
        for row in local:
            require(safe(row['path']), f"Unsafe local source path: {row['id']}")
            require(bool(re.fullmatch('[0-9a-f]{64}', row['content_sha256'])), f"Invalid source SHA256: {row['id']}")
        for row in external:
            require(urlsplit(row['url']).scheme == 'https', f"Invalid external URL: {row['id']}")
        checks.append('Source IDs, locators and hash syntax')

        require(plan['schema_version'] == 3, 'Expected native plan schema 3')
        tasks = plan['tasks']
        ids = [t['id'] for t in tasks]
        require(len(ids) == len(set(ids)), 'Duplicate task IDs')
        by_id = {t['id']: t for t in tasks}
        gate_ids = []
        edges: dict[str, list[str]] = {}
        for task in tasks:
            tid = task['id']
            require(bool(task.get('objective')) and bool(task.get('title')), f'Missing task purpose: {tid}')
            if task.get('parent_id'):
                require(task['parent_id'] in by_id, f'Unknown parent: {tid}')
            scope = task.get('scope', {})
            for field in ('exclusive_paths', 'read_paths', 'forbidden_paths'):
                require(all(safe(p) for p in scope.get(field, [])), f'Unsafe scope: {tid}/{field}')
            allowed = scope.get('exclusive_paths', []) + scope.get('read_paths', [])
            edges[tid] = []
            for dep in task.get('depends_on', []):
                require(dep['type'] == 'task', f'Unexpected dependency form: {tid}')
                target = dep['task_id']
                require(target in by_id, f'Unknown dependency {target}: {tid}')
                require(target != task.get('parent_id'), f'Aggregate incorrectly blocks child: {tid}')
                edges[tid].append(target)
            for ref in task.get('references', []):
                require(ref.startswith(PREFIX) and (ROOT / ref[len(PREFIX):]).is_file(), f'Unresolved package reference: {tid}: {ref}')
            for gate in task.get('gates', []):
                gate_ids.append(gate['id'])
                if gate['type'] == 'command':
                    argv = gate.get('argv')
                    require(isinstance(argv, list) and bool(argv) and all(isinstance(v, str) for v in argv), f'Invalid argv: {gate["id"]}')
                    require(gate.get('cwd') == '.', f'Unexpected gate cwd: {gate["id"]}')
                    require(bool(gate.get('input_paths')), f'Unscoped command gate: {gate["id"]}')
                    for p in gate.get('input_paths', []):
                        require(safe(p) and within(p, allowed), f'Gate input outside declared scope: {gate["id"]}: {p}')
                elif gate['type'] == 'task_state':
                    require(gate.get('task_id') in by_id and gate.get('status') == 'done', f'Invalid aggregate gate: {gate["id"]}')
                else:
                    errors.append(f'Unexpected gate type: {gate["id"]}')
        require(len(gate_ids) == len(set(gate_ids)), 'Duplicate gate IDs')
        visiting, visited = set(), set()
        def visit(tid: str) -> None:
            if tid in visiting:
                raise ValueError(f'Dependency cycle at {tid}')
            if tid in visited or tid not in edges:
                return
            visiting.add(tid)
            for dep in edges[tid]:
                visit(dep)
            visiting.remove(tid)
            visited.add(tid)
        for tid in ids:
            visit(tid)
        for run in plan['runs']:
            lanes = run['lanes']
            require(sum(not lane.get('parent_lane_id') for lane in lanes) == 1, 'Run must have one root lane')
            assigned = [tid for lane in lanes for tid in lane['tasks']]
            require(len(assigned) == len(set(assigned)) and set(assigned) == set(ids), 'Tasks must be assigned once in the run')
            require(run['root_task_id'] in by_id, 'Unknown run aggregate')
            for ref in run['charter'].get('references', []):
                require(ref.startswith(PREFIX) and (ROOT / ref[len(PREFIX):]).is_file(), f'Unresolved charter reference: {ref}')
        checks.append('Plan IDs, DAG, references, gate scope and lane assignment (structural only)')

        for p in ROOT.rglob('*.md'):
            text = p.read_text(encoding='utf-8')
            for group in re.findall(r'\[([^\]\n]+)\](?!\()', text):
                for sid in re.findall(r'\b(?:KW|SK|AT|PC|EXT|CAND)-[A-Z0-9-]+\b', group):
                    require(sid in source_ids, f'Unresolved citation {sid}: {p.relative_to(ROOT)}')
            for raw in re.findall(r'\[[^\]\n]*\]\(([^)]+)\)', text):
                parts = urlsplit(raw)
                if parts.scheme or raw.startswith('#'):
                    continue
                target = (p.parent / unquote(parts.path)).resolve()
                require(target.is_relative_to(ROOT) and target.exists(), f'Broken local link {raw}: {p.relative_to(ROOT)}')
        checks.append('Markdown local links and cited ledger IDs')

        source_files = list((ROOT/'seed/sources').rglob('*.md'))
        hardware_files = list((ROOT/'seed/wiki/hardware').glob('*.md'))
        require(len(source_files) == adoption['source_pages'], 'Seed source count mismatch')
        require(len(hardware_files) == adoption['hardware_pages'], 'Seed hardware count mismatch')
        try:
            import yaml
        except ImportError:
            warnings.append('PyYAML not installed: seed frontmatter checks skipped; install PyYAML to include them')
        else:
            pages = {}
            for p in source_files + hardware_files:
                match = re.match(r'^---\n(.*?)\n---\n', p.read_text(encoding='utf-8'), re.S)
                require(bool(match), f'Missing seed frontmatter: {p.name}')
                if not match:
                    continue
                fm = yaml.safe_load(match.group(1))
                require(fm['id'] not in pages, f'Duplicate seed page ID: {fm["id"]}')
                pages[fm['id']] = fm
                required = ('id','title','architectures','tags')
                if p in hardware_files:
                    required += ('type','confidence','related','sources','aliases')
                    require(fm.get('confidence') == 'source-reported' and fm.get('reproducibility') == 'concept', f'Unqualified maturity claim in seed: {p.name}')
                else:
                    required += ('url','source_category','retrieved_at')
                require(all(k in fm and fm[k] is not None for k in required), f'Missing seed fields: {p.name}')
            seed_sources = {sid for sid in pages if sid.startswith(('doc-', 'blog-'))}
            for sid, fm in pages.items():
                require(all(v in seed_sources for v in fm.get('sources', [])), f'Unresolved seed source: {sid}')
                require(all(v in pages for v in fm.get('related', [])), f'Unresolved seed related page: {sid}')
            require(pages['doc-volta-gdn']['architectures'] == [], 'Algorithm-only GDN source must not imply SM70 execution')
            checks.append('Seed field presence, IDs, links, draft maturity and algorithm/hardware boundary (not full native schema)')

        if args.verify_manifest:
            manifest = ROOT/'MANIFEST.sha256'
            listed = set()
            for line in manifest.read_text(encoding='utf-8').splitlines():
                digest, name = line.split('  ', 1)
                require(safe(name), f'Unsafe manifest name: {name}')
                listed.add(name)
                p = ROOT/name
                require(p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == digest, f'Manifest mismatch: {name}')
            actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and p.name != 'MANIFEST.sha256' and '__pycache__' not in p.parts}
            require(listed == actual, 'Manifest file coverage mismatch')
            checks.append('Manifest hashes and file coverage')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f'{type(exc).__name__}: {exc}')

    result = {
        'status': 'passed' if not errors else 'failed',
        'checks': checks, 'warnings': warnings, 'errors': errors,
        'limits': ['No native Todo validator or application executed',
                   'No full KernelWiki native validation executed',
                   'No CUDA compilation, GPU execution or performance measurement executed'],
    }
    print(json.dumps(result, indent=2))
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
