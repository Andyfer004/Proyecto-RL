"""Fixed reset curriculum. No physical, reward or algorithm modifications."""
from .train import validation_score

def phase_at(step, protocol):
    for i, phase in enumerate(protocol['phases'], 1):
        if step < phase['until']:
            return i, phase
    raise ValueError('Presupuesto de currículo agotado')

def choose_scenario(cases, rng, step, protocol):
    _, phase = phase_at(step, protocol)
    levels = list(phase['weights'])
    level = int(rng.choice(levels, p=list(phase['weights'].values())))
    pool = [s for s in cases if s['source_level'] == level]
    if not pool:
        raise ValueError('Nivel sin escenarios')
    return pool[int(rng.integers(len(pool)))]

def annotate_validation(rows, manifest):
    levels = {s['id']: s['source_level'] for s in manifest['scenarios']}
    for row in rows:
        row['source_level'] = levels[row['scenario_id']]

def checkpoint_scores(rows):
    return {level: validation_score([r for r in rows if r['source_level'] == level])
            for level in sorted({r['source_level'] for r in rows})}

def stop_after_level1(runners):
    """Stop only if BOTH agents have no training or validation successes."""
    if len(runners) != 2:
        raise ValueError('Se requieren ambos algoritmos para decidir')
    return all(not any(r['success'] for r in runner['episodes']) and
               not any(r['success'] for r in runner['validation']) for runner in runners)
