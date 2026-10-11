"""Piloto de tiempo: una sesión train, GRU + CTC, sin evaluación de calidad.

    python scripts/medir_tiempo_gru.py --config configs/piloto_tiempo_gru.yaml --mode prepare
    python scripts/medir_tiempo_gru.py --config configs/piloto_tiempo_gru.yaml --mode benchmark --variant grande
    python scripts/medir_tiempo_gru.py --config configs/piloto_tiempo_gru.yaml --mode summary

Consultar docs/prueba_tiempo_entrenamiento.md para ejecutar con límites de RAM
y tiempo. El modo summary no importa PyTorch ni utiliza la GPU.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from time import perf_counter
from urllib.request import urlopen

import numpy as np
import yaml


def ensure_dictionary(root: Path, cfg: dict) -> None:
    """Descarga solo los recursos ausentes y comprueba su SHA-256."""
    dc = cfg['datos']
    for path_key, url_key, hash_key in [
        ('diccionario', 'url_diccionario', 'hash_diccionario'),
        ('licencia', 'url_licencia', 'hash_licencia'),
    ]:
        path = root / dc[path_key]
        if path.exists():
            payload = path.read_bytes()
        else:
            with urlopen(dc[url_key], timeout=dc['timeout_descarga']) as response:
                payload = response.read()
            if hashlib.sha256(payload).hexdigest() != dc[hash_key]:
                raise ValueError(f'El SHA-256 del recurso {path_key} no coincide.')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        if hashlib.sha256(payload).hexdigest() != dc[hash_key]:
            raise ValueError(f'El recurso local {path_key} no coincide con el SHA-256.')


def prepare(repo: Path, root: Path, cfg: dict) -> None:
    """Guarda una muestra real y etiquetas de fonemas; nunca abre test."""
    from scipy.ndimage import gaussian_filter1d
    from src.data.speech import load_session
    ensure_dictionary(root, cfg)
    started = perf_counter()
    dc = cfg['datos']
    dictionary = root / dc['diccionario']
    if hashlib.sha256(dictionary.read_bytes()).hexdigest() != dc['hash_diccionario']:
        raise ValueError('El diccionario no coincide con el recurso verificado.')
    words = {}
    for line in dictionary.read_text().splitlines():
        fields = line.split()
        if len(fields) < 2 or fields[0].startswith(';;;'):
            continue
        word = re.sub(r'\(\d+\)$', '', fields[0]).lower()
        phones = [re.sub(r'\d', '', phone) for phone in fields[1:] if not phone.startswith('#')]
        if word not in words and all(phone in dc['fonemas'] for phone in phones):
            words[word] = phones
    mapping = {phone: index + 1 for index, phone in enumerate(dc['fonemas'])}
    session = load_session(repo / dc['carpeta'] / dc['sesion'])
    nfeatures = cfg['modelo']['entrada']
    sums = np.zeros(nfeatures)
    squares = np.zeros(nfeatures)
    bins = 0
    candidates = []
    skipped = {'sin_texto': 0, 'palabra_desconocida': 0, 'ctc_imposible': 0}
    for trial in session.trials:
        x = trial.features_6v().astype(np.float64)
        sums += x.sum(axis=0)
        squares += np.square(x).sum(axis=0)
        bins += x.shape[0]
        if trial.text is None:
            skipped['sin_texto'] += 1
            continue
        tokens = re.findall(dc['patron_palabras'], trial.text.lower())
        if not tokens or any(token not in words for token in tokens):
            skipped['palabra_desconocida'] += 1
            continue
        target = []
        for token in tokens:
            target.extend(mapping[phone] for phone in words[token])
            target.append(mapping['SIL'])
        output_length = (x.shape[0] - cfg['modelo']['kernel']) // cfg['modelo']['stride'] + 1
        repeats = sum(a == b for a, b in zip(target, target[1:]))
        if output_length < len(target) + repeats:
            skipped['ctc_imposible'] += 1
            continue
        candidates.append((trial.index, target))
    mean = sums / bins
    scale = np.sqrt(np.maximum(squares / bins - mean ** 2, 0))
    scale[scale == 0] = dc['escala_si_cero']
    rng = np.random.default_rng(cfg['experimento']['seed'])
    chosen = rng.choice(len(candidates), size=min(dc['muestras'], len(candidates)), replace=False)
    if len(chosen) < max(v['batch'] for v in cfg['variantes'].values()):
        raise ValueError('Muestra insuficiente para los lotes de la prueba.')
    xs, ys, indices = [], [], []
    for choice in chosen:
        index, target = candidates[int(choice)]
        x = ((session.trials[index].features_6v() - mean) / scale).astype(np.float32)
        x = gaussian_filter1d(x, sigma=dc['suavizado_sigma'], axis=0,
                              truncate=dc['suavizado_truncate']).astype(np.float32)
        xs.append(x)
        ys.append(np.asarray(target, dtype=np.int64))
        indices.append(index)
    np.savez(root / 'sample.npz', x=np.concatenate(xs), y=np.concatenate(ys),
             lengths=np.array([len(x) for x in xs]), target_lengths=np.array([len(y) for y in ys]))
    metadata = {'session': session.name, 'session_trials': len(session.trials),
                'eligible': len(candidates), 'selected': len(chosen), 'indices': indices,
                'skipped': skipped, 'seconds': perf_counter() - started,
                'duration_quantiles_s': np.quantile([len(x) * dc['bin_segundos'] for x in xs],
                                                   [0, .5, .9, 1]).tolist()}
    (root / 'sample.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata), flush=True)


def benchmark(repo: Path, root: Path, cfg: dict, name: str) -> None:
    """Cronometra pasos reales de ajuste y forward; no calcula PER ni WER."""
    if name not in cfg['variantes']:
        raise ValueError(f'Variante desconocida: {name}. Opciones: {list(cfg["variantes"])}')
    if not (root / 'sample.npz').is_file():
        raise ValueError('Primero ejecuta --mode prepare para crear la muestra.')
    import torch
    from torch import nn
    from torch.nn.utils.rnn import pad_sequence
    from src.utils.seed import fijar_seed
    import wandb

    started = perf_counter()
    torch.set_num_threads(cfg['limites']['cpu_threads'])
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA no disponible: no se hace una prueba en CPU por defecto.')
    torch.cuda.set_per_process_memory_fraction(cfg['limites']['fraccion_gpu'], 0)
    fijar_seed(cfg['experimento']['seed'])
    vc = cfg['variantes'][name]
    mc = cfg['modelo']
    with np.load(root / 'sample.npz') as data:
        xs = [torch.from_numpy(x.copy()) for x in np.split(data['x'], np.cumsum(data['lengths'])[:-1])]
        ys = [torch.from_numpy(y.copy()) for y in np.split(data['y'], np.cumsum(data['target_lengths'])[:-1])]

    class TimingGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.days = nn.ModuleList([nn.Linear(mc['entrada'], mc['capa_dia'])
                                      for _ in range(cfg['datos']['sesiones_modelo'])])
            self.dropout = nn.Dropout(mc['dropout_entrada'])
            self.gru = nn.GRU(mc['kernel'] * mc['capa_dia'], vc['unidades'],
                              num_layers=vc['capas'], dropout=mc['dropout_gru'], batch_first=True)
            self.head = nn.Linear(vc['unidades'], len(cfg['datos']['fonemas']) + 1)

        def forward(self, x):
            x = self.dropout(torch.nn.functional.softsign(self.days[0](x)))
            x = x.unfold(1, mc['kernel'], mc['stride']).permute(0, 1, 3, 2).contiguous()
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x, _ = self.gru(x)
            return self.head(x)

    model = TimingGRU().to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['entreno']['learning_rate'])
    criterion = nn.CTCLoss(blank=0, zero_infinity=False)
    run_cfg = dict(cfg)
    run_cfg['variante_medida'] = name
    run = wandb.init(project=cfg['wandb']['proyecto'], mode=cfg['wandb']['modo'],
                     name=cfg['experimento']['nombre'] + '_' + name,
                     config=run_cfg, dir=str(root / 'results'))
    rng = np.random.default_rng(cfg['experimento']['seed'])

    def batch():
        indices = rng.choice(len(xs), size=vc['batch'], replace=False)
        chosen_x = [xs[int(i)] for i in indices]
        chosen_y = [ys[int(i)] for i in indices]
        x = pad_sequence(chosen_x, batch_first=True).to('cuda')
        y = torch.cat(chosen_y).to('cuda')
        lengths = torch.tensor([(len(a) - mc['kernel']) // mc['stride'] + 1 for a in chosen_x])
        target_lengths = torch.tensor([len(a) for a in chosen_y])
        padding = x.shape[0] * x.shape[1] / sum(len(a) for a in chosen_x)
        return x, y, lengths, target_lengths, padding

    def update():
        step_started = perf_counter()
        x, y, lengths, target_lengths, padding = batch()
        torch.cuda.synchronize()
        prepared = perf_counter()
        optimizer.zero_grad(set_to_none=True)
        output = model(x).log_softmax(-1).transpose(0, 1)
        loss = criterion(output, y, lengths, target_lengths)
        if not torch.isfinite(loss):
            raise RuntimeError('CTC no finita.')
        loss.backward()
        gradient = nn.utils.clip_grad_norm_(model.parameters(), cfg['entreno']['clip_gradiente'])
        if not torch.isfinite(gradient):
            raise RuntimeError('Gradiente no finito.')
        optimizer.step()
        torch.cuda.synchronize()
        finished = perf_counter()
        return {'seconds': finished-step_started, 'prepare_seconds': prepared-step_started,
                'compute_seconds': finished-prepared, 'loss': float(loss.detach()), 'padding': padding}

    try:
        print('INICIO ' + name, flush=True)
        model.train()
        for _ in range(cfg['entreno']['warmup']):
            update()
        setup = perf_counter() - started
        torch.cuda.reset_peak_memory_stats()
        measured = []
        train_started = perf_counter()
        for step in range(cfg['entreno']['pasos']):
            if perf_counter() - train_started > cfg['limites']['segundos_entreno']:
                break
            result = update()
            logged = perf_counter()
            run.log(result, step=step)
            result['logging_seconds'] = perf_counter() - logged
            measured.append(result)
            print(f'{name} paso {step+1}: {result["seconds"]:.4f}s', flush=True)
        if not measured:
            raise RuntimeError('No hay pasos medidos.')
        model.eval()
        forward_seconds = []
        with torch.no_grad():
            for _ in range(cfg['entreno']['evaluaciones']):
                x, y, lengths, target_lengths, _ = batch()
                torch.cuda.synchronize()
                before = perf_counter()
                loss = criterion(model(x).log_softmax(-1).transpose(0, 1), y, lengths, target_lengths)
                if not torch.isfinite(loss):
                    raise RuntimeError('CTC de forward no finita.')
                torch.cuda.synchronize()
                forward_seconds.append(perf_counter() - before)
        seconds = [r['seconds']+r['logging_seconds'] for r in measured]
        result = {'name': name, 'config': vc, 'parameters': sum(p.numel() for p in model.parameters()),
                  'gpu': torch.cuda.get_device_name(0), 'precision': 'float32',
                  'setup_and_warmup_seconds': setup, 'steps': len(measured),
                  'mean_step_seconds': float(np.mean(seconds)), 'p90_step_seconds': float(np.quantile(seconds,.9)),
                  'mean_forward_seconds': float(np.mean(forward_seconds)),
                  'mean_padding_ratio': float(np.mean([r['padding'] for r in measured])),
                  'peak_allocated_mb': torch.cuda.max_memory_allocated()/2**20,
                  'peak_reserved_mb': torch.cuda.max_memory_reserved()/2**20,
                  'total_process_seconds': perf_counter()-started,
                  'step_records': measured}
        run.summary.update({k:v for k,v in result.items() if isinstance(v,(int,float,str))})
        (root / 'results' / (name+'.json')).write_text(json.dumps(result, indent=2))
        print('RESULTADO ' + json.dumps({k:v for k,v in result.items() if k != 'step_records'}), flush=True)
    finally:
        run.finish()
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()


def summarize(root: Path, cfg: dict, results_dir: Path) -> None:
    """Recalcula escenarios de duración a partir de pasos ya medidos."""
    records = []
    pc = cfg['proyeccion']
    lines = ['Estimaciones de entrenamiento GRU + CTC', '',
             'Solo entrenamiento; no incluye preparación completa, validación ni checkpoints.',
             f'Margen multiplicativo: {pc["margen"]}. Épocas de esta tabla: {pc["epocas_tabla"]}.',
             'Las épocas son un escenario de cálculo; no se ha comprobado la calidad.', '',
             'Modelo | lote | ms/paso | ' + ' | '.join(f'{n} frases' for n in pc['ensayos'])]
    for name in cfg['variantes']:
        path = results_dir / (name + '.json')
        if not path.exists():
            print(f'Sin medición para {name}; se omite.', flush=True)
            continue
        result = json.loads(path.read_text())
        step_seconds = result['mean_step_seconds']
        if step_seconds <= 0 or not math.isfinite(step_seconds):
            raise ValueError(f'Tiempo inválido en {path}.')
        vc = result['config']
        forecasts = {
            str(n): {str(e): math.ceil(n / vc['batch']) * step_seconds * e * pc['margen']
                     for e in pc['epocas']} for n in pc['ensayos']
        }
        result['training_projections_seconds'] = forecasts
        records.append(result)
        times = [forecasts[str(n)][str(pc['epocas_tabla'])] / 60 for n in pc['ensayos']]
        lines.append(f'{vc["capas"]} x {vc["unidades"]} | {vc["batch"]} | {step_seconds*1000:.1f} | '
                     + ' | '.join(f'{time:.2f} min' for time in times))
    if not records:
        raise ValueError('No hay mediciones para resumir.')
    destination = root / 'results'
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'estimaciones.json').write_text(json.dumps(
        {'proyeccion': pc, 'resultados': records}, indent=2, ensure_ascii=False) + '\n')
    report = '\n'.join(lines) + '\n'
    (destination / 'estimaciones.txt').write_text(report)
    print(report)


def main() -> None:
    """Punto de entrada con rutas explícitas y configuración externa."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--mode', choices=['prepare', 'benchmark', 'summary'], required=True)
    parser.add_argument('--variant')
    parser.add_argument('--results-dir', type=Path,
                        help='Directorio con mediciones existentes para el modo summary.')
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = args.config if args.config.is_absolute() else repo / args.config
    cfg = yaml.safe_load(config_path.read_text())
    root = repo / cfg['salida']
    root.mkdir(parents=True, exist_ok=True)
    (root / 'results').mkdir(exist_ok=True)
    sys.path.insert(0, str(repo))
    if args.mode == 'prepare':
        prepare(repo, root, cfg)
    elif args.mode == 'summary':
        results_dir = args.results_dir or root / 'results'
        if not results_dir.is_absolute():
            results_dir = repo / results_dir
        summarize(root, cfg, results_dir)
    else:
        if args.variant is None:
            parser.error('--mode benchmark requiere --variant.')
        benchmark(repo, root, cfg, args.variant)


if __name__ == '__main__':
    main()
