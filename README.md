# How Small Can the Orchestrator Be?

Code, results, and provenance for the paper:

> **How Small Can the Orchestrator Be? From LLM Agents to Tiny DQNs for
> Energy- and Latency-Aware Task Offloading in IoT Edge Computing**
> Ege Erberk Uslu, Orhan Dağdeviren — Ege University, İzmir, Türkiye

A footprint-aware, size-spectrum study of the offloading orchestrator: DQN/PPO
agents shrunk to under a thousand parameters at one end, a zero-shot LLM agent
at the other, all evaluated on one Gymnasium benchmark against a strong myopic
Greedy baseline.

## Layout

```
env/              Gymnasium offloading environment (Markov channel, edge queue, battery)
agents/           DQN/PPO training, heuristics, LLM agent (Ollama)
experiments/      experiment drivers (evaluate.py, lightweight.py, robustness, scarce-energy, real traces)
results/          per-run CSVs — every number in the paper regenerates from these
figures.py        builds all paper figures from results/
inject_numbers.py regenerates paper macros (numbers.tex) from results/
run_all.py        end-to-end entry point
tests/            unit tests
ASSUMPTIONS.md    modeling assumptions, stated explicitly
```

## Reproducing

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_all.py          # full study; DRL training seeds {0..4}, eval episodes seeds 1000-1099
python figures.py          # regenerate all figures from results/
```

The LLM-agent experiments require a local [Ollama](https://ollama.com) server
with `llama3.1:8b` pulled. Real-trace replay uses the public ClarkNet and NASA
HTTP traces from the Internet Traffic Archive
(<http://ita.ee.lbl.gov/html/traces.html>); place the gzipped logs under `data/`.

## License

MIT — see `LICENSE`.
