# A2CMP: Pedestrian-Aware Traffic Signal Control with SUMO

A research prototype for traffic-signal control that combines **Advantage Actor-Critic (A2C)** with **max-pressure guidance** in a SUMO simulation.

The project extends the [SUMO-RL](https://github.com/LucasAlegre/sumo-rl) environment with pedestrian-aware phase selection, vehicle and pedestrian pressure signals, traffic-emission observations, and queue/CO₂-oriented rewards.

## What this repository explores

- A2C-based control with Stable-Baselines3
- Vehicle max-pressure signals derived from incoming and outgoing lanes
- Pedestrian pressure derived from walking areas and intended directions
- Dynamic switching between reduced and full phase sets
- Observations containing phase, density, queue, and normalized emission information
- Reward shaping using queue length, CO₂ estimates, waiting time, and pressure consistency
- Single-intersection SUMO scenarios for repeatable experiments

> **Status:** research prototype. Several parameters, phase mappings, and network identifiers are scenario-specific. Review the network files and signal configuration before applying the code to another intersection.

## Repository map

- `run3_a2cmp.py` — A2C training entry point for the included two-way intersection
- `sumo_rl/environment/newsignal.py` — pedestrian/vehicle pressure logic and reward shaping
- `sumo_rl/environment/traffic_signal.py` — traffic-signal state, observation, reward, and phase control
- `nets/` — SUMO network and route definitions
- `outputs/` — experiment outputs and plotting utilities

## Requirements

- Python 3.9+
- [SUMO](https://sumo.dlr.de/docs/Installing/index.html)
- Gymnasium
- NumPy
- pandas
- Stable-Baselines3
- TraCI / sumolib

Set `SUMO_HOME` before running an experiment:

```bash
export SUMO_HOME=/path/to/sumo
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"
```

Install the Python dependencies used by the training script:

```bash
pip install gymnasium numpy pandas stable-baselines3 traci sumolib
```

## Run the included experiment

The current entry point uses the network and route files under `nets/2way-single-intersection/` and opens the SUMO GUI:

```bash
python run3_a2cmp.py
```

The script trains an A2C `MlpPolicy` for 14,400 timesteps and writes simulation output under `outputs/`.

For headless runs, change `use_gui=True` to `use_gui=False` in the experiment configuration.

## Control design

At each decision step, the environment can use:

1. lane occupancy and queue information;
2. normalized lane-level CO₂ emissions;
3. vehicle pressure from incoming versus downstream traffic;
4. pedestrian demand and walking direction;
5. a consistency bonus when the selected phase agrees with the pressure-derived reference phase.

The included phase indices and lane IDs are tied to the bundled intersection. They should be moved into configuration before using a different SUMO network.

## Reproducibility notes

- Record the SUMO and Python package versions used for each experiment.
- Keep random seeds and evaluation demand files separate from training demand.
- Compare against fixed-time, max-pressure-only, and A2C-only baselines.
- Report vehicle delay, pedestrian waiting time, queue length, throughput, and emissions together.

## Attribution

This repository is based on [LucasAlegre/sumo-rl](https://github.com/LucasAlegre/sumo-rl) and adapts it for pedestrian-aware A2C and max-pressure experiments. Please retain the upstream license and attribution when reusing the code.
