# Station 60 Fault Detection

This repository contains two independent machine-learning implementations for reactive fault detection on the AutFab smart-factory Station 60 pneumatic press.

## Repository layout

```
modules/
├── Aeon-Implementation/    # Aeon-based pipeline
└── Sktime-Implement/       # sktime-based MiniROCKET pipeline
```

The modules deliberately have separate source code, dependency lists, data utilities, and result files. They do not share runtime code, so changes to one implementation do not affect the other.

## Run an implementation

Choose one module, open a terminal in that directory, install its dependencies, and use the module-specific entry point or instructions:

```powershell
cd modules/Aeon-Implementation
pip install -r requirements.txt
python main.py
```

For the sktime implementation, see `modules/Sktime-Implement/run.txt` and `Auto_fab_Minirocket.py`.

## Notes

- Each module keeps its own `requirements.txt` because their dependencies differ.
- Generated IDE files, Python caches, virtual environments, and local shortcuts are ignored at repository level.
