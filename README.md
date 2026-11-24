# Refactored Brocolli 

A computer vision project that recreates a teleportation effect using real-time webcam capture, human segmentation, and particle animations.

## Features
- Real-time webcam capture using OpenCV
- Human segmentation using MediaPipe
- Background capture and replacement
- Blue energy aura effect
- Particle-based disappearance
- Particle-based reappearance

## Tech Stack
- Python
- OpenCV
- MediaPipe
- NumPy

## Project Structure
```text
refactored-brocolli/
├── background/
├── effects/
├── segmentation/
├── models/
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/negative-one-1/refactored-brocolli.git
cd refactored-brocolli

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

Place `selfie_segmenter.tflite` inside the `models/` directory.

## Run

```bash
python effects/teleportation.py
```

## Controls

| Key | Action |
|---|---|
| B | Capture background |
| SPACE | Teleport |
| R | Reappear |
| Q | Quit |

## Author
**Pradeep Bhandari**


## Project Status
Actively under development.
