# Shadow Dungeon Evolution-based Gaming AI

This is a standalone AI for Shadow Dungeon (action-RPG) using neural networks and evolution mechanics on Linux Desktop.

## Features
- Neural network-based decision making
- Genetic algorithms for character/creature evolution
- Integration with Shadow Dungeon game mechanics
- Training system for neural networks

## Structure
- `capture/`: 60 FPS video (ffmpeg) + evdev input recording
- `training/`: dataset (30 FPS temporal windows), CNN+GRU policy, training loop
- `game_integration/`: live play (mss screenshots, UInput virtual controller)
- `neural_networks/`: original pure-Python NN scaffold (kept)
- `evolution/`: original GA scaffold (kept, for a later RL/evolution stage)
- `utils/`: config, logging
- `main.py` (project root): CLI — capture / build / train / play / devices

See the top-level `README.md` for the full pipeline and usage.

## Installation
1. Clone this repository
2. Install dependencies (see requirements.txt)
3. Configure game integration settings
4. Run the AI

## Usage
The AI will evolve characters/creatures over multiple generations, training neural networks for optimal gameplay.

## Contributing
Contributions are welcome! Please submit pull requests with detailed descriptions.

## License
MIT License
