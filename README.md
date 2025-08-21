# Multi-Agent Reinforcement Learning for Cooperative Robotics

This project investigates and implements **Multi-Agent Reinforcement Learning (MARL)** to enable coordinated decision-making in complex, partially observable environments.  
We developed a custom **drone delivery simulator** and benchmarked algorithm performance on the **PettingZoo Simple Spread v3** environment.

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![Libraries: PettingZoo, PyTorch, NumPy](https://img.shields.io/badge/libs-PettingZoo%20%7C%20PyTorch-orange.svg)


---

## 📜 Project Overview

This project explores Multi-Agent Reinforcement Learning (MARL) across two distinct scenarios:

1. **Custom Drone Delivery System:**  
   A multi-agent environment where two drones must cooperate to deliver packages across a 6x6 grid, navigating no-fly zones and avoiding collisions.  

2. **Simple Spread v3 (PettingZoo):**  
   A standard cooperative MARL benchmark where agents learn to cover landmarks while avoiding each other.  

The objective was to **implement, compare, and analyze** the performance of various **tabular** and **deep reinforcement learning** algorithms in cooperative, partially observable settings.

---

## 🤖 Algorithms Implemented

A wide range of MARL algorithms were implemented and benchmarked across the two environments.

### In the Custom Drone Delivery Environment:
* **Tabular Methods:**
  - **Q-Learning**: Learned optimal policies via value iteration.  
  - **SARSA**: Trained agents using on-policy updates.  
  - **Double Q-Learning**: Reduced overestimation bias by decoupling action selection and evaluation.  

* **Deep RL Methods:**
  - **DQN**: Used a centralized neural network to approximate a joint Q-function.  
  - **QMIX**: Enabled Centralized Training with Decentralized Execution (CTDE) using a mixing network.  

### In the Simple Spread v3 Environment:
* **Deep RL Methods:**
  - **DQN**: Applied deep Q-networks for decentralized cooperative action learning.  
  - **Double DQN**: Improved stability by mitigating Q-value overestimation.  
  - **Dueling DQN**: Separated state value and advantage streams to enhance learning efficiency.  

---

## 📊 Key Results & Findings

- **Tabular Methods Performance:** In the drone environment, all tabular methods successfully learned cooperative policies. Q-Learning achieved the highest final reward (~452) and converged the fastest (~120 episodes). SARSA and Double Q-Learning converged more conservatively (~438).  

- **Deep RL in Custom Environment:** QMIX outperformed centralized DQN, showing faster convergence and more stable rewards by addressing the credit assignment problem.  

- **Deep RL in Benchmark Environment:** In the Simple Spread benchmark, the final performance ranking was **Dueling DQN > Double DQN > DQN**, with Dueling DQN achieving the most stable convergence and highest average reward.  

- **CTDE Effectiveness:** The success of QMIX confirmed that **Centralized Training with Decentralized Execution (CTDE)** is highly effective for cooperative multi-agent systems under partial observability.  

---

## 🔧 Setup & Usage

The project is organized into two main folders, one for each environment.

### 1. Clone the repository:
```bash
git clone https://github.com/AishwaryaVirigineni/RL_Project.git
cd RL_Project
```

### 2. Navigate to an environment folder:
```bash
cd "Custom_MARL-Drone Delivery"
# OR
cd "Existing_MARL-SimpleSpreadv3"
```

### 3. Install dependencies:
```bash
# Recommended: create a Python virtual environment
pip install -r requirements.txt
# Common libraries: pettingzoo, numpy, matplotlib, torch, gymnasium
```

### 4. Run the experiments:
- For the **Drone Delivery environment**, open and run:
  - `Tabular_Methods_DroneDeliveryEnv.ipynb` for tabular methods  
  - `DQN_DroneDeliveryEnv.ipynb` or `QMIX_DroneDeliveryEnv.ipynb` for deep RL  

- For the **Simple Spread environment**, open and run:
  - `SimpleSpreadV3_MARL.ipynb`  

---

## 📁 Directory Structure
```
RL_Project/
├── Custom_MARL-Drone Delivery/      # Code for the custom drone environment
│   ├── codes/                       # Algorithms (Q-Learning, SARSA, etc.)
│   ├── images/                      # Rendering assets
│   ├── pickle/                      # Saved Q-tables, model weights, logs
│   ├── Rendering/                   # PDF visualizations
│   ├── Tabular_Methods_DroneDeliveryEnv.ipynb
│   ├── DQN_DroneDeliveryEnv.ipynb
│   └── QMIX_DroneDeliveryEnv.ipynb
│
├── Existing_MARL-SimpleSpreadv3/    # PettingZoo benchmark environment
│   ├── agents/                      # DQN, DoubleDQN, DuelingDQN
│   ├── dqn_models/                  # Saved models for DQN
│   ├── double_dqn_models/           # Saved models for DoubleDQN
│   ├── dueling_dqn_models/          # Saved models for DuelingDQN
│   ├── gifs/                        # Visualizations of runs
│   └── SimpleSpreadV3_MARL.ipynb
│
└── REPORT.pdf  # Final project report
```

---

## 👥 Team & Contributions
This project was a collaborative effort for **CSE 4/546: Reinforcement Learning** at the University at Buffalo (Spring 2025).  

- **Aishwarya Virigineni**  
- **Nithya Kaandru**  
- **Prajesh Gupta Vizzapu**  

All members contributed equally to environment design, algorithm implementation, experimentation, and report writing.  

---
