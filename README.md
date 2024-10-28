# Intent-based Prompt Calibration (IPC)

A system for automatic prompt optimization using Claude 3 models. IPC enables dynamic prompt refinement through iterative testing and improvement cycles, particularly useful for complex tasks like content moderation.

## Features

- Automatic prompt optimization through systematic testing
- Synthetic test case generation for edge cases
- Comprehensive error analysis and tracking
- Multi-model architecture using Claude 3 models
- Simple configuration and usage

## Quick Start

```python
from ipc_system import AdvancedIPCConfig, EnhancedIPCSystem
import anthropic

# Initialize client
client = anthropic.Anthropic(api_key="your-api-key")

# Configure system
config = AdvancedIPCConfig(
    task_description="Your task description",
    labels=["label1", "label2"],
    initial_prompt="Your initial prompt"
)

# Run IPC
ipc = EnhancedIPCSystem(client, config)
best_prompt, history = ipc.calibrate()



[Intent-based Prompt Calibration: Enhancing prompt
optimization with synthetic boundary cases] Research paper: https://arxiv.org/pdf/2402.03099 
