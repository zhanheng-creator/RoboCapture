<div align="center">

<h1> RoboCapture: A Unified Pipeline for Automated Robotic Data Collection </h1>

</div>

## Overview

Framework overview for RoboCapture. RoboCapture is adept at acquiring images from a myriad of simulators and real-world scenarios, autonomously generating diverse multi-level tasks and planning them based on MLLMs. RoboCapture modularly selects and integrates skills, orchestrating the execution of tasks across various hardware platforms through middleware. With limited human supervision, RoboCapture performs real-time analysis of task status and failure causes, and collects multidimensional data, as illustrated in module 6.

<img title="" src="./fig/fig2.jpg" alt="fig2.jpg" data-align="center">

## Usage

Our code is divided into five parts: task generation and task planning, instruction fine-tuning, simulator data collection, real-world data collection, and the RT-1 model. The usage tutorial and code are currently being organized. Here are the usage tutorials for the different modules in our paper.

- [task generation and task planning](./md/task_generation_and_task_planning.md)

- [instruction fine-tuning](./md/instruction_fine-tuning.md)

- [simulator data collection](./md/simulator_data_collection.md)

- [real-world data collection](./md/real-world_data_collection.md)

- [RT-1](./md/RT-1.md)

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Acknowledgment

We extend our gratitude to the open-source efforts of [LLaVA](https://github.com/haotian-liu/LLaVA), [Yi](https://github.com/01-ai/Yi),[Reproducing rt-1 in pytorch](https://github.com/ioai-tech/pytorch_rt1_with_trainer_and_tester) and [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).
