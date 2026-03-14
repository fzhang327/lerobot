# Decoupled Admittance Control via Reduced-Order ESO 🤖

![Status](https://img.shields.io/badge/Status-Work_in_Progress-orange)
![Robotics](https://img.shields.io/badge/Robotics-Embodied_AI-blue)
![Framework](https://img.shields.io/badge/Integration-LeRobot-yellow)

A lightweight, sensorless admittance control middleware designed for **black-box position-controlled manipulators**. 

This project solves the fundamental "Compliance Paradox" in Embodied AI: achieving high-bandwidth trajectory tracking for Vision-Language-Action (VLA) models while maintaining extreme physical compliance during contact-rich tasks, all **without** expensive Force/Torque (F/T) sensors.

## 🚀 Visual Comparison

| Baseline: Pure Position Servo | Ours: Force-Scaled Admittance + 2nd-Order LESO |
| :---: | :---: |
| <img src="media/readme/baseline_rigid.gif" width="400"/> | <img src="media/readme/ours_compliant.gif" width="400"/> |
| *Hard contact* | *High-bandwidth tracking in free space + butter-smooth yielding upon physical contact.* |



## ✨ Key Features

* **Singular Perturbation Decoupling:** Bypasses the acceleration-masking effect of proprietary high-gain PID loops by formulating a 1st-order kinematic equivalent plant.
* **Position-Driven 2nd-Order LESO:** A (reduced-order) Linear Extended State Observer that cleanly extracts external interaction wrenches with zero phase lag, highly resilient to encoder noise.
* **Force-Scaled Admittance Law:** Introduces a scaling matrix $\alpha$ to physically decouple the tracking bandwidth from the contact compliance, eliminating the trade-off between sluggish tracking and rigid contact.
* **LeRobot Integration:** Designed as a drop-in middleware for the HuggingFace `lerobot` framework, ready to bridge VLA policies (like Pi0, OpenVLA) with low-level hardware.

## 🚧 Code Status: Coming Soon!

The underlying mathematics have been validated, and we are currently cleaning up the Python implementation (including the semi-implicit Euler discrete integrator and teleoperation dashboard). 

**The full source code, hardware setup instructions, and tuning guides will be uploaded shortly. Stay tuned!**

---
*For researchers: A paper detailing the theoretical proofs (Lyapunov stability, passivity analysis of force-scaling) is currently in preparation.*
