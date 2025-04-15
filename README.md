```markdown
# Reactive Configuration Space Collision-Free Motion Planning for Mobile Manipulator in Unforeseen Scenarios

This repository contains the official source code for the paper:

**"Reactive Configuration Space Collision-Free Motion Planning for Mobile Manipulator in Unforeseen Scenarios"**

The method enables mobile manipulators to perform real-time, collision-free motion planning in dynamic and unforeseen environments using a reactive, sampling-based control strategy.

## 🌟 Features

- Real-time collision avoidance in the full configuration space
- Reactive adaptation to unforeseen obstacles
- Integration with ROS and full mobile manipulator simulation
- Based on Model Predictive Path Integral Control and Dynamical Systems

---

## 🛠️ Installation

### 1. Clone the Repository

Clone this repo **with submodules** into the `src` directory of your Catkin workspace:

```bash
cd ~/catkin_ws/src
git clone --recursive https://github.com/AboutTime-JNL/Reactive_Planning_for_MM.git
```

### 2. Install System Dependencies

```bash
sudo apt-get install libyaml-cpp-dev libglfw3-dev
```

### 3. Install ROS Dependencies

Use `rosdep` to install ROS-related dependencies:

```bash
cd ~/catkin_ws
rosdep init
rosdep update
rosdep install --from-paths src/Whole_Body --ignore-src -y
```

---

## 🧱 Build

Use the following commands to build the workspace with performance-optimized CMake flags:

```bash
cd ~/catkin_ws
catkin config --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
catkin build
```

---

## 🚀 Running Examples

After building, you can run the demo with the following commands:

```bash
roslaunch src/examples/launch/panda_mobile_control.launch
python3 src/Base/mppi_ds/scripts/mppi_ds.py
python3 src/examples/scripts/esay_test.py
```

---

## 📄 Citation

If you find this work useful, please consider citing the paper:

```bibtex
@inproceedings{your_paper_citation,
  title={Reactive Configuration Space Collision-Free Motion Planning for Mobile Manipulator in Unforeseen Scenarios},
  author={Ninglong Jin, Guangbao Zhao, Jianhua Wu, Zhenhua Xiong ,
  booktitle={...},
  year={2025}
}
```

---

## 📬 Contact

For questions, please open an issue or contact the authors.
```
