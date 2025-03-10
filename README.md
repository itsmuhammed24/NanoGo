# Nanogo

<p align="center">
  <img src="img/nanogo.png" alt="Nanogo - Go Board" width="500">
</p>

Nanogo is a Deep Learning project for training a neural network to play Go. The model is trained using self-play games from **Katago**, with a constraint of **fewer than 100,000 parameters** to ensure fair resource usage.

## Project Overview
- The training dataset consists of **1,000,000 self-play games from Katago**.
- Input data includes **31 planes of 19x19** (board states, ladders, color to play, etc.).
- The network predicts:
  - **Policy** (a vector of size 361, indicating the move played).
  - **Value** (close to 1.0 if White wins, close to 0.0 if Black wins).
- Teams can have a **maximum of two students**.

---

## 🚀 Running Nanogo on macOS

### **Why a Special Setup for macOS?**  
When compiling the `golois` library on macOS, using the default system compiler (`clang++`) can cause issues.  
To avoid this, we need to **install LLVM** and explicitly use the correct version of `clang++` provided by LLVM.

---

### **🛠 Step 1: Install LLVM**  
LLVM is required to get the correct version of `clang++`. Install it via **Homebrew**:

```bash
brew install llvm
