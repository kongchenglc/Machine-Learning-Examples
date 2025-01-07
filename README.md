# Simple Gradient Descent Example

This project demonstrates a simple implementation of the gradient descent algorithm to minimize a given objective function. It includes an example of both 1D and 2D gradient descent.

## Installation

To run the example, you'll need Python 3.6+ and the required libraries. You can install the dependencies using `pip`:

1. Clone this repository:
   ```bash
   git clone https://github.com/kongchenglc/Machine-Learning-Examples.git
   cd Machine-Learning-Examples
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Ensure you have the required libraries installed.
2. Run the gradient descent script using:
   ```bash
   python gradient-descent1.py
   ```

The script demonstrates how to minimize the objective function using gradient descent, and visualizes the process of finding the minimum point.

### Example
For 1D gradient descent:
- The objective function is \( J(x) = x^2 - 4x + 4 \)
- The algorithm will attempt to find the minimum of the function.

For 2D gradient descent:
- The objective function is \( J(x, y) = 2x^2 - 0.3x + 3y^2 - 0.8y + 7 \)
- The algorithm will attempt to minimize the function in both \(x\) and \(y\) dimensions.

## Objective Function

The objective function being minimized in this example is:

1. **1D Example**:
   \[
   J(x) = x^2 - 4x + 4
   \]

2. **2D Example**:
   \[
   J(x, y) = 2x^2 - 0.3x + 3y^2 - 0.8y + 7
   \]

## Dependencies

The following Python packages are required:

- `numpy` for numerical operations
- `matplotlib` for plotting the results

You can install them using:

```bash
pip install numpy matplotlib
```