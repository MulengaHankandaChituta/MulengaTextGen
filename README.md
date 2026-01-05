# MulengaTextGen

MulengaTextGen is a **C++ character-level language model** designed for CPU-based text generation.  
This lightweight project demonstrates how to implement a simple neural language model entirely in C++, capable of predicting the next character and generating text sequences.

---

## Features

- **Character-level text generation**  
  Generates text one character at a time using a hidden layer neural network.
- **CPU optimized**  
  Lightweight implementation suitable for low-end hardware (tested on Dell Latitude 3340).
- **No external ML frameworks required**  
  Pure C++ implementation using standard libraries.
- **Probabilistic sampling**  
  Uses softmax to sample next character, allowing variable and creative output.



## Project Structure

MulengaTextGen/
├── main.cpp # Entry point for running text generation
├── model.h # Header file defining the MiniLM class
├── model.cpp # Implementation of the MiniLM class
└── weights.txt # Optional: pre-trained weights for text generation



## Getting Started

### Prerequisites

- C++ compiler (g++, clang++, or Visual Studio)
- Basic knowledge of compiling C++ projects

### Compile and Run


# Compile the project
g++ -O2 main.cpp model.cpp -o MulengaTextGen

# Run the program
./MulengaTextGen
Example Output
kotlin

hello worldhello mulenga learning ai is fun...
Output varies due to probabilistic sampling.

Tech Stack
C++

STL

CPU-based neural network implementation

Author
Mulenga Chituta

GitHub: YourGitHubUsername

LinkedIn: YourLinkedInProfile

License
This project is open-source and free to use under the MIT License.


