#ifndef MODEL_H
#define MODEL_H

// Standard library includes
#include <vector> // For dynamic arrays (weights, biases, hidden state)
#include <string> // For handling text and file paths

/*
 * MulengaLM
 * A minimal character-level language model for text generation.
 * This class defines the structure and interface for a simple
 * neural network that predicts the next character given an input.
 * 
 * The model is designed for CPU-only inference and educational use.
*/
class MulengaLM{
    public:
    /*
    * Constructor
    * Initializes the model with a given vocabulary size
    * and hidden layer size.
    * 
    * @param vocab_size Number of unique characters in the vocabulary
    * @param hidden_size Number of neurons in the hidden layer
    */
    MulengaLM(int vocab_size, int hidden_size);

   /*
    * load_weights
    * Loads pretrained model wights from a file
    * This allows the model to run inference without retraining.
    * 
    * @param file Path to weights file
   */
   void load_weights(const std::string& file);

  /*
   * predict
   * Predicts the next character based on a single input character.
   * 
   * @param input_clear Input character
   * @return Predicted next character
  */
   char predict(char input_char);

 /*
  * generate
  * Generates a sequence of characters starting from a seed character.
  * 
  * @param seed Starting character for generation
  * @param length Number of characters to generate
  * @return Generated text string
 */
  std::string generate(char seed, int length);

private:
   // Size of the character vocabulary
   int vocab_size;

   // Number of neurons in the hidden layer
   int hidden_size;

   // Weight matrix from input to hidden layer
   // Shape: [hidden_size][vocab_size]
   std::vector<std::vector<float>> Wxh;

   // Weight matrix from hidden layer to output layer
   // Shape: [vocab_size][hidden_size]
   std::vector<std::vector<float>> Why;

   // Bias vector for the hidden layer
   std::vector<float> bh;

   // Bias vector for the output layer
   std::vector<float> by;

   // Hidden state vector (stores intermediate activations)
   std::vector<float> h;

   /*
    * char_to_index
    * Converts a character into its corresponding vocabulary index.
    * @param c Input character
    * @param Integer index of the character
   */
   int char_to_index(char c);

   /*
    * index_to_char
    * Converts a vocabulary index back into its character representation.
    * @param i Vocabulary index
    * @return Corresponding character
   */
   char index_to_char(int i);

   /*
    * softmax
    * Applies the softmax function to convert raw scores (logits)
    * into probabilities.
    * @param x vector of logits
    * @return Vector of probabilities summing to 1
   */
    std::vector<float> softmax(const std::vector<float>& x);

};

#endif // MODEL_H



}