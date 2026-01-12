#include <iostream>
#include "model.h"

using namespace std;

int main() {
    MulengaLM model(26, 32);
    cout << model.generate('h', 100) << endl;

    return 0;

}