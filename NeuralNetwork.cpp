#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <fstream>
#include <vector>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

using namespace std;
using namespace Eigen;

double activationFunction(double x){
    return 1 / (1 + exp(-x));
}

double scaleData(double x){
    return (((255-x)/255) * 0.99)+ 0.01;
}

MatrixXd reshape(MatrixXd mat , int rows , int cols){
    MatrixXd matB(1 , rows * cols);
    int ind = 0;
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            matB(0,ind) =  mat(i,j);
            ind++;
        }
    }
    return matB;
}

void saveData(string fileName, MatrixXd  matrix){
    IOFormat CSVFormat(FullPrecision, DontAlignCols, ",", "\n");
 
    ofstream file(fileName);
    if (file.is_open())
    {
        file << matrix.format(CSVFormat);
        file.close();
    }
}

MatrixXd loadData(string fileToOpen){
    vector<double> matrixEntries;
    ifstream matrixDataFile(fileToOpen);
    string matrixRowString;
    string matrixEntry;
    int matrixRowNumber = 0;
    
    while (getline(matrixDataFile, matrixRowString)){
        stringstream matrixRowStringStream(matrixRowString); 
 
        while (getline(matrixRowStringStream, matrixEntry, ',')){
            matrixEntries.push_back(stod(matrixEntry));
        }
        matrixRowNumber++;
    }
    return Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(matrixEntries.data(), matrixRowNumber, matrixEntries.size() / matrixRowNumber);
}

int findMaxIndex(MatrixXd mat) {
    double maxValue = mat.maxCoeff();
    int maxIndex = 0;
    for(int i = 0; i < mat.size(); i++) {
        if(mat(i) == maxValue) {
            maxIndex = i;
            break;
        }
    }
    return maxIndex;
}


MatrixXd imgToMatrix(string fileName){
    int width, height, channels;
    unsigned char* data = stbi_load(fileName.c_str(), &width, &height, &channels, STBI_grey);
    MatrixXd imgMatrix(height, width);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            imgMatrix(i, j) = data[i * width + j];
        }
    }
    if(height!=28 || width!=28)
    imgMatrix.conservativeResize(28,28);
    stbi_image_free(data);
    return imgMatrix;
}

class NeuralNetwork{
    public:
        int inodes,hnodes,onodes;
        double lr;
        MatrixXd wih,who;

        NeuralNetwork(int inputNodes,int hiddenNides,int outputNodes,double learningRate){
            this->inodes = inputNodes;
            this->hnodes = hiddenNides;
            this->onodes = outputNodes;
            this->lr = learningRate;
            this->wih = MatrixXd::Random(this->hnodes , this->inodes);
            this->who = MatrixXd::Random(this->onodes , this->hnodes);
        }

        void train(MatrixXd inputList , MatrixXd targetList){
            MatrixXd inputs = inputList.transpose();
            MatrixXd targets = targetList.transpose();

            MatrixXd hiddenInputs = this->wih * inputs;
            MatrixXd hiddenOutputs = hiddenInputs.unaryExpr(&activationFunction);
            MatrixXd finalInputs = this->who * hiddenOutputs;
            MatrixXd finalOutputs = finalInputs.unaryExpr(&activationFunction);

            MatrixXd errorOutput = targets - finalOutputs;
            MatrixXd errorHidden = this->who.transpose() * errorOutput;

            this->wih += this->lr * ( (errorHidden.array() * hiddenOutputs.array() * (1 - hiddenOutputs.array())).matrix() * (inputs.transpose()) );
            this->who += this->lr * ( (errorOutput.array() * finalOutputs.array() * (1 - finalOutputs.array())).matrix() * (hiddenOutputs.transpose()) );
        }

        MatrixXd query(MatrixXd inputList){
            MatrixXd inputs = inputList.transpose();

            MatrixXd hiddenInputs = this->wih * inputs;
            MatrixXd hiddenOutputs = hiddenInputs.unaryExpr(&activationFunction);
            MatrixXd finalInputs = this->who * hiddenOutputs;
            MatrixXd finalOutputs = finalInputs.unaryExpr(&activationFunction);
            return finalOutputs;
        }
};

int main(){
    NeuralNetwork n(784,350,10,0.1);


    //////////////////////////////////////////Training//////////////////////////////////////////////////
    // n.wih = loadData("weights_input_hidden.csv");
    // n.who = loadData("weights_hidden_output.csv");    
    // int examplesTrained = 0;
    // int times = 3;
    // for(int i = 0; i < times; i++){
    //     ifstream dataFile("mnist_test.csv");
    //     string line;
    //     while (getline(dataFile , line)){
    //         int label = line[0] - '0';
    //         vector<double> inputsVector;
    //         stringstream ss(line);
    //         string item;
    //         while(getline(ss, item, ',')) {
    //             inputsVector.push_back( ((stod(item) / 255) * 0.99) + 0.01);
    //         }
    //         inputsVector.erase(inputsVector.begin());
    //         Map< Matrix<double, 1, Dynamic> > inputs(inputsVector.data(),1,inputsVector.size());
    //         MatrixXd targets = MatrixXd::Constant(1,10,0.01);
    //         targets(0 , label) = 0.99;
    //         n.train(inputs , targets);
    //         examplesTrained++;
    //         cout << examplesTrained << " Examples fed\n";
    //     } 
    //     saveData("weights_input_hidden.csv" , n.wih);
    //     saveData("weights_hidden_output.csv" , n.who);
    //     cout << "Saved!"<<endl<<endl;
    //     dataFile.close();
    // }
    




    /////////////////////////////////////////////Testing/////////////////////////////////////////////////
    // n.wih = loadData("weights_input_hidden.csv");
    // n.who = loadData("weights_hidden_output.csv");
    // ifstream trainingFile("mnist_test_10.csv");
    // string line;
    // float noOfLines = 0;
    // float correctGuess = 0;
    // while (getline(trainingFile , line)){
    //     noOfLines++;
    //     int label = line[0] - '0';
    //     vector<double> inputVector;
    //     stringstream ss(line);
    //     string value;
    //     while (getline(ss,value , ',')){
    //         inputVector.push_back( ((stod(value) / 255)*0.99) + 0.01 );
    //     }
    //     inputVector.erase(inputVector.begin());
    //     Map<MatrixXd> inputs(inputVector.data(),1,inputVector.size());
    //     MatrixXd output = n.query(inputs);
    //     int guess = findMaxIndex(output);
    //     cout << "Correct:\t" << label << "\nGuessed:\t" <<guess<<endl<<endl;
    //     if(guess == label){
    //         correctGuess++;
    //     }
    // }
    // cout << "Accuracy: " << (correctGuess/noOfLines) * 100 <<"%";
    // cout << "\n" << noOfLines;
    // trainingFile.close();
    


    //////////////////////////////////////////Querying////////////////////////////////////////////////
    n.wih = loadData("weights_input_hidden.csv");
    n.who = loadData("weights_hidden_output.csv");    
    MatrixXd input = imgToMatrix("img_query.png");
    input = input.unaryExpr(&scaleData);
    input = reshape(input , 28,28);
    MatrixXd output = n.query(input);
    int guess = findMaxIndex(output);
    cout << output <<endl;
    cout << "The guessed number is:\t" << guess;
    cout << "\nAccuracy: \t" << output.maxCoeff() * 100 << "%";


    return 0;
}