#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <dirent.h>
#include <cstdlib>
#include <chrono>

using namespace cv;
using namespace dnn;
using namespace std;

vector<string> readClassNames(const string& filename) {
    vector<string> classes;
    ifstream fp(filename);
    if (!fp.is_open()) {
        cerr << "File with classes labels not found: " << filename << endl;
        exit(-1);
    }
    string name;
    while (getline(fp, name)) {
        classes.push_back(name);
    }
    fp.close();
    return classes;
}

vector<string> getImagePaths(const string& folderPath) {
    vector<string> imagePaths;
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(folderPath.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            string filename = ent->d_name;
            if (filename.find(".JPEG") != string::npos || 
                filename.find(".jpg") != string::npos ||
                filename.find(".png") != string::npos) {
                imagePaths.push_back(folderPath + "/" + filename);
            }
        }
        closedir(dir);
    } else {
        cerr << "Could not open directory: " << folderPath << endl;
        exit(-1);
    }
    return imagePaths;
}

void saveTotalTimings(const std::string& filename, const std::vector<long>& totalTimings) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    
    for (const auto& time_ms : totalTimings) {
        outFile << time_ms << "\n"; 
    }
    
    outFile.close();
    std::cout << "Saved total timings to: " << filename << std::endl;
}

int main() {
    // find more pre-trained models at: https://github.com/onnx/models/
    std::vector<std::string> modelPaths = {
        "mobilenetv2-10.onnx"
    };
    string classFile = "imagenet_classes.txt";
    string testImageFolder = "imagenet_test_1000";  

    for (const auto& modelPath : modelPaths) {
        
        std::vector<long> totalTimings; 

        std::cout << "Running inference with model: " << modelPath << std::endl;

        Net net = readNetFromONNX(modelPath);
        if (net.empty()) {
            cerr << "Cannot load model: " << modelPath << endl;
            return -1;
        }
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        //net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        //net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        vector<string> classes = readClassNames(classFile);

        vector<string> imagePaths = getImagePaths(testImageFolder);
        if (imagePaths.empty()) {
            cerr << "No test images found in: " << testImageFolder << endl;
            return -1;
        }
        
        int totalImages = 0;
        int correctPredictions = 0;
        auto time_start = std::chrono::high_resolution_clock::now();
        auto time_end = std::chrono::high_resolution_clock::now();
        
        int TOTAL = 1000;
        for (const string& imagePath : imagePaths) {

            size_t start = imagePath.find_last_of("/") + 1;
            size_t end = imagePath.find_first_of("_");
            if (end == string::npos) {
                cerr << "Invalid image filename format: " << imagePath << endl;
                continue;
            }
            string trueClassId_str = imagePath.substr(start, end - start);
            int trueClassId = stoi(trueClassId_str);
    
            Mat image = imread(imagePath);
            if (image.empty()) {
                cerr << "Cannot load image: " << imagePath << endl;
                continue;
            }
            
            Mat blob;
            blobFromImage(image, blob, 1.0/255.0, Size(224, 224), Scalar(0.485, 0.456, 0.406), true, false);
            
            Scalar mean(0.485, 0.456, 0.406);
            Scalar std(0.229, 0.224, 0.225);
            divide(blob - mean, std, blob);

            time_start = std::chrono::high_resolution_clock::now();
            net.setInput(blob);
            Mat output = net.forward();
            time_end = std::chrono::high_resolution_clock::now();
            long total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
            totalTimings.push_back(total_ms);
            
            Point classIdPoint;
            double confidence;
            minMaxLoc(output.reshape(1, 1), nullptr, &confidence, nullptr, &classIdPoint);
            int predictedClassId = classIdPoint.x;
            
            if (predictedClassId == trueClassId) {
                correctPredictions++;
            }
            totalImages++;

            if (totalImages % 100 == 0) {
                cout << "Processed " << totalImages << " images..." << endl;
            }
    
            cout << "Image: " << imagePath << endl;
            cout << "True Class ID: " << trueClassId << ", Predicted Class ID: " << predictedClassId 
                << ", Confidence: " << fixed << setprecision(2) << confidence * 100 << "%" 
                << ", Time taken: " << total_ms << " ms" << endl;

            if (totalImages == TOTAL) break;

        }

        string basename = modelPath.substr(0, modelPath.find_last_of("."));
        saveTotalTimings("total_times_" + basename + ".txt", totalTimings);

        float accuracy = static_cast<float>(correctPredictions) / totalImages * 100;
        cout << "\nFinal Results:" << endl;
        cout << "Total images: " << totalImages << endl;
        cout << "Correct predictions: " << correctPredictions << endl;
        cout << "Accuracy: " << fixed << setprecision(2) << accuracy << "%" << endl;

    }
    
    std::cout.flush();
    std::cerr.flush();
    std::_Exit(0);
}
