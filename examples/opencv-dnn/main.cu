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
    int warmups = 0;
    int runs = 1;

    if (const char* env = std::getenv("BENCH_WARMUPS")) {
        warmups = std::atoi(env);
        if (warmups < 0) warmups = 0;
    }

    if (const char* env = std::getenv("BENCH_RUNS")) {
        runs = std::atoi(env);
        if (runs < 1) runs = 1;
    }

    string modelPath = "mobilenetv2-10.onnx";
    if (const char* env = std::getenv("BENCH_MODEL")) {
        modelPath = env;
    }

    string classFile = "imagenet_classes.txt";
    if (const char* env = std::getenv("BENCH_CLASSES")) {
        classFile = env;
    }

    string testImageFolder = "imagenet_test_1000";
    if (const char* env = std::getenv("BENCH_IMAGE_FOLDER")) {
        testImageFolder = env;
    }

    cout << "Running inference with model: " << modelPath << endl;
    cout << "Benchmark warmups: " << warmups << endl;
    cout << "Benchmark runs: " << runs << endl;

    Net net = readNetFromONNX(modelPath);
    if (net.empty()) {
        cerr << "Cannot load model: " << modelPath << endl;
        return -1;
    }

    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    vector<string> imagePaths = getImagePaths(testImageFolder);
    if (imagePaths.empty()) {
        cerr << "No test images found in: " << testImageFolder << endl;
        return -1;
    }

    string imagePath = imagePaths.front();

    size_t startName = imagePath.find_last_of("/") + 1;
    size_t endName = imagePath.find_first_of("_");
    int trueClassId = -1;
    if (endName != string::npos && endName > startName) {
        try {
            trueClassId = stoi(imagePath.substr(startName, endName - startName));
        } catch (...) {
            trueClassId = -1;
        }
    }

    Mat image = imread(imagePath);
    if (image.empty()) {
        cerr << "Cannot load image: " << imagePath << endl;
        return -1;
    }

    Mat blob;
    blobFromImage(image, blob, 1.0/255.0, Size(224, 224), Scalar(0.485, 0.456, 0.406), true, false);

    Scalar mean(0.485, 0.456, 0.406);
    Scalar std(0.229, 0.224, 0.225);
    divide(blob - mean, std, blob);

    vector<long> totalTimings;
    int correctPredictions = 0;

    auto run_once = [&](const string& type, int run_id, bool measured) {
        auto time_start = std::chrono::high_resolution_clock::now();

        net.setInput(blob);
        Mat output = net.forward();

        auto time_end = std::chrono::high_resolution_clock::now();
        long total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();

        Point classIdPoint;
        double confidence;
        minMaxLoc(output.reshape(1, 1), nullptr, &confidence, nullptr, &classIdPoint);
        int predictedClassId = classIdPoint.x;

        if (measured) {
            totalTimings.push_back(total_ms);
            if (trueClassId >= 0 && predictedClassId == trueClassId) {
                correctPredictions++;
            }
        }

        cout << "BENCH_RESULT,type=" << type
             << ",run=" << run_id
             << ",inference_ms=" << total_ms
             << ",predicted=" << predictedClassId
             << ",confidence=" << fixed << setprecision(4) << confidence
             << endl;
    };

    for (int i = 1; i <= warmups; ++i) {
        run_once("warmup", i, false);
    }

    for (int i = 1; i <= runs; ++i) {
        run_once("measure", i, true);
    }

    string basename = modelPath.substr(0, modelPath.find_last_of("."));
    saveTotalTimings("total_times_" + basename + ".txt", totalTimings);

    float accuracy = 0.0f;
    if (trueClassId >= 0 && runs > 0) {
        accuracy = static_cast<float>(correctPredictions) / runs * 100.0f;
    }

    cout << "\nFinal Results:" << endl;
    cout << "Total images: " << runs << endl;
    cout << "Correct predictions: " << correctPredictions << endl;
    cout << "Accuracy: " << fixed << setprecision(2) << accuracy << "%" << endl;

    cout.flush();
    cerr.flush();

    if (std::getenv("GVIRTUS_FAST_EXIT_AFTER_RESULT") != nullptr) {
        std::_Exit(0);
    }

    return 0;
}
