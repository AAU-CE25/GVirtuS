#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace cv;
using namespace cv::dnn;
using namespace std;

static int getInternalRuns() {
    const char* env = std::getenv("BENCH_INTERNAL_RUNS");
    if (!env) return 1;

    int runs = std::atoi(env);
    return runs > 0 ? runs : 1;
}

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

    DIR* dir;
    struct dirent* ent;

    if ((dir = opendir(folderPath.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            string filename = ent->d_name;

            if (
                filename.find(".JPEG") != string::npos ||
                filename.find(".jpg") != string::npos ||
                filename.find(".png") != string::npos
            ) {
                imagePaths.push_back(folderPath + "/" + filename);
            }
        }

        closedir(dir);
    } else {
        cerr << "Could not open directory: " << folderPath << endl;
        exit(-1);
    }

    sort(imagePaths.begin(), imagePaths.end());
    return imagePaths;
}

void saveTotalTimings(const string& filename, const vector<long>& totalTimings) {
    ofstream outFile(filename);

    if (!outFile.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return;
    }

    for (const auto& time_ms : totalTimings) {
        outFile << time_ms << "\n";
    }

    outFile.close();
    cout << "Saved total timings to: " << filename << endl;
}

int main() {
    vector<string> modelPaths = {
        "mobilenetv2-10.onnx"
    };

    string classFile = "imagenet_classes.txt";
    string testImageFolder = "imagenet_test_1000";

    const int internalRuns = getInternalRuns();
    const int TOTAL = 50;

    cout << "[BENCH] BENCH_INTERNAL_RUNS=" << internalRuns << endl;

    vector<string> classes = readClassNames(classFile);

    vector<string> imagePaths = getImagePaths(testImageFolder);
    if (imagePaths.empty()) {
        cerr << "No test images found in: " << testImageFolder << endl;
        return -1;
    }

    for (const auto& modelPath : modelPaths) {
        cout << "Running inference with model: " << modelPath << endl;

        Net net = readNetFromONNX(modelPath);
        if (net.empty()) {
            cerr << "Cannot load model: " << modelPath << endl;
            return -1;
        }

        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

        vector<long> totalTimingsAllRuns;

        long long totalImagesAllRuns = 0;
        long long correctPredictionsAllRuns = 0;

        for (int run = 1; run <= internalRuns; ++run) {
            cout << "[BENCH] internal run " << run << "/" << internalRuns
                 << " model=" << modelPath << endl;

            vector<long> totalTimingsThisRun;

            int totalImages = 0;
            int correctPredictions = 0;

            for (const string& imagePath : imagePaths) {
                size_t start = imagePath.find_last_of("/") + 1;
                size_t end = imagePath.find_first_of("_", start);

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
                blobFromImage(
                    image,
                    blob,
                    1.0 / 255.0,
                    Size(224, 224),
                    Scalar(0.485, 0.456, 0.406),
                    true,
                    false
                );

                Scalar mean(0.485, 0.456, 0.406);
                Scalar std(0.229, 0.224, 0.225);
                divide(blob - mean, std, blob);

                auto time_start = chrono::high_resolution_clock::now();

                net.setInput(blob);
                Mat output = net.forward();

                auto time_end = chrono::high_resolution_clock::now();

                long total_ms = chrono::duration_cast<chrono::milliseconds>(
                    time_end - time_start
                ).count();

                totalTimingsThisRun.push_back(total_ms);
                totalTimingsAllRuns.push_back(total_ms);

                Point classIdPoint;
                double confidence;

                minMaxLoc(
                    output.reshape(1, 1),
                    nullptr,
                    &confidence,
                    nullptr,
                    &classIdPoint
                );

                int predictedClassId = classIdPoint.x;

                if (predictedClassId == trueClassId) {
                    correctPredictions++;
                }

                totalImages++;

                if (totalImages % 100 == 0) {
                    cout << "[BENCH] run " << run << "/" << internalRuns
                         << " processed " << totalImages << " images..." << endl;
                }

                cout << "Image: " << imagePath << endl;
                cout << "True Class ID: " << trueClassId
                     << ", Predicted Class ID: " << predictedClassId
                     << ", Confidence: " << fixed << setprecision(2)
                     << confidence * 100 << "%"
                     << ", Time taken: " << total_ms << " ms" << endl;

                if (totalImages == TOTAL) break;
            }

            totalImagesAllRuns += totalImages;
            correctPredictionsAllRuns += correctPredictions;

            float runAccuracy = totalImages > 0
                ? static_cast<float>(correctPredictions) / totalImages * 100.0f
                : 0.0f;

            cout << "\nRun Results:" << endl;
            cout << "Model: " << modelPath << endl;
            cout << "Run: " << run << "/" << internalRuns << endl;
            cout << "Total images: " << totalImages << endl;
            cout << "Correct predictions: " << correctPredictions << endl;
            cout << "Accuracy: " << fixed << setprecision(2) << runAccuracy << "%" << endl;
        }

        string basename = modelPath.substr(0, modelPath.find_last_of("."));
        saveTotalTimings("total_times_" + basename + ".txt", totalTimingsAllRuns);

        float accuracy = totalImagesAllRuns > 0
            ? static_cast<float>(correctPredictionsAllRuns) / totalImagesAllRuns * 100.0f
            : 0.0f;

        cout << "\nFinal Results:" << endl;
        cout << "Model: " << modelPath << endl;
        cout << "Internal runs: " << internalRuns << endl;
        cout << "Total images: " << totalImagesAllRuns << endl;
        cout << "Correct predictions: " << correctPredictionsAllRuns << endl;
        cout << "Accuracy: " << fixed << setprecision(2) << accuracy << "%" << endl;
    }

    return 0;
}
