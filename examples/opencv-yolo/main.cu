#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using namespace cv;
using namespace cv::dnn;
using namespace std;

static string getenv_or(const char* name, const string& fallback) {
    const char* v = getenv(name);
    return v ? string(v) : fallback;
}

static int getenv_int(const char* name, int fallback) {
    const char* v = getenv(name);
    return v ? atoi(v) : fallback;
}

static double mean(const vector<double>& xs) {
    if (xs.empty()) return 0.0;
    return accumulate(xs.begin(), xs.end(), 0.0) / xs.size();
}

static double stddev(const vector<double>& xs) {
    if (xs.size() < 2) return 0.0;

    double m = mean(xs);
    double acc = 0.0;

    for (double x : xs) {
        acc += (x - m) * (x - m);
    }

    return sqrt(acc / (xs.size() - 1));
}

static string first_file_with_ext(const vector<string>& exts) {
    for (const auto& entry : fs::recursive_directory_iterator(".")) {
        if (!entry.is_regular_file()) continue;

        string p = entry.path().string();
        string ext = entry.path().extension().string();

        for (const auto& wanted : exts) {
            if (ext == wanted) return p;
        }
    }

    return "";
}

static string first_image_not_output() {
    string first = first_file_with_ext({".jpg", ".jpeg", ".png"});

    if (!first.empty() && first.find("output") == string::npos) {
        return first;
    }

    for (const auto& entry : fs::recursive_directory_iterator(".")) {
        if (!entry.is_regular_file()) continue;

        string p = entry.path().string();
        string ext = entry.path().extension().string();

        if ((ext == ".jpg" || ext == ".jpeg" || ext == ".png") &&
            p.find("output") == string::npos) {
            return p;
        }
    }

    return "";
}

static size_t total_elements(const vector<Mat>& outs) {
    size_t total = 0;

    for (const auto& out : outs) {
        total += out.total();
    }

    return total;
}

int main() {
    try {
        string modelPath = getenv_or("YOLO_MODEL", "");
        string configPath = getenv_or("YOLO_CONFIG", "");
        string weightsPath = getenv_or("YOLO_WEIGHTS", "");
        string imagePath = getenv_or("YOLO_IMAGE", "");
        string csvPath = getenv_or("STEADY_CSV", "steady_yolo_results.csv");

        int warmups = getenv_int("WARMUPS", 5);
        int runs = getenv_int("RUNS", 50);
        int inputSize = getenv_int("YOLO_INPUT_SIZE", 640);

        if (modelPath.empty()) {
            modelPath = first_file_with_ext({".onnx"});
        }

        if (imagePath.empty()) {
            imagePath = first_image_not_output();
        }

        Net net;

        if (!modelPath.empty()) {
            cerr << "STEADY_YOLO_MODEL=" << modelPath << endl;
            net = readNet(modelPath);
        } else {
            if (configPath.empty()) {
                configPath = first_file_with_ext({".cfg"});
            }

            if (weightsPath.empty()) {
                weightsPath = first_file_with_ext({".weights"});
            }

            cerr << "STEADY_YOLO_CONFIG=" << configPath << endl;
            cerr << "STEADY_YOLO_WEIGHTS=" << weightsPath << endl;

            if (configPath.empty() || weightsPath.empty()) {
                cerr << "ERROR: Could not find .onnx model or .cfg/.weights pair." << endl;
                return 2;
            }

            net = readNetFromDarknet(configPath, weightsPath);
        }

        if (net.empty()) {
            cerr << "ERROR: Could not load YOLO network." << endl;
            return 3;
        }

        cerr << "STEADY_YOLO_IMAGE=" << imagePath << endl;
        cerr << "STEADY_WARMUPS=" << warmups << endl;
        cerr << "STEADY_RUNS=" << runs << endl;
        cerr << "STEADY_INPUT_SIZE=" << inputSize << endl;
        cerr << "STEADY_CSV=" << csvPath << endl;

        Mat image = imread(imagePath);

        if (image.empty()) {
            cerr << "ERROR: Could not load image: " << imagePath << endl;
            return 4;
        }

        net.setPreferableBackend(DNN_BACKEND_CUDA);
        net.setPreferableTarget(DNN_TARGET_CUDA);

        Mat blob = blobFromImage(
            image,
            1.0 / 255.0,
            Size(inputSize, inputSize),
            Scalar(),
            true,
            false
        );

        vector<string> outputNames = net.getUnconnectedOutLayersNames();

        ofstream csv(csvPath);

        if (!csv.is_open()) {
            cerr << "ERROR: Could not open CSV for writing: " << csvPath << endl;
            return 7;
        }

        csv << "phase,run,status,detection_ms,output_blobs,output_elements\n";

        auto run_one = [&](const string& phase, int run) -> double {
            vector<Mat> outs;

            auto t0 = chrono::steady_clock::now();

            net.setInput(blob);
            net.forward(outs, outputNames);

            cudaError_t syncErr = cudaDeviceSynchronize();

            if (syncErr != cudaSuccess) {
                cerr << "ERROR: cudaDeviceSynchronize failed: "
                     << cudaGetErrorString(syncErr) << endl;
                return -1.0;
            }

            auto t1 = chrono::steady_clock::now();

            double ms = chrono::duration<double, milli>(t1 - t0).count();
            size_t elems = total_elements(outs);

            csv << phase << ","
                << run << ","
                << "OK" << ","
                << fixed << setprecision(3) << ms << ","
                << outs.size() << ","
                << elems << "\n";

            csv.flush();

            cout << "STEADY_YOLO_RESULT"
                 << " phase=" << phase
                 << " run=" << run
                 << " status=OK"
                 << " detection_ms=" << fixed << setprecision(3) << ms
                 << " output_blobs=" << outs.size()
                 << " output_elements=" << elems
                 << endl;

            return ms;
        };

        for (int i = 1; i <= warmups; ++i) {
            double ms = run_one("warmup", i);

            if (ms < 0.0) {
                return 5;
            }
        }

        vector<double> measured;
        measured.reserve(runs);

        for (int i = 1; i <= runs; ++i) {
            double ms = run_one("measure", i);

            if (ms < 0.0) {
                return 6;
            }

            measured.push_back(ms);
        }

        cerr << "STEADY_YOLO_SUMMARY"
             << " runs=" << measured.size()
             << " mean_ms=" << fixed << setprecision(3) << mean(measured)
             << " std_ms=" << fixed << setprecision(3) << stddev(measured)
             << endl;

        return 0;
    } catch (const exception& e) {
        cerr << "ERROR: Exception: " << e.what() << endl;
        return 1;
    }
}