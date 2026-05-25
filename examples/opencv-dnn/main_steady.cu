#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

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
    for (double x : xs) acc += (x - m) * (x - m);
    return sqrt(acc / (xs.size() - 1));
}

int main() {
    try {
        string modelPath = getenv_or("DNN_MODEL", "mobilenetv2-10.onnx");
        string imagePath = getenv_or("DNN_IMAGE", "imagenet_test_1000/0_dummy.png");
        string csvPath = getenv_or("STEADY_CSV", "steady_results.csv");

        int warmups = getenv_int("WARMUPS", 5);
        int runs = getenv_int("RUNS", 50);

        cerr << "STEADY_MODEL=" << modelPath << endl;
        cerr << "STEADY_IMAGE=" << imagePath << endl;
        cerr << "STEADY_WARMUPS=" << warmups << endl;
        cerr << "STEADY_RUNS=" << runs << endl;

        Net net = readNetFromONNX(modelPath);
        if (net.empty()) {
            cerr << "ERROR: Cannot load model: " << modelPath << endl;
            return 2;
        }

        net.setPreferableBackend(DNN_BACKEND_CUDA);
        net.setPreferableTarget(DNN_TARGET_CUDA);

        Mat image = imread(imagePath);
        if (image.empty()) {
            cerr << "ERROR: Cannot load image: " << imagePath << endl;
            return 3;
        }

        Mat blob = blobFromImage(
            image,
            1.0 / 255.0,
            Size(224, 224),
            Scalar(),
            true,
            false
        );

        ofstream csv(csvPath);
        csv << "phase,run,status,inference_ms,predicted_class,confidence_pct\n";

        auto run_one = [&](const string& phase, int run) -> double {
            auto t0 = chrono::steady_clock::now();

            net.setInput(blob);
            Mat output = net.forward();

            cudaError_t syncErr = cudaDeviceSynchronize();
            if (syncErr != cudaSuccess) {
                cerr << "ERROR: cudaDeviceSynchronize failed: "
                     << cudaGetErrorString(syncErr) << endl;
                return -1.0;
            }

            auto t1 = chrono::steady_clock::now();
            double ms = chrono::duration<double, milli>(t1 - t0).count();

            Point classIdPoint;
            double confidence = 0.0;
            minMaxLoc(output.reshape(1, 1), nullptr, &confidence, nullptr, &classIdPoint);

            csv << phase << ","
                << run << ","
                << "OK" << ","
                << fixed << setprecision(3) << ms << ","
                << classIdPoint.x << ","
                << fixed << setprecision(3) << confidence * 100.0
                << "\n";

            cout << "STEADY_RESULT"
                 << " phase=" << phase
                 << " run=" << run
                 << " status=OK"
                 << " inference_ms=" << fixed << setprecision(3) << ms
                 << " predicted_class=" << classIdPoint.x
                 << " confidence_pct=" << fixed << setprecision(3) << confidence * 100.0
                 << endl;

            return ms;
        };

        for (int i = 1; i <= warmups; ++i) {
            double ms = run_one("warmup", i);
            if (ms < 0.0) return 4;
        }

        vector<double> measured;
        measured.reserve(runs);

        for (int i = 1; i <= runs; ++i) {
            double ms = run_one("measure", i);
            if (ms < 0.0) return 5;
            measured.push_back(ms);
        }

        cerr << "STEADY_SUMMARY"
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
