#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <vector>
#include <chrono>

#define OPENPOSE_FLAGS_DISABLE_POSE
#include <openpose/flags.hpp>
#include <openpose/headers.hpp>

// Defaults point to the bind-mounted folder inside the container
DEFINE_string(image_dir, "/opt/openpose/examples/media/coco_test_images",
              "Input image directory.");
DEFINE_string(output_dir, "/opt/openpose/examples/media",
              "Where the rendered image will be saved.");
DEFINE_bool(no_display, true, "Disable visual display.");

// Force higher net input resolution to stress per-call H2D bytes. Default is
// OpenPose's normal "-1x368" which produces ~1 MB H2D regardless of input size.
// Bigger values: -1x720 (~3MB), -1x1024 (~5MB), -1x1080 (~6MB), 1920x1080 (~6MB),
// 3840x2160 (~25 MB, "4K equivalent"). The aspect-preserving "-1xH" form keeps
// the input image's width and only constrains height.
DEFINE_string(custom_net_resolution, "-1x368",
              "Net input size, e.g. -1x368 (default), -1x720, -1x1080, 1920x1080.");

static std::string deriveOutputPath(const std::string& inputPath,
                                    const std::string& outDir)
{
    namespace fs = std::filesystem;
    fs::path dir(outDir);
    if (!fs::exists(dir)) fs::create_directories(dir);

    fs::path in(inputPath);
    std::string stem = in.has_stem() ? in.stem().string() : "output";
    return (dir / (stem + "_pose.png")).string();
}

void saveOutput(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr,
                const std::string& inputPath)
{
    try {
        if (datumsPtr && !datumsPtr->empty()) {
            const cv::Mat cvMat = OP_OP2CVCONSTMAT(datumsPtr->at(0)->cvOutputData);
            if (!cvMat.empty()) {
                const auto outPath = deriveOutputPath(inputPath, FLAGS_output_dir);
                if (!cv::imwrite(outPath, cvMat))
                    op::opLog("Failed to write image to: " + outPath, op::Priority::High);
                else
                    op::opLog("Saved rendered pose image to: " + outPath, op::Priority::High);
            } else {
                op::opLog("Empty cv::Mat as output.", op::Priority::High);
            }
        } else {
            op::opLog("Nullptr or empty datumsPtr found.", op::Priority::High);
        }
    } catch (const std::exception& e) {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

void printKeypoints(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr)
{
    try {
        if (datumsPtr && !datumsPtr->empty())
            op::opLog("Body keypoints: " + datumsPtr->at(0)->poseKeypoints.toString(), op::Priority::High);
        else
            op::opLog("Nullptr or empty datumsPtr found.", op::Priority::High);
    } catch (const std::exception& e) {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

std::vector<std::string> getImagePaths(const std::string& imageDir, int maxImages = 100)
{
    std::vector<std::string> imagePaths;
    namespace fs = std::filesystem;
    
    try {
        if (!fs::exists(imageDir) || !fs::is_directory(imageDir)) {
            op::opLog("Image directory does not exist: " + imageDir, op::Priority::High);
            return imagePaths;
        }

        int count = 0;
        for (const auto& entry : fs::directory_iterator(imageDir)) {
            if (count >= maxImages) break;
            
            if (entry.is_regular_file()) {
                std::string extension = entry.path().extension().string();
                std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
                
                if (extension == ".jpg" || extension == ".jpeg" || 
                    extension == ".png" || extension == ".bmp") {
                    imagePaths.push_back(entry.path().string());
                    count++;
                }
            }
        }
        
        op::opLog("Found " + std::to_string(imagePaths.size()) + " images to process.", op::Priority::High);
    } catch (const std::exception& e) {
        op::error("Error reading image directory: " + std::string(e.what()), __LINE__, __FUNCTION__, __FILE__);
    }
    
    return imagePaths;
}

int tutorialApiCpp()
{
    try {
        op::opLog("Starting OpenPose demo...", op::Priority::High);
        const auto totalTimer = op::getTimerInit();

        // Initialize OpenPose wrapper
        op::Wrapper opWrapper{op::ThreadManagerMode::Asynchronous};
        // Apply --custom_net_resolution to stress per-call payload sizes.
        op::WrapperStructPose poseParams;
        poseParams.netInputSize = op::flagsToPoint(
            op::String(FLAGS_custom_net_resolution), "-1x368");
        opWrapper.configure(poseParams);
        op::opLog("Net input resolution: " + FLAGS_custom_net_resolution,
                  op::Priority::High);
        if (FLAGS_disable_multi_thread) opWrapper.disableMultiThreading();
        opWrapper.start();

        // Get image paths
        std::vector<std::string> imagePaths = getImagePaths(FLAGS_image_dir, 100);
        if (imagePaths.empty()) {
            op::opLog("No images found in directory: " + FLAGS_image_dir, op::Priority::High);
            return -1;
        }

        // Process each image and record time
        std::vector<double> processingTimes;
        int processedCount = 0;

        for (const auto& imagePath : imagePaths) {
            op::opLog("Processing: " + imagePath, op::Priority::High);
            
            // Start timer for individual image
            const auto imageTimer = op::getTimerInit();

            const cv::Mat cvImageToProcess = cv::imread(imagePath);
            if (cvImageToProcess.empty()) {
                op::opLog("Could not read input image: " + imagePath, op::Priority::High);
                continue;
            }

            const op::Matrix imageToProcess = OP_CV2OPCONSTMAT(cvImageToProcess);
            auto datumProcessed = opWrapper.emplaceAndPop(imageToProcess);
            
            if (datumProcessed) {
                printKeypoints(datumProcessed);
                saveOutput(datumProcessed, imagePath);
                processedCount++;
            } else {
                op::opLog("Image could not be processed: " + imagePath, op::Priority::High);
            }

            // Record processing time for this image
            double imageTime = op::getTimeSeconds(imageTimer);
            processingTimes.push_back(imageTime);
            op::opLog("Image processed in " + std::to_string(imageTime) + " seconds", op::Priority::High);
        }

        // Print summary statistics
        op::printTime(totalTimer, "Total processing time for " + std::to_string(processedCount) + 
                      " images: ", " seconds.", op::Priority::High);
        
        if (!processingTimes.empty()) {
            double totalTime = 0.0;
            double minTime = processingTimes[0];
            double maxTime = processingTimes[0];
            
            for (double time : processingTimes) {
                totalTime += time;
                if (time < minTime) minTime = time;
                if (time > maxTime) maxTime = time;
            }
            
            double avgTime = totalTime / processingTimes.size();
            
            op::opLog("Processing time statistics:", op::Priority::High);
            op::opLog("  Average: " + std::to_string(avgTime) + " seconds", op::Priority::High);
            op::opLog("  Minimum: " + std::to_string(minTime) + " seconds", op::Priority::High);
            op::opLog("  Maximum: " + std::to_string(maxTime) + " seconds", op::Priority::High);
            op::opLog("  Total: " + std::to_string(totalTime) + " seconds", op::Priority::High);

                std::ofstream outFile("/opt/openpose/examples/media/processing_times.txt");
    if (outFile.is_open()) {
        outFile << "ImageIndex,TimeSeconds\n";
        for (size_t i = 0; i < processingTimes.size(); ++i) {
            outFile << i+1 << "," << processingTimes[i] << "\n";
        }
        outFile.close();
        op::opLog("Saved per-image processing times to processing_times.txt", op::Priority::High);
    } else {
        op::opLog("Failed to open processing_times.txt for writing.", op::Priority::High);
    }

        }

        return 0;
    } catch (const std::exception& e) {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        return -1;
    }
}

int main(int argc, char* argv[])
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    return tutorialApiCpp();
}
