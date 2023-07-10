//============== Sample_Feature_tracker code modified by Joon Oh KIM.==============
//This code is to run the Visual Odometry at the Nvidia Drive AGX Xavier
//If you have any questions about Driveworks code, please send the e-mail to me.

//=================================================================================
#include <dw/core/VersionCurrent.h>
#include <framework/DriveWorksSample.hpp> //This is the main operation part. It operates the onProcess() and onRender().
#include <framework/SampleFramework.hpp>
#include <framework/SimpleCamera.hpp>
#include <framework/SimpleStreamer.hpp>
#include <framework/WindowGLFW.hpp>

#include <dwvisualization/core/RenderEngine.h>
#include <dw/sensors/Sensors.h>
#include <dw/sensors/camera/Camera.h>
#include <dw/image/Image.h>
#include <dw/interop/streamer/ImageStreamer.h>
#include <dw/imageprocessing/tracking/FeatureTracker.h>
#include <dw/imageprocessing/features/FeatureDetector.h>

//== Added by new libraries ==
//If you want to add new libraires (ex: eigen), please follow the below steps.
//1) Add you libraries files at the compiler path (/usr/local/cuda-10.2/targets/x86_64-linux/include),(/usr/local/cuda-10.2/targets/aarch64-linux/include)
//2) Click the mouse right button at your project name(Left menu - below the "Project Explorer"
//3) Go to (Build->Settings->Tool Settings->Includes->Include paths->Enter your path which I mentioned at num 1).
#include <atomic>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <vector>
#include <fstream>
#include <string>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <utility>
#include <future>
#include <map>
#include <cstdlib>
#include <ctime>
#include <random>
#include <algorithm>
#define float16_t eigen_broken_float16_t
#undef Success
#include "eigen/Eigen/Dense"
#include "eigen/Eigen/Core"
#include "eigen/Eigen/SVD"
#undef float16_t
#include <fstream>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace dw_samples::common;
using namespace dw::common;


//------------------------------------------------------------------------------
// Camera Scaling Tracker
// The Camera Scaling Tracker sample demonstrates the scaling feature
// tracking capabilities of the dw_features module. It loads a video stream and
// reads the images sequentially. For each frame, it tracks scaling features from the
// previous frame. It doesn't detect new features, when there's no scaling features
// in the frame, the video replay will be paused automatically, you can use
// mouse to drag the boxes to track and press space to start replay/tracking.
//------------------------------------------------------------------------------
// Global variable
uint32_t frameCounter = 0;
class CameraFeatureTracker : public DriveWorksSample
{
private:
    // ------------------------------------------------
    // Sample specific variables
    // ------------------------------------------------
    dwContextHandle_t m_context           = DW_NULL_HANDLE;
    dwVisualizationContextHandle_t m_viz  = DW_NULL_HANDLE;
    dwRenderEngineHandle_t m_renderEngine = DW_NULL_HANDLE;
    dwSALHandle_t sal                     = DW_NULL_HANDLE;

    dwCameraProperties cameraProps     = {};
    dwImageProperties cameraImageProps = {};

    dwImageHandle_t frameCudaRgba = DW_NULL_HANDLE;
    dwImageHandle_t frameGL       = DW_NULL_HANDLE;
    std::unique_ptr<SimpleImageStreamerGL<>> streamerCUDA2GL;
    std::unique_ptr<SimpleCamera> camera;

    bool drawHistory = true;

    // tracker handles
    dwFeature2DTrackerHandle_t tracker;
    dwFeature2DDetectorHandle_t detector;
    uint32_t maxFeatureCount;
    uint32_t historyCapacity;

    // pyramid handles
    dwPyramidImage pyramidCurrent;
    dwPyramidImage pyramidPrevious;

    //These point into the buffers of featureList
    dwFeatureHistoryArray featureHistoryCPU;
    dwFeatureHistoryArray featureHistoryGPU;
    dwFeatureArray featuresDetected;
    float32_t* d_nccScores;

    //==== Declare the Global Variables ====
    typedef struct vertex
    {
        dwVector2f pos;
        dwRenderEngineColorRGBA color;
    } vertex;
    // 이전과 현재 프레임의 특징점 위치를 저장할 std::vector 객체를 선언
    std::vector<std::pair<int, int>> prevLocationsVector;
    std::vector<std::pair<int, int>> currentLocationsVector;

    // Initialize R_t0 and t_t0
	Eigen::Matrix3d R_t0 = Eigen::Matrix3d::Identity();
	Eigen::Vector3d t_t0 = Eigen::Vector3d::Zero();

    #define DW_FEATURE_STATUS_INVALID 0
    dwVector2f* prevLocations = NULL;
    uint32_t maxFeatures = 0;
    bool isFirstCall = true;
    int CalculateEssentialMatrixCallCount = 0;

public:
    CameraFeatureTracker(const ProgramArguments& args)
        : DriveWorksSample(args) {}
    /// -----------------------------
    /// Initialize Renderer, Sensors, Image Streamers and Tracker
    /// -----------------------------
    bool onInitialize() override
    {
        // -----------------------------------------
        // Get values from command line
        // -----------------------------------------
        maxFeatureCount = std::stoi(getArgument("maxFeatureCount"));
        historyCapacity = std::stoi(getArgument("historyCapacity"));

        uint32_t detectMode                 = std::stoi(getArgument("detectMode"));
        uint32_t cellSize                   = std::stoi(getArgument("cellSize"));
        uint32_t pyramidLevel               = std::stoi(getArgument("pyramidLevel"));
        uint32_t detectLevel                = std::stoi(getArgument("detectLevel"));
        uint32_t numEvenDistributionPerCell = std::stoi(getArgument("numEvenDistributionPerCell"));
        uint32_t harrisRadius               = std::stoi(getArgument("harrisRadius"));
        uint32_t nmsRadius                  = std::stoi(getArgument("NMSRadius"));
        uint32_t maskType                   = std::stoi(getArgument("maskType"));
        uint32_t enableMaskAdjustment       = std::stoi(getArgument("enableMaskAdjustment"));

        float32_t scoreThreshold  = std::stof(getArgument("scoreThreshold"));
        float32_t detailThreshold = std::stof(getArgument("detailThreshold"));
        float32_t harrisK         = std::stof(getArgument("harrisK"));

        uint32_t trackMode               = std::stoi(getArgument("trackMode"));
        uint32_t windowSize              = std::stoi(getArgument("windowSize"));
        uint32_t numIterTranslation      = std::stoi(getArgument("numIterTranslation"));
        uint32_t numIterScaling          = std::stoi(getArgument("numIterScaling"));
        uint32_t numTranslationOnlyLevel = std::stoi(getArgument("numTranslationOnlyLevel"));

        float32_t nccKillThreshold       = std::stof(getArgument("nccKillThreshold"));
        float32_t nccUpdateThreshold     = std::stof(getArgument("nccUpdateThreshold"));

        // -----------------------------------------
        // Initialize DriveWorks context and SAL
        // -----------------------------------------
        {
            initializeDriveWorks(m_context);
            CHECK_DW_ERROR(dwSAL_initialize(&sal, m_context));
        }

        // -----------------------------
        // Initialize renderer
        // -----------------------------
        {
            CHECK_DW_ERROR(dwVisualizationInitialize(&m_viz, m_context));

            // Setup render engine
            dwRenderEngineParams params{};
            CHECK_DW_ERROR(dwRenderEngine_initDefaultParams(&params, getWindowWidth(), getWindowHeight()));
            params.defaultTile.lineWidth = 0.2f;
            params.defaultTile.font      = DW_RENDER_ENGINE_FONT_VERDANA_20;
            params.bufferSize            = maxFeatureCount * sizeof(vertex) * historyCapacity * 2; //added *2
            CHECK_DW_ERROR(dwRenderEngine_initialize(&m_renderEngine, &params, m_viz));
        }

        // -----------------------------
        // initialize sensors
        // -----------------------------
        {
        	//==== This part is to run the sample video which is located in the driveworks. ====
        	// Sample video path : '/usr/local/driveworks-2.2/data/samples/sfm/triangulation'
        	// You can select the path, below main() => "ProgramArguments::Option_t("video",(DataPath::get() + "/samples/sfm/triangulation/straight_final_Trim2.mp4").c_str())"
            std::string videoPath = getArgument("video");
            std::string file      = "video=" + videoPath;

            dwSensorParams sensorParams{};
            sensorParams.protocol   = "camera.virtual";
            sensorParams.parameters = file.c_str();

            std::string ext = videoPath.substr(videoPath.find_last_of(".") + 1);
            if (ext == "h264" || ext == "h265" || ext == "mp4")
            {
                camera.reset(new SimpleCamera(sensorParams, sal, m_context));
                dwImageProperties outputProps = camera->getOutputProperties();
                outputProps.type              = DW_IMAGE_CUDA;
                outputProps.format            = DW_IMAGE_FORMAT_RGB_FLOAT16_PLANAR;
                camera->setOutputProperties(outputProps);
            }
            else if (ext == "raw" || ext == "lraw")
            {
                camera.reset(new RawSimpleCamera(DW_IMAGE_FORMAT_RGB_FLOAT16_PLANAR,
                                                 sensorParams, sal, m_context, 0,
                                                 DW_CAMERA_OUTPUT_NATIVE_PROCESSED));
            }
            //===================================================================================

// ====== This part is to run the live GMSL Camera. =======
//
//            dwSensorParams params;
//            std::string parameterString = std::string("output-format=raw+data,camera-type=") +
//                    std::string(getArgument("camera-type"));
//            parameterString             += std::string(",camera-group=") + getArgument("camera-group").c_str();
//            parameterString             += std::string(",format=") + std::string(getArgument("serializer-type"));
//            parameterString             += std::string(",fifo-size=") + std::string(getArgument("camera-fifo-size"));
//            parameterString             += std::string(",slave=") + std::string(getArgument("tegra-slave"));
//            params.parameters           = parameterString.c_str();
//            params.protocol             = "camera.gmsl";
//            // DW_IMAGE_FORMAT = 2002, cudastream = 0, DW_CAMERA_OUTPUT =
//            camera.reset(new RawSimpleCamera(DW_IMAGE_FORMAT_RGB_FLOAT16_PLANAR,
//                                             params, sal, m_context, 0,
//                                             DW_CAMERA_OUTPUT_NATIVE_PROCESSED));
//=========================================================================================

            dwImageProperties displayProps = camera->getOutputProperties();
            displayProps.format            = DW_IMAGE_FORMAT_RGBA_UINT8;
            CHECK_DW_ERROR(dwImage_create(&frameCudaRgba, displayProps, m_context));

            streamerCUDA2GL.reset(new SimpleImageStreamerGL<>(displayProps, 1000, m_context));


            //출력 : Camera image with 1280x800 at 30.000000 FPS
            cameraProps      = camera->getCameraProperties();
            cameraImageProps = camera->getOutputProperties();

            printf("Camera image with %dx%d at %f FPS\n", cameraImageProps.width,
                   cameraImageProps.height, cameraProps.framerate);

            // we would like the application run as fast as the original video
            setProcessRate(cameraProps.framerate);
        }

        // -----------------------------
        // Initialize feature tracker
        // -----------------------------
        {
            CHECK_DW_ERROR(dwFeatureHistoryArray_create(&featureHistoryCPU, maxFeatureCount,
                                                        historyCapacity, DW_MEMORY_TYPE_CPU,
                                                        m_context));

            CHECK_DW_ERROR(dwFeatureHistoryArray_create(&featureHistoryGPU, maxFeatureCount,
                                                        historyCapacity, DW_MEMORY_TYPE_CUDA,
                                                        m_context));

            CHECK_DW_ERROR(dwFeatureArray_create(&featuresDetected, maxFeatureCount,
                                                 DW_MEMORY_TYPE_CUDA, m_context));

            dwFeature2DDetectorConfig detectorConfig = {};
            dwFeature2DDetector_initDefaultParams(&detectorConfig);
            detectorConfig.type                       = static_cast<dwFeature2DDetectorType>(detectMode);
            detectorConfig.imageWidth                 = cameraImageProps.width;
            detectorConfig.imageHeight                = cameraImageProps.height;
            detectorConfig.maxFeatureCount            = maxFeatureCount;
            detectorConfig.detectionLevel             = detectLevel;
            detectorConfig.cellSize                   = cellSize;
            detectorConfig.scoreThreshold             = scoreThreshold;
            detectorConfig.detailThreshold            = detailThreshold;
            detectorConfig.numEvenDistributionPerCell = numEvenDistributionPerCell;
            detectorConfig.harrisRadius               = harrisRadius;
            detectorConfig.harrisK                    = harrisK;
            detectorConfig.NMSRadius                  = nmsRadius;
            detectorConfig.isMaskAdjustmentEnabled    = enableMaskAdjustment;
            detectorConfig.maskType                   = static_cast<dwFeature2DSelectionMaskType>(maskType);

            CHECK_DW_ERROR(dwFeature2DDetector_initialize(&detector, &detectorConfig, 0, m_context));

            dwFeature2DTrackerConfig trackerConfig = {};
            dwFeature2DTracker_initDefaultParams(&trackerConfig);
            trackerConfig.algorithm               = static_cast<dwFeature2DTrackerAlgorithm>(trackMode);
            trackerConfig.detectorType            = detectorConfig.type;
            trackerConfig.maxFeatureCount         = maxFeatureCount;
            trackerConfig.historyCapacity         = historyCapacity;
            trackerConfig.imageWidth              = cameraImageProps.width;
            trackerConfig.imageHeight             = cameraImageProps.height;
            trackerConfig.pyramidLevelCount       = pyramidLevel;
            trackerConfig.windowSizeLK            = windowSize;
            trackerConfig.numIterTranslationOnly  = numIterTranslation;
            trackerConfig.numIterScaling          = numIterScaling;
            trackerConfig.numLevelTranslationOnly = numTranslationOnlyLevel;
            trackerConfig.nccUpdateThreshold      = nccUpdateThreshold;
            trackerConfig.nccKillThreshold      = nccKillThreshold;

            CHECK_DW_ERROR(dwFeature2DTracker_initialize(&tracker, &trackerConfig, 0, m_context));

            CHECK_DW_ERROR(dwPyramid_create(
                &pyramidPrevious, pyramidLevel, cameraImageProps.width,
                cameraImageProps.height, DW_TYPE_FLOAT32, m_context));
            CHECK_DW_ERROR(dwPyramid_create(
                &pyramidCurrent, pyramidLevel, cameraImageProps.width,
                cameraImageProps.height, DW_TYPE_FLOAT32, m_context));

            CHECK_CUDA_ERROR(cudaMalloc(&d_nccScores, maxFeatureCount * sizeof(float32_t)));
        }

        return true;
    }

    ///------------------------------------------------------------------------------
    /// When user requested a reset we playback the video from beginning
    ///------------------------------------------------------------------------------
    void onReset() override
    {
        camera->resetCamera();
        CHECK_DW_ERROR(dwFeatureHistoryArray_reset(&featureHistoryGPU));
        CHECK_DW_ERROR(dwFeatureArray_reset(&featuresDetected));
        CHECK_DW_ERROR(dwFeature2DDetector_reset(detector));
        CHECK_DW_ERROR(dwFeature2DTracker_reset(tracker));
    }

    ///------------------------------------------------------------------------------
    /// Release acquired memory
    ///------------------------------------------------------------------------------
    void onRelease() override
    {
        // stop sensor
        camera.reset();

        // release feature tracker
        CHECK_DW_ERROR(dwPyramid_destroy(pyramidPrevious));
        CHECK_DW_ERROR(dwPyramid_destroy(pyramidCurrent));
        CHECK_DW_ERROR(dwFeatureHistoryArray_destroy(featureHistoryCPU));
        CHECK_DW_ERROR(dwFeatureHistoryArray_destroy(featureHistoryGPU));
        CHECK_DW_ERROR(dwFeatureArray_destroy(featuresDetected));

        CHECK_DW_ERROR(dwFeature2DDetector_release(detector));
        CHECK_DW_ERROR(dwFeature2DTracker_release(tracker));

        CHECK_CUDA_ERROR(cudaFree(d_nccScores));

        // release renderer
        if (m_renderEngine != DW_NULL_HANDLE)
        {
            CHECK_DW_ERROR(dwRenderEngine_release(m_renderEngine));
        }

        // release streamer
        CHECK_DW_ERROR(dwImage_destroy(frameCudaRgba));
        streamerCUDA2GL.reset();

        // -----------------------------------------
        // Release DriveWorks handles, context and SAL
        // -----------------------------------------
        {
            CHECK_DW_ERROR(dwSAL_release(sal));
            CHECK_DW_ERROR(dwVisualizationRelease(m_viz));
            CHECK_DW_ERROR(dwRelease(m_context));
            CHECK_DW_ERROR(dwLogger_release());
        }
    }

    virtual void onKeyDown(int key, int /* scancode*/, int /*mods*/) override
    {
        if (key == GLFW_KEY_H)
            drawHistory = !drawHistory;
    }

    ///------------------------------------------------------------------------------
    /// Change renderer properties when main rendering window is resized
    ///------------------------------------------------------------------------------
    void onResizeWindow(int width, int height) override
    {
        {
            CHECK_DW_ERROR(dwRenderEngine_reset(m_renderEngine));
            dwRectf rect;
            rect.width  = width;
            rect.height = height;
            rect.x      = 0;
            rect.y      = 0;
            CHECK_DW_ERROR(dwRenderEngine_setBounds(rect, m_renderEngine));
        }
    }

    ///------------------------------------------------------------------------------
    ///**Render the window : You can change the Rendering code to watch 3ways.
    /// 1) Normal video view
    /// 2) Feature detection view
    /// 3) Feature tracking view
    ///------------------------------------------------------------------------------
    void onRender() override
      {
        CHECK_DW_ERROR(dwRenderEngine_setTile(0, m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_resetTile(m_renderEngine));

        if (!frameGL)
            return;

        dwVector2f range{};
        dwImageGL* imageGL;
        dwImage_getGL(&imageGL, frameGL);
        range.x = imageGL->prop.width;
        range.y = imageGL->prop.height;
        CHECK_DW_ERROR(dwRenderEngine_setCoordinateRange2D(range, m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_renderImage2D(imageGL,
                                                    {0.f, 0.f, range.x, range.y}, m_renderEngine));

        int32_t drawCount = 0;

        ///////////////////
        //Draw features
        CHECK_DW_ERROR(dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_GREEN, m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_setLineWidth(1, m_renderEngine));

        std::vector<vertex> featVector;
        dwRenderEnginePrimitiveType primitive = DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_2D;
        if (drawHistory)
        {
            primitive = DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINES_2D;
            for (uint32_t i = 0; i < *featureHistoryCPU.featureCount; i++)
            {
                uint32_t age = featureHistoryCPU.ages[i];

                dwVector4f color = getFeatureRenderingColor(age);
                //color.z          = 1.0f;

                // age is not capped by historyCapacity, so this operation is necessary when accessing locationHistroy.
                const uint32_t drawAge = std::min(age, historyCapacity);

                dwFeatureArray preFeature;
                CHECK_DW_ERROR(dwFeatureHistoryArray_get(&preFeature, drawAge - 1, &featureHistoryCPU));
                for (int32_t histIdx = static_cast<int32_t>(drawAge) - 2; histIdx >= 0; histIdx--)
                {
                    dwFeatureArray curFeature;
                    CHECK_DW_ERROR(dwFeatureHistoryArray_get(&curFeature, histIdx, &featureHistoryCPU));

                    vertex preVertex{};
                    preVertex.pos   = preFeature.locations[i];
                    preVertex.color = color;
                    featVector.push_back(preVertex);

                    vertex curVertex{};
                    curVertex.pos   = curFeature.locations[i];
                    curVertex.color = color;
                    featVector.push_back(curVertex);
                    drawCount++;

                    preFeature = curFeature;
                }
            }
        }
        else
        {
            dwFeatureArray curFeature;
            CHECK_DW_ERROR(dwFeatureHistoryArray_getCurrent(&curFeature, &featureHistoryCPU));

            for (uint32_t i = 0; i < *curFeature.featureCount; i++)
            {
                vertex preVertex{};
                preVertex.pos   = curFeature.locations[i];
                preVertex.color = getFeatureRenderingColor(curFeature.ages[i]);
                featVector.push_back(preVertex);
                drawCount++;
            }
        }

        dwRenderEngine_setColorByValue(DW_RENDER_ENGINE_COLOR_BY_VALUE_MODE_ATTRIBUTE_RGBA, 1.0f, m_renderEngine);
        dwRenderEngine_setPointSize(4.0f, m_renderEngine);

        CHECK_DW_ERROR(dwRenderEngine_render(primitive,
                                             featVector.data(),
                                             sizeof(vertex),
                                             0,
                                             drawCount,
                                             m_renderEngine));

        renderutils::renderFPS(m_renderEngine, getCurrentFPS());
        CHECK_GL_ERROR();
      }

    ///------------------------------------------------------------------------------
    /// Main processing of the sample
    ///     - grab a frame from the camera
    ///     - convert frame to RGB
    ///     - push frame through the streamer to convert it into GL
    ///     - track the features in the frame
    ///------------------------------------------------------------------------------
    void onProcess() override
    {
        ProfileCUDASection s(getProfilerCUDA(), "ProcessFrame");

        // ---------------------------
        // grab frame from camera
        // ---------------------------
        dwImageHandle_t frameRgb = camera->readFrame();

        if (frameRgb == nullptr)
        {
            reset();
            return;
        }

        CHECK_DW_ERROR(dwImage_copyConvert(frameCudaRgba, frameRgb, m_context));
        frameGL = streamerCUDA2GL->post(frameCudaRgba);

        dwImageCUDA* imageCUDA = nullptr;
        dwImageCUDA planeG;
        CHECK_DW_ERROR(dwImage_getCUDA(&imageCUDA, frameRgb));
        CHECK_DW_ERROR(dwImageCUDA_getPlaneAsImage(&planeG, imageCUDA, 1));

        // ---------------------------
        // track the features in the frame
        // ---------------------------
        trackFrame(&planeG);
    }

    /// -----------------------------
    /// Initialize Logger and DriveWorks context
    /// -----------------------------
    void initializeDriveWorks(dwContextHandle_t& context) const
    {
        // initialize logger to print verbose message on console in color
        CHECK_DW_ERROR(dwLogger_initialize(getConsoleLoggerCallback(true)));
        CHECK_DW_ERROR(dwLogger_setLogLevel(DW_LOG_VERBOSE));

        // initialize SDK context, using data folder
        dwContextParameters sdkParams = {};
        sdkParams.dataPath            = DataPath::get_cstr();

#ifdef VIBRANTE
        sdkParams.eglDisplay = getEGLDisplay();
#endif

        CHECK_DW_ERROR(dwInitialize(&context, DW_VERSION, &sdkParams));
    }


//=============== This part is added my JKIM for VO operation ===============
    Eigen::Matrix3d CalculateEssentialMatrix(std::vector<std::pair<int, int>> prev_corners, std::vector<std::pair<int, int>> curr_corners)
    {
        std::random_device rd;
        std::mt19937 g(rd());
        int maxInliers = 0;
        int numIterations = 0; //*You can adjust the num
        double threshold_err = 1.0;
        Eigen::Matrix3d bestE;

        // Pair corners for shuffle
        std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> paired_corners;
        for (size_t i = 0; i < prev_corners.size(); i++) {
            paired_corners.push_back(std::make_pair(prev_corners[i], curr_corners[i]));
        }

        // for loop for RANSAC Cal
        for (int i = 0; i < numIterations; i++)
        {
            // Shuffle paired corners
            std::shuffle(paired_corners.begin(), paired_corners.end(), g);

            std::vector<std::pair<int, int>> selectedPrevCorners;
            std::vector<std::pair<int, int>> selectedCurrCorners;
            for (int j = 0; j < 8; ++j) {
                selectedPrevCorners.push_back(paired_corners[j].first);
                selectedCurrCorners.push_back(paired_corners[j].second);
            }

            Eigen::MatrixXd A(selectedPrevCorners.size(), 9);
            for (size_t i = 0; i < selectedPrevCorners.size(); i++)
            {
            	double u1 = selectedPrevCorners[i].first;
				double v1 = selectedPrevCorners[i].second;
				double u2 = selectedCurrCorners[i].first;
				double v2 = selectedCurrCorners[i].second;

				A.row(i) << u2 * u1, u2 * v1, u2, v2 * u1, v2 * v1, v2, u1, v1, 1;
            }
            // SVD and forcing E to satisfy the singularity constraint
			Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

			Eigen::Matrix3d E;
			E << svd.matrixV()(0,8), svd.matrixV()(1,8), svd.matrixV()(2,8),
				 svd.matrixV()(3,8), svd.matrixV()(4,8), svd.matrixV()(5,8),
				 svd.matrixV()(6,8), svd.matrixV()(7,8), svd.matrixV()(8,8);

			Eigen::JacobiSVD<Eigen::Matrix3d> svd_E(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
			Eigen::Matrix3d S = svd_E.singularValues().asDiagonal();
			S(2, 2) = 0.0;
			E = svd_E.matrixU() * S * svd_E.matrixV().transpose();

			// Count inliers
			int numInliers = 0;
			for (size_t j = 0; j < prev_corners.size(); j++)
			{
				Eigen::Vector3d p1(prev_corners[j].first, prev_corners[j].second, 1.0);
				Eigen::Vector3d p2(curr_corners[j].first, curr_corners[j].second, 1.0);

				Eigen::Vector3d Ep1 = E * p1;
				Eigen::Vector3d E_T_p2 = E.transpose() * p2;
				double p2_T_E_p1 = p2.transpose() * Ep1;

				double error = p2_T_E_p1 * p2_T_E_p1 / (Ep1.head<2>().squaredNorm() + E_T_p2.head<2>().squaredNorm());

				if (error < threshold_err)
				{
					numInliers++;
				}
			}

			if (numInliers > maxInliers)
			{
			  maxInliers = numInliers;
			  bestE = E;
			}
		}
		return bestE;
	}

    int callCount = 0; // 함수 호출 횟수를 추적하는 변수를 선언합니다.
    void decomposeEssentialMatrix(const Eigen::Matrix3d &bestE, Eigen::Matrix3d &R_t0, Eigen::Vector3d &t_t0)
    {
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(bestE, Eigen::ComputeFullU | Eigen::ComputeFullV);

        Eigen::Matrix3d U = svd.matrixU();
        Eigen::Matrix3d V = svd.matrixV();

        // Ensure proper rotation
        if (U.determinant() < 0)
        {
            U.col(2) *= -1;
        }
        if (V.determinant() < 0)
        {
            V.col(2) *= -1;
        }

        Eigen::Matrix3d W;
        W << 0, -1, 0,
            1, 0, 0,
            0, 0, 1;

        Eigen::Matrix3d R = U * W * V.transpose();
        Eigen::Vector3d t = U.col(2);

        // Update R_t0 and t_t0
        t_t0 = t_t0 + R_t0 * t; // R_t0 : Rotation from previous pixel, t_t0 : transformation from previous pixel
        R_t0 = R * R_t0;

        std::cout << "========" << std::endl;
        std::cout << "Rotation Matrix : " << std::endl;
	std::cout << R_t0 << std::endl;
        std::cout << "Translation Matrix: " << std::endl;
        std::cout << t_t0 << std::endl;
        std::cout << "" << std::endl;
    }

    void printFeatureMatches(const dwFeatureHistoryArray* featureHistoryCPU)
    {
    	  std::ofstream csvFile;  // CSV file object
    	  frameCounter++;

         //**Added to ajust the Calculation time!
    	// if(frameCounter % 30 != 0) {
        //    return;
        //}

        if (featureHistoryCPU->featureCount == 0) {
            printf("No feature points to display.\n");
            return;
        }

        uint32_t currentTimeIdx = featureHistoryCPU->currentTimeIdx;

        for (uint32_t i = 0; i < featureHistoryCPU->maxFeatures; i++) {
            if (featureHistoryCPU->statuses[i] != DW_FEATURE_STATUS_INVALID) {
                dwVector2f currentLocation = featureHistoryCPU->locationHistory[currentTimeIdx * featureHistoryCPU->maxFeatures + i];

                currentLocationsVector.push_back(std::make_pair((int)currentLocation.x, (int)currentLocation.y));

                if(prevLocations != NULL) {
                    dwVector2f prevLocation = prevLocations[i];

                    prevLocationsVector.push_back(std::make_pair((int)prevLocation.x, (int)prevLocation.y));

                    prevLocations[i] = currentLocation;
                }
                else {
                    prevLocations = (dwVector2f*)malloc(sizeof(dwVector2f) * featureHistoryCPU->maxFeatures);
                    maxFeatures = featureHistoryCPU->maxFeatures;
                    prevLocations[i] = currentLocation;
                }
            }
        }

        // 함수가 처음 호출되지 않은 경우에만 Essential Matrix를 계산합니다.
        if (!isFirstCall) {
            Eigen::Matrix3d E = CalculateEssentialMatrix(prevLocationsVector, currentLocationsVector);
           // std::cout << "== Frame : " << frameCounter << std::endl;
			decomposeEssentialMatrix(E, R_t0, t_t0);

			prevLocationsVector.clear();
			currentLocationsVector.clear();
        }
        else {
            isFirstCall = false;
        }
    }

    void cleanup() {
        if(prevLocations != NULL) {
            free(prevLocations);
            prevLocations = NULL;
            maxFeatures = 0;
        }
    }
//============================== The End ==============================

     void trackFrame(dwImageCUDA* plane)
 	{
 		ProfileCUDASection s(getProfilerCUDA(), "trackFrame");

 		std::swap(pyramidCurrent, pyramidPrevious);
 		dwFeatureArray featurePredicted;

 		{
 			ProfileCUDASection s(getProfilerCUDA(), "computePyramid");
 			CHECK_CUDA_ERROR(dwImageFilter_computePyramid(&pyramidCurrent, plane,
 														  0, m_context));
 		}

 		{
 			ProfileCUDASection s(getProfilerCUDA(), "trackCall");
 			CHECK_CUDA_ERROR(dwFeature2DTracker_trackFeatures(
 				&featureHistoryGPU, &featurePredicted, d_nccScores,
 				&featuresDetected, nullptr, &pyramidPrevious, &pyramidCurrent, tracker));
 		}


 		{
 			ProfileCUDASection s(getProfilerCUDA(), "detectNewFeatures");
 			CHECK_DW_ERROR(dwFeature2DDetector_detectFromPyramid(
 				&featuresDetected, &pyramidCurrent,
 				&featurePredicted, d_nccScores, detector));
 		}

 		{
 			//Get tracked feature info to CPU => **If you want to get the information about features, use the CPU data!! (No GPU or OpenGL format)
 			ProfileCUDASection s(getProfilerCUDA(), "downloadToCPU");
 			CHECK_DW_ERROR(dwFeatureHistoryArray_copyAsync(&featureHistoryCPU, &featureHistoryGPU, 0));

 			//=============== This part is added my JKIM for VO operation ===============
 			// 이전 프레임과 현재 프레임의 특징점을 매칭쌍으로 출력
 			printFeatureMatches(&featureHistoryCPU);
 			//============================== The End ====================================
 		}
 	}

    dwVector4f getFeatureRenderingColor(uint32_t age)
    {
        dwVector4f color;
        if (age < 5)
        {
            color = DW_RENDERER_COLOR_RED;
        }
        else if (age < 10)
        {
            color = DW_RENDERER_COLOR_YELLOW;
        }
        else if (age < 20)
        {
            color = DW_RENDERER_COLOR_GREEN;
        }
        else
        {
            color = DW_RENDERER_COLOR_LIGHTBLUE;
        }
        return color;
    }
};

//------------------------------------------------------------------------------
int main(int argc, const char** argv)
{
    // define all arguments used by the application
    ProgramArguments args(argc, argv,
                          {
							ProgramArguments::Option_t("camera-type", "ar0231-rccb-bae-sf3324", "camera gmsl type (see sample_sensors_info for all available camera types on this platform)\n"),

							ProgramArguments::Option_t("camera-group", "a", "Camera group [values between a and d, default a]"),

							ProgramArguments::Option_t("serializer-bitrate", "8000000", "Serialization bitrate\n"),
							ProgramArguments::Option_t("serializer-framerate", "1", "Serialization framerate\n"), //30
							ProgramArguments::Option_t("write-file", "", "If this string is not empty, then the serializer will record in this location\n"),
							ProgramArguments::Option_t("tegra-slave", "0", "Optional parameter used only for Tegra B, enables slave mode.\n"),
							ProgramArguments::Option_t("use-custom-ipp-callback", "0", "Allows to use the custom defined callback function customNvMediaIPPManagerEventCallback()"
												", for NvMedia IPP events handling (by default events are handled by the Camera module\n"),
							ProgramArguments::Option_t("custom-board", "0", "If true, then the configuration for board and camera "
												"will be input from the config-file\n"),
							ProgramArguments::Option_t("custom-config", "", "Set of custom board extra configuration\n"),
							ProgramArguments::Option_t("custom-plugin", "", "Custom capture plugin path\n"),
							ProgramArguments::Option_t("isp-mode", "yuv420-uint8", "Tegra ISP output type [yuv420-uint8, yuv420-uint16, yuv444-uint8, yuv444-uint16]"),

							//**You can select the Video path!!!
							ProgramArguments::Option_t("video",
													 (DataPath::get() + "/samples/sfm/triangulation/straight_final_Trim2.mp4").c_str()),
							ProgramArguments::Option_t("maxFeatureCount", "4000"), //4000 //8000 for feature matching
							ProgramArguments::Option_t("historyCapacity", "60"),
							ProgramArguments::Option_t("pyramidLevel", "5"),

							ProgramArguments::Option_t("detectMode", "1", //0 for feature matching
													 "--detectMode=0, use standard harris detector;"
													 "--detectMode=1, use extended harris detector"),
							ProgramArguments::Option_t("detectLevel", "0"),
							ProgramArguments::Option_t("cellSize", "64"),
							ProgramArguments::Option_t("scoreThreshold", "4.05e-6"),
							ProgramArguments::Option_t("detailThreshold", "0.0128",
													 "features with score > detailThreshold will be kept, valid only when --detectMode=0"),
							ProgramArguments::Option_t("numEvenDistributionPerCell", "5",
													 "number of features even distribution per cell, valid only when --detectMode=0"),
							ProgramArguments::Option_t("harrisK", "0.05",
													 "harris K during detection, valid only when --detectMode=1"),
							ProgramArguments::Option_t("harrisRadius", "1",
													 "harris radius, valid only when --detectMode=1"),
							ProgramArguments::Option_t("NMSRadius", "2",
													 "non-maximum suppression filter radius, valid only when --detectMode=1"),
							ProgramArguments::Option_t("maskType", "1",
													 "--maskType=0 provides a uniform distribution output,"
													 "--maskType=1 provides a gaussian distribution output,"
													 "valid only when --detectMode=1"),
							ProgramArguments::Option_t("enableMaskAdjustment", "1",
													 "set it as 1 will update distribution mask before each detection, "
													 "valid only when --detectMode=1"),

							ProgramArguments::Option_t("trackMode", "1",
													 "--trackMode=0, use translation only KLT tracker;"
													 "--trackMode=1, use translation-scale KLT tracker"),
							ProgramArguments::Option_t("windowSize", "10"),
							ProgramArguments::Option_t("numIterTranslation", "8",
													 "KLT iteration number for translation-only tracking"),
							ProgramArguments::Option_t("numIterScaling", "10",
													 "KLT iteration number for translation-scaling tracking, valid only when --detectMode=1"),
							ProgramArguments::Option_t("numTranslationOnlyLevel", "4",
													 "tracker will apply translation-only tracking on the highest numTranslationOnlyLevel "
													 "level images, valid only when --trackMode=1"),
							ProgramArguments::Option_t("nccUpdateThreshold", "0.95"),
							ProgramArguments::Option_t("nccKillThreshold", "0.3"),
                          },
                          "Camera Tracker sample which tracks Harris features and playback the results in a GL window.");

    // -------------------
    // initialize and start a window application
    CameraFeatureTracker app(args);

    app.initializeWindow("Feature Tracker Sample", 1280, 800, args.enabled("offscreen"));

    return app.run();
}



// <Print prev feature locations and current locations>
//
//#define DW_FEATURE_STATUS_INVALID 0
//dwVector2f* prevLocations = NULL;
//uint32_t maxFeatures = 0;
//
//// dwFeatureHistoryArray 이용하여 이전 프레임과 현재 프레임의 특징점을 매칭쌍으로 출력
//void printFeatureMatches(const dwFeatureHistoryArray* featureHistoryCPU)
//{
//   // 매칭쌍이 있는지 확인
//   if (featureHistoryCPU->featureCount == 0) {
//	   printf("No feature points to display.\n");
//	   return;
//   }
//
//   // 현재 시간 인덱스 계산
//   uint32_t currentTimeIdx = featureHistoryCPU->currentTimeIdx;
//
//   // 각 특징점에 대하여
//   for (uint32_t i = 0; i < featureHistoryCPU->maxFeatures; i++) {
//	   // 매칭된 특징점만 선택
//	   if (featureHistoryCPU->statuses[i] != DW_FEATURE_STATUS_INVALID) {
//		   // 현재 프레임의 특징점 위치
//		   dwVector2f currentLocation = featureHistoryCPU->locationHistory[currentTimeIdx * featureHistoryCPU->maxFeatures + i];
//		   // 현재 프레임과 이전 프레임에서의 특징점 위치 출력
//		   printf("Feature point match #%d:\n", i);
//		   printf("Current frame location: (%f, %f)\n", currentLocation.x, currentLocation.y);
//
//		   if(prevLocations != NULL) {
//			   // 이전 프레임의 특징점 위치 출력
//			   dwVector2f prevLocation = prevLocations[i];
//			   printf("Previous frame location: (%f, %f)\n", prevLocation.x, prevLocation.y);
//
//			   // 현재 프레임의 특징점 위치를 prevLocations에 저장
//			   prevLocations[i] = currentLocation;
//		   }
//		   else {
//			   // 첫 번째 프레임에서는 prevLocations를 할당하고, 현재 프레임의 특징점 위치를 저장
//			   prevLocations = (dwVector2f*)malloc(sizeof(dwVector2f) * featureHistoryCPU->maxFeatures);
//			   maxFeatures = featureHistoryCPU->maxFeatures;
//			   prevLocations[i] = currentLocation;
//		   }
//	   }
//   }
//}
//
//// 프로그램 종료시, 동적으로 할당한 메모리를 해제
//void cleanup() {
//   if(prevLocations != NULL) {
//	   free(prevLocations);
//	   prevLocations = NULL;
//	   maxFeatures = 0;
//   }
//}
//



//void trackFrame(dwImageCUDA* plane)
//    {
//        ProfileCUDASection s(getProfilerCUDA(), "trackFrame");
//
//        std::swap(pyramidCurrent, pyramidPrevious);
//        dwFeatureArray featurePredicted;
//
//        {
//            ProfileCUDASection s(getProfilerCUDA(), "computePyramid");
//            CHECK_CUDA_ERROR(dwImageFilter_computePyramid(&pyramidCurrent, plane,
//                                                          0, m_context));
//        }
//
//        {
//            ProfileCUDASection s(getProfilerCUDA(), "trackCall");
//            CHECK_CUDA_ERROR(dwFeature2DTracker_trackFeatures(
//                &featureHistoryGPU, &featurePredicted, d_nccScores,
//                &featuresDetected, nullptr, &pyramidPrevious, &pyramidCurrent, tracker));
//        }
//
//        {
//            ProfileCUDASection s(getProfilerCUDA(), "detectNewFeatures");
//            CHECK_DW_ERROR(dwFeature2DDetector_detectFromPyramid(
//                &featuresDetected, &pyramidCurrent,
//                &featurePredicted, d_nccScores, detector));
//        }
//
//        {
//            //Get tracked feature info to CPU
//            ProfileCUDASection s(getProfilerCUDA(), "downloadToCPU");
//            CHECK_DW_ERROR(dwFeatureHistoryArray_copyAsync(&featureHistoryCPU, &featureHistoryGPU, 0));
//        }
//    }




