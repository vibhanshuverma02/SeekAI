// This code follows Google C++ Style Guide

#ifndef COMPUTERVISION20200907T072717Z001_OBJECTDETECTOR_OBJECTDETECTOR_H_
#define COMPUTERVISION20200907T072717Z001_OBJECTDETECTOR_OBJECTDETECTOR_H_


#include <memory>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>  
#include <opencv2/dnn/dnn.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <gst/gst.h>

#include <gst/app/gstappsrc.h>
#include <gst/app/gstappsink.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <iostream>


namespace yolov5 {

#define CLAMP(lower, x, upper) std::max(lower, std::min(x, upper));
#define DEBUG_PRINT(var) std::cout << #var << " = " << var << "\n";

struct ObjectInfo {
  cv::Rect bbox_rect;
  float class_score;
  int class_id;
};

struct LetterboxInfo {
  int original_height;
  int original_width;
  float scale;
  int padding_height;
  int padding_width;
};


   struct GstData {
    GstElement *pipeline;
    GstElement *source;
    GstElement *depay;
    GstElement *parse;
    GstElement *decoder;
    GstElement *videoconvert1;
    GstElement *appsink_queue;
    GstElement *appsink;
    GstElement *appsrc_queue;
    GstElement *appsrc;
    GstElement *videoconvert;
    GstElement *videosink;
    GMainLoop  *loop;
};

// GstData appData; 
class ObjectDetector {

 public:
    GstElement *appsrc;

  ObjectDetector(const std::string& model_filename)
      : input_height_(640),
        input_width_(640),
        nms_max_bbox_size_(4096) {
    std::string height_prefix = "-H";
    std::size_t height_pos = model_filename.find(height_prefix);
    std::string height_string = model_filename.substr(height_pos + height_prefix.length(), 4);
    height_string.erase(height_string.find_last_not_of("-_") + 1);

    int input_height = std::stoi(height_string);
    if (input_height != input_height_) {
      std::cerr << "[ObjectDetector()] Error: (input_height)=" << input_height
                << " doesn't match to (input_image_size_)=" << input_height_ << "\n";
      std::exit(EXIT_FAILURE);
    }

    std::string width_prefix = "-W";
    std::size_t width_pos = model_filename.find(width_prefix);
    std::string width_string = model_filename.substr(width_pos + width_prefix.length(), 4);
    width_string.erase(width_string.find_last_not_of("-_") + 1);

    int input_width = std::stoi(width_string);
    if (input_width != input_width_) {
      std::cerr << "[ObjectDetector()] Error: (input_width)=" << input_width
                << " doesn't match to (input_image_size_)=" << input_width_ << "\n";
      std::exit(EXIT_FAILURE);
    }

    std::cout << "Input height = " << input_height_ << "\n";
    std::cout << "Input width = " << input_width_ << "\n\n"; 

    // Deserializes the ScriptModule from a file using torch::jit::load()
    // https://pytorch.org/tutorials/advanced/cpp_export.html#a-minimal-c-application
    try {
      std::cout << "[ObjectDetector()] torch::jit::load( " << model_filename << " ); ... \n";
      model_ = torch::jit::load(model_filename);
      std::cout << "[ObjectDetector()] " << model_filename << " has been loaded \n\n";
    }
    catch (const c10::Error& e) {
      std::cerr << e.what() << "\n";
      std::exit(EXIT_FAILURE);
    }
    catch (...) {
      std::cerr << "[ObjectDetector()] Exception: Could not load " << model_filename << "\n";
      std::exit(EXIT_FAILURE);
    }

    bool is_found_gpu_string = (model_filename.find("_gpu") != std::string::npos);
    is_gpu_ = (is_found_gpu_string && torch::cuda::is_available());

    if (is_gpu_) {
      std::cout << "Inference on GPU with CUDA \n\n";
      model_.to(torch::kCUDA);
      model_.to(torch::kHalf);
    } else {
      std::cout << "Inference on CPU \n\n";
      model_.to(torch::kCPU);
    }

    model_.eval(); 
  }
//  ObjectDetector(const std::string& rtsp_url, int width, int height, int framerate);

  ~ObjectDetector() {}


  bool LoadClassNames(const std::string& class_name_filename);
  
  bool LoadInputImagePaths(const std::string& input_directory);

  void Inference( const cv::Mat& input_image , float confidence_threshold, float iou_threshold);    
  
  // static GstFlowReturn on_new_sample(GstElement *sink, gpointer user_data,float confidence_threshold, float iou_threshold);
  // void GstreamerPipeline2( int width, int height, int framerate);
  // void StartPipeline();   
  //void GstreamerPipeline(const std:: string rtsp , float confidence_threshold , float iou_threshold);

  // GstFlowReturn on_new_sample(GstElement *sink);

  // GstFlowReturn on_new_sample(GstElement *sink);

   


 private:
  

 
  void Detect(const cv::Mat& input_image,
              float confidence_threshold, float iou_threshold,
              std::vector<ObjectInfo>& results);

  LetterboxInfo PreProcess(const cv::Mat& input_image,
                           std::vector<torch::jit::IValue>& inputs);

  LetterboxInfo Letterboxing(const cv::Mat& input_image, cv::Mat& letterbox_image);

  void PostProcess(const at::Tensor& output_tensor,
                   const LetterboxInfo& letterbox_info,
                   float confidence_threshold, float iou_threshold,
                   std::vector<ObjectInfo>& results);
  
  void XcenterYcenterWidthHeight2TopLeftBottomRight(const at::Tensor& xywh_bbox_tensor,
                                                    at::Tensor& tlbr_bbox_tensor);

  void RestoreBoundingboxSize(const std::vector<ObjectInfo>& bbox_infos,
                              const LetterboxInfo& letterbox_info,
                              std::vector<ObjectInfo>& restored_bbox_infos);

  void SaveResultImage(const cv::Mat& input_image,
                       const std::vector<ObjectInfo>& results
                      //  const std::string& input_image_path
                      );
            

  int input_height_;
  int input_width_;
  int nms_max_bbox_size_;
  torch::jit::script::Module model_;
  bool is_gpu_;
  std::vector<std::string> class_names_;

  // std::vector<std::string> input_image_paths_;

};

class RTSPReader {
public:
    RTSPReader(const std::string& rtsp_url,float confidence_threshold, float iou_threshold) ;
    void start() ;
   void stop() ;

private:
GstData data;
void captureLoop();
void inferenceLoop();
void gstreamerLoop();

    std::string rtsp_url;
    cv::VideoCapture cap;
    std::thread capture_thread;
    std::thread inference_thread;
    std::thread gstreamer_thread;
    std::mutex mutex;
    std::condition_variable capture_cv;
    std::condition_variable inference_cv;
    std::condition_variable gstreamer_cv;
    bool stopped;
    cv::Mat frame;
    cv::Mat inferred_frame;
    float confidence_threshold;
    float iou_threshold;
    ObjectDetector& detector;
  // namespace yolov5


};
}

#endif  // COMPUTERVISION20200907T072717Z001_OBJECTDETECTOR_OBJECTDETECTOR_H_
