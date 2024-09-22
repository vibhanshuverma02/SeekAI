// This code follows Google C++ Style Guide

#include <iostream>
#include <string>
#include <vector>
#include <cxxopts.hpp>
#include <object_detector.h>
#include <opencv2/opencv.hpp>

// #include <gst/gst.h>
// #include <gst/app/gstappsrc.h>
using namespace std;
using namespace yolov5;



int main(int argc, char* argv[]) {
  // Decomposes argv[0] into directory path and file name
   gst_init(&argc, &argv);
  std::string argv0(argv[0]);
  std::size_t last_slash_pos = argv0.find_last_of('/');
  std::string executable_directory = argv0.substr(0, last_slash_pos + 1);
  std::string executable_filename = argv0.substr(last_slash_pos + 1);

  std::string help_string = "Detects cars and pedstrians from an image by YOLOv5";
  cxxopts::Options option_parser(executable_filename, help_string); 

  std::string rtsp_url;
  std::string model_filename;
  float confidence_threshold;
  float iou_threshold;

  // Parses program options
  // https://tadaoyamaoka.hatenablog.com/entry/2019/01/30/235251
  try {
    option_parser.add_options()
       ("rtsp-url", "String: RTSP stream URL", cxxopts::value<std::string>()) 
      ("model-file", "String: Path to TorchScript model file", cxxopts::value<std::string>())
      ("conf-thres", "Float: Object confidence threshold", cxxopts::value<float>()->default_value("0.25"))
      ("iou-thres", "Float: IoU threshold for NMS", cxxopts::value<float>()->default_value("0.45"))
      ("h,help", "Print usage");

    option_parser.parse_positional({"rtsp-url", "model-file"});
    auto options = option_parser.parse(argc, argv);

    if (options.count("help")) {
      std::cout << option_parser.help({}) << std::endl;
      return EXIT_SUCCESS;
    }

    model_filename = options["model-file"].as<std::string>();
    confidence_threshold = options["conf-thres"].as<float>();
    iou_threshold = options["iou-thres"].as<float>();
    rtsp_url = options["rtsp-url"].as<std::string>(); 

    std::cout << "\n";
    std::cout << "model_filename = " << model_filename << "\n";
    std::cout << "confidence_threshold = " << confidence_threshold << "\n";
    std::cout << "iou_threshold = " << iou_threshold << "\n";
    std::cout << "rtsp_url = " << rtsp_url<< "\n";
    std::cout << "\n";
  }
  catch (cxxopts::OptionException &e) {
    std::cout << option_parser.usage() << "\n";
    std::cerr << e.what() << "\n";
    std::exit(EXIT_FAILURE);
  }
  
  yolov5::ObjectDetector detector(model_filename);

 std::string class_name_filename = executable_directory + "../../coco.names";
   rtsp_url = "rtsp://localhost:8554/live";
  if (!detector.LoadClassNames(class_name_filename)) {

    // appData.loop = g_main_loop_new(NULL, FALSE);
    // g_main_loop_run(appData.loop);

    // gst_element_set_state(appData.pipeline, GST_STATE_NULL);
    // gst_object_unref(appData.pipeline);
    // g_main_loop_unref(appData.loop);
    return EXIT_FAILURE;
  }
 
  // if (!object_detector.LoadInputImagePaths(input_directory)) {
  //   return EXIT_FAILURE;
  
    //detector.Inference(rtsp , confidence_threshold , iou_threshold);
    

   
   
    yolov5:: RTSPReader reader(rtsp_url,confidence_threshold, iou_threshold);
    reader.start();

    std::cout << "Press Enter to stop..." << std::endl;
    std::cin.get();

    reader.stop();
   

  return EXIT_SUCCESS;
}
