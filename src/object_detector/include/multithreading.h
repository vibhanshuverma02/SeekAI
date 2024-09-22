#ifndef MULTITHREADING_H
#define MULTITHREADING_H

#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <string>

class Multithreading {
    std::string rtsp_url; 
    cv::VideoCapture cap; 
    cv::Mat frame; 
    std::thread thread; 
    std::mutex mutex; 
    std::condition_variable cond_var; 
    bool stopped; 
   
public:
    Multithreading(const std::string rtsp_url);
    ~Multithreading();

    cv::Mat getFrame();
    void start();
    void stop();

private:
    void update();
      std::chrono::steady_clock::time_point Tgrab;
    std::chrono::steady_clock::time_point Tyet;
};

#endif // MULTITHREADING_H
