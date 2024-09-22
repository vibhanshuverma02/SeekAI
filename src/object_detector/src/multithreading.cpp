#include "multithreading.h"
#include <iostream>
#include <chrono>

using namespace std;
Multithreading::Multithreading(const std::string rtsp_url) : rtsp_url(rtsp_url), stopped(false) {
    cap.open(rtsp_url);
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open the RTSP stream!" << std::endl;
        stopped = true;
    }
}

Multithreading::~Multithreading() {
    stop();
}



void Multithreading::start()  {
  thread =  std::thread(&Multithreading::update, this);
}

void Multithreading::stop() {
    {
        stopped = true;
    }
    cond_var.notify_all();
    if (thread.joinable()) {
        thread.join();
    }
    cap.release();
}

void Multithreading::update() {
 std::chrono::steady_clock::time_point Tgrab;
 
    while (true) {
        {
        while (!stopped) {
        cv::Mat new_frame; 
        Tgrab = std::chrono::steady_clock::now();
        cap.read(new_frame);
        if (!new_frame.empty()) {
            std::unique_lock<std::mutex> lock(mutex);
            frame = new_frame.clone();
            cond_var.notify_one();
        }
           }
        }
      
    }
}
cv::Mat Multithreading::getFrame() {
    std::chrono::steady_clock::time_point Tyet = std::chrono::steady_clock::now();
    double Elapse = std::chrono::duration_cast<std::chrono::milliseconds>(Tyet - Tgrab).count();

    std::unique_lock<std::mutex> lock(mutex);
    cond_var.wait(lock, [this]() { return !frame.empty(); });

    std::cout << "Elapsed time: " << Elapse << " ms" << std::endl;
    return frame.clone();
}