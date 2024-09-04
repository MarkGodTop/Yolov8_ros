#ifndef TRAJECTORY_REPLAN_H
#define TRAJECTORY_REPLAN_H

#include <ros/ros.h>
//for trajectory planning
#include <visualization_msgs/Marker.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <iostream>
//for cv
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <yolov8_ros_msgs/BoundingBox.h>
#include <yolov8_ros_msgs/BoundingBoxes.h>
#include <memory>
#include <vector>
#include <Eigen/Eigen>
#include <chrono>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <regex>
#include <fstream>
#include <jsoncpp/json/json.h>
#include "vehicles/multirotor/api/MultirotorRpcLibClient.hpp"
using namespace Json;
using namespace std;
using namespace cv;
class ThreadPool {
public:
    ThreadPool(size_t threads) : stop(false) {
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
                        if (this->stop && this->tasks.empty()) return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    template<class F, class... Args>
    void enqueue(F&& f, Args&&... args) {
        std::function<void()> task(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks.emplace(std::move(task));
        }
        condition.notify_one();
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread &worker : workers)
            worker.join();
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic<bool> stop;
};
class TrajectoryReplanNode {

private:
    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;
    yolov8_ros_msgs::BoundingBoxesConstPtr yolo_;
    ros::Timer timer_;
    nav_msgs::OdometryConstPtr odom_;
    ros::Subscriber odom_sub_, yolo_sub_;
    ros::Time stamp1;
    ros::Time stamp2;
    ros::Time stamp3;
    std::string vehicle = "UnmannedAirplane_1";
    int flag1 = 0;
    int resX = 640;
    int resY = 480;
    std::string cameraName = "Custom_MV_CAMERA_001_01";

public:

    TrajectoryReplanNode(const ros::NodeHandle &nh, const ros::NodeHandle &nh_private);
    ~TrajectoryReplanNode();

    void depthImageCallback(const sensor_msgs::ImageConstPtr& msg);
    void boundingBoxes(const yolov8_ros_msgs::BoundingBoxesConstPtr &msg);
    void odomCallback(const nav_msgs::OdometryConstPtr &msg);
    void getCircleCenter(const ros::TimerEvent &e);
    void publishTopic_dingwei();
    void shared_yolo();
};


#endif