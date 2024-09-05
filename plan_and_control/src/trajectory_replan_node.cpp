#include "plan_and_control/trajectory_replan_node.h"

msr::airlib::MultirotorRpcLibClient client("192.168.1.51");

ThreadPool threadPool(3);

TrajectoryReplanNode::TrajectoryReplanNode(const ros::NodeHandle &nh, const ros::NodeHandle &nh_private) 
:nh_(nh), nh_private_(nh_private) {
    yolo_sub_ = 
        nh_.subscribe<yolov8_ros_msgs::BoundingBoxes>("/yolov8/BoundingBoxes",10, &TrajectoryReplanNode::boundingBoxes,this);
    odom_sub_ =
        nh_.subscribe<nav_msgs::Odometry>("/airsim_node/UnmannedAirplane_1/odom_local_ned",1, &TrajectoryReplanNode::odomCallback,this);
    timer_ = nh_.createTimer(ros::Duration(1/300), &TrajectoryReplanNode::getCircleCenter, this);

}
void TrajectoryReplanNode::boundingBoxes(const yolov8_ros_msgs::BoundingBoxesConstPtr &msg){
    if (!msg) {
        std::cerr << "BoundingBoxes message is null!" << std::endl;
        return;
    }
    yolo_ = msg;
}

void TrajectoryReplanNode::shared_yolo(){
    if(odom_ == nullptr){
            return;
        }
    while(odom_->pose.pose.position.z < 0)
    {
        // std::lock_guard<std::mutex> lock1(mtx1);
        if(odom_ == nullptr){
            return;
        }
        if(stamp3 != yolo_->header.stamp)   
        {
            stamp3 = yolo_->header.stamp;
            Value data(Json::arrayValue);
            std::string type_class;
            for (const auto& box : yolo_->bounding_boxes) {
                Value obj;
                if(box.Class.empty())
                    type_class = "0";
                if(box.Class == "class_1")
                    type_class = "1";
                if(box.Class == "class_0"){
                    type_class = "8";
                    client.simFireNavMissile(vehicle, 1, 1, 1, false);
                }  
                obj["ObjType"] = type_class;
                obj["ULPointX"] = static_cast<double>(box.xmin); // xmin
                obj["ULPointY"] = static_cast<double>(box.ymin); // ymin
                obj["DRPointX"] = static_cast<double>(box.xmax); // xmax
                obj["DRPointY"] = static_cast<double>(box.ymax); // ymax
                if(box.xmin)
                {
                    std::cout << "obj contains: " << obj.toStyledString() << std::endl;
                }
                // 将边界框添加到数据数组中
                data.append(obj);
            }
            Value root;
            uint64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            root["timestamp"] = Json::Value(static_cast<Int64>(ms)); // 将uint64_t转换为Json::Value::UInt64类型 // 将时间戳转换为纳秒
            root["resX"] = resX;
            root["resY"] = resY;
            root["cameraName"] = cameraName;
            root["data"] = data;
            // 使用jsoncpp的StreamWriter来格式化JSON
            StreamWriterBuilder builder;
            builder["indentation"] = "\t"; // 使用制表符缩进
            const std::string json_str = Json::writeString(builder, root);
            client.simUpdateLocalDetectTargetNumData(vehicle, json_str);
            // std::this_thread::sleep_for(std::chrono::milliseconds(11));  
        }
    }   
    flag1 = 0;
    return ;
}

void TrajectoryReplanNode::publishTopic_dingwei() {  
    if(odom_ == nullptr){
        return;
    }
    
    // std::lock_guard<std::mutex> lock3(mtx3);
    while (odom_->pose.pose.position.z < 0) {
        if(stamp1 != odom_->header.stamp)
        {
            stamp1 = odom_->header.stamp;  
            uint64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();  
            client.simUpdateLocalPositionData(vehicle, odom_->pose.pose.position.x, odom_->pose.pose.position.y, odom_->pose.pose.position.z, ms, 8);
        }      
        // std::this_thread::sleep_for(std::chrono::milliseconds(11));  
    }
    flag1 = 0;
    return ;
}

void TrajectoryReplanNode::odomCallback(const nav_msgs::OdometryConstPtr &msg) {
    odom_ = msg;
}

void TrajectoryReplanNode::getCircleCenter(const ros::TimerEvent &e) {
    if(odom_ == nullptr){
        return;
    }
    if(odom_->pose.pose.position.z < 0 && flag1 == 0){
        flag1 = 1;
        threadPool.enqueue(&TrajectoryReplanNode::publishTopic_dingwei, this);
        threadPool.enqueue(&TrajectoryReplanNode::shared_yolo, this);
    }
        
}

TrajectoryReplanNode::~TrajectoryReplanNode() {}

int main(int argc, char **argv) {
    ros::init(argc, argv, "trajectory_replan_node");
    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");
    namedWindow("depth2gray");
    
    TrajectoryReplanNode trajectory_replan_node(nh, nh_private);

    ros::spin();
    return 0;
}