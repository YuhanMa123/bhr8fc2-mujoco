#### Thanks for the help of base code from Dknt0 and the reference code from wangsy!

#### You can get the bitbot realease from :
> https://github.com/limymy/bitbot_frontend-release/releases/tag/v0.0.1

#### To use the code, you can follow the steps below:
1. Checkout the path of important files(xmls) in this project;
2. Checkout the prams in "user_func.cpp" and "inference_net.hpp";
3. Put your policy model in the folder "Checkpoint";
4. follow the steps below:
   * cd build
   * cmake .. 
   * make 
   * cd bin && ./main_app
   * ./bitbot-frontend in **your bitbot path** and click the button "Connect" and input "8","p","r" in the left input block.

#### How to modify robot.xml
In this project,the 'robot.xml' has some differences between the xml from Mr-Zqr:
* We train models in legged-gym with a shoulder/elbow fixed urdf. In this project, we have to modify the **type="fixed"** to **type="revolute"**, then change URDF to xml.
You can follow the link to change URDF to xml:  https://zhuanlan.zhihu.com/p/699075829
   
* In the 'robot.xml', you should modify some importantce:
  1. The **meshdir="meshes/"** should be **meshdir="../meshes/"**.
  2. Add some default config and light and floor.
  3. Add **body name 'torso'** and set 'torso' a freejoint. 
  4. Add **site name="imu_site"** and **site name="imu"**
  5. Add actuator and sensor config.
