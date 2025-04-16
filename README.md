Thanks for the help of the base code from Dknt0 and the reference code from wangsy!

You can get the bitbot realease from 
> [git@github.com:YuhanMa123/bhr8fc2-mujoco.git](https://github.com/limymy/bitbot_frontend-release/releases/tag/v0.0.1)

To use the code, you can follow the steps below:
1. Checkout the path of important files(xmls) in this project;
2. Checkout the prams in user_func.cpp and inference_net.hpp;
3. Put your policy model in the folder "Checkpoint";
4. follow the steps below:
   * cd build
   * cmake .. 
   * make -j32
   * ./bitbot-frontend in **your bitbot path** and click the button "Connect" and input "8","p","r" in the left input block.

