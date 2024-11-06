# Developing a Computer Vision-Based Detection System using Machine Learning
## Scenario 2: - Computer vision based Human Pose Detection System
## Name of the Project: - Faint Detection System

Student ID: - 1000130 <br>
Karan Amol Rajankar </br>
Delhi Public School Bangalore North </br>
Email ID: - karan.rajankar07@gmail.com </br>

### Introduction
“Computer vision and machine learning have really started to take off, but for most people, the whole idea of what is a computer seeing when it's looking at an image is relatively obscure.” – Mike Krieger </br>
Computer Vision is one of the booming technologies that AI engineers are working upon. It is used in autonomous vehicles, recognition systems, and robotics to provide computers the eyes that they need for detecting entities. Combining human accident occurrences with computer vision can help detect and immediately call the necessary authorities, therefore making sure that the person receives aid on time. This report will be covering an application of computer vision that can detect a person fainting and sending a message to the closest family member (considering that twilio requires a fee to pay for calls, we have used Whatsapp as a mode of communication for our project).

### Problem Definition
Syncope is a type of condition where the blood flow slows down, leading to in home fainting. A lot of times people stay alone and unfortunately if they faint, no one will realise, leading to late medications. The statistics below shows that how the syncope rate for adults is more, especially the older population. </br> 
![image](https://github.com/user-attachments/assets/992f35de-7db6-4e0f-8f09-d1aec43bc269)

About 30 – 40% of adults experience a fall related injury in their homes. Not only this, about 35 – 40% of fainting episodes lead to injuries that require immediate medical attention. </br>
Looking at the statistics, I feel that this issue must be taken into deep consideration and addressed. Also, while addressing the issue, conditions like accuracy and quicker messaging must also be taken into consideration.

### Goal and Objective
To develop a system can, detect a person fainting and immediately inform the closest family member.

### Existing projects
1. Wait Kit Wong: - https://www.researchgate.net/publication/228792571_Home_Alone_Faint_Detection_Surveillance_System_Using_Thermal_Camera <br>
2. IEEE Explore: - http://ieeexplore.ieee.org/document/5489510/
### Approach
Here is how the Software Development Life Cycle (SDLC) helped me create the faint detection system: - </br>
#### 1.	Planning: - 
Having discussed this with my teachers and family members, I found that there are a higher rate of injuries due to falling in homes and fainting that needs to be resolved. So, I thought of using computer vision to make a faint detection system that can detect three different poses: sitting, standing, and fainting. Along with this, when fainting is detection, it will WhatsApp message the closest family member. I came up with this idea because my grandparents had recently met with two fainting episodes that my family feels need to be detected immediately and provide them medications before it becomes too late. Keeping this in mind, I generated a base problem statement with statistics, which will help me work on my project. I had first thought of detecting only fainting, but I then decided to even detect sitting and standing so that the accuracy of detecting faint increases. <br>
 
#### 2.	Defining: -
•	Required Python computer vision, media pipe and WhatsApp message coding experience. I learned about it through many YouTube videos and websites like GeeksForGeeks, Java Tutorials, etc. <br>
•	Used websites like vector images and Shutterstock to download different sitting, standing, and fainting videos. Added 15 videos for each pose to increase accuracy. <br>
•	Asked ChatGPT to provide the different landmarks of the body to detect three different poses. <br>
•	Finally, understood the use of pywhatkit for the code to send the WhatsApp message to the provided number. This will be the final output of the code. <br>
 
#### 3.	Designing: - 
• Began designing the faint detection. It is a working detection system that checks three different poses. If fainting is detected, then it sends a WhatsApp message to the given number saying that fainting has been detected. <br>
•	designed a storyboard to express how this detection system can be beneficial and a requirement for all houses. <br>
•	Finally, designed a flowchart to show the system process from the start to the end. <br>

#### 3.	Building: - 
•	Started building my faint detection system on Visual Studio Code. <br>
•	Used modules like media pipe, pywhatkit, cv2, and sklearn to extract features from different training videos, trained a Decision Tree Classifier model to increase the accuracy of the system, and send an immediate WhatsApp message. <br>
•	Used cv2 in order to capture real time frames and make this system functional real time. <br>
 
#### 4.	Testing: -
I tested my system by introducing it to my peers, teachers, and family members. They found the idea and the system quite interesting, and they all agreed that the system has a greater scope in the future. Finally, I introduced a feedback survey to make sure that the system was fulfilling the objective. <br>
 
#### 5. Model Deployment: -
•	The entire code is uploaded in Github for the users to understand the code and gain insights from it.<br>
•	Along with this, a detailed ReadMe file includes screenshots of the output and the frame window. <br>

### Detailed Understanding about the code
#### Data Selection
To train the model with the proper frames extracted from videos, I have taken videos of different poses from websites like shutterstock and vector images. Around 15 videos was taken for each pose recorded in different angles so that the model can succesfully detect any pose without any hesitance. To differentiate between the poses I have made separate folders for each pose and provided that for the respective pose variable. 

#### Data preprocessing and Exploratory Data Analysis (EDA)
This process is applicable for both model training and real time detection. <br>
• For the model training, we defined a function that can process every single video. Once the function is called, it takes in the video from the video path. An empty list features[] will be used to store the landmark values. We then check whether the model can process the video. If yes, then the process begins. First, a loop is created that processes every frame of the video. The frame is then converted to RGB and is processed by the pose class from mediapipe to detect any body landmarks. If the pose detects landmarks, then it begins extracting 15 different body landmarks. Once this is done, the model then flattens the vectors for the machine learning model to process. Now this will then be appended in the features[] list and frame count increases by 1. This whole process is looped over all videos of the dataset. Finally the dataset is split into training and test set with a ratio of 80:20. <br>
• For the real time detection, a conditional statement about whether the frame is running or not is compiled. Then a loop runs over each frame. Similar landmarks are drawn after converting the frame to RGB. Then, the vectors are flattened and predictions are made.

#### Model Selection and Training
Once the data is preprocessed, we take in three different classification models to examine which one is better. For extracting features mediapipe was used. But for the predictions, we will choosing from DecisionTreeClassifier, RandomForestClassifier, and K Nearest Neighbors. We train it with the training data collected from the dataset.
![image](https://github.com/user-attachments/assets/f626ff50-41c4-404f-a522-89735c39ac52)

#### Model Evaluation and Testing
Once the data is trained, it is tested on the test dataset and the accuracy is checked. The model with the best accuracy will be chosen and will be dumped as a permanent pretrained model that can be used without the need of processing it again.
![image](https://github.com/user-attachments/assets/3c1ddd05-734e-4b64-84a1-c2786e37ded7)

#### Model Deployment
Finally, the main code is the file where we load the pretrained model and begin detecting real time poses. If there is a faint detected, the model uses PyWhatKit to immediately open WhatsApp and send a message to the provided number in the code. <br>

### Monitoring and Maintenance
Now that we have deployed the model, it is to be monitored to prevent false alarms and continuously update the training data to increase accuracy. Once we get a deeper understanding about advanced computer vision, I would continue to integrate this model with hardware devices to make it a fully functional product. But monitoring it will prevent any issues arising in the code.
#### 1. Future Scope
This system can be made much better by introducing it in the public surveillance cameras and introduce vital systems so that the camera provides the recorded vitals when the person had fainted. Along with this, immediate ambulance services and increased accuracy can make the project better. 
#### 2. Limitations
•	Can only send a WhatsApp message as Twilio requires a subscription for calling services <br>
•	Not much detailed feature extraction and limited to only one person at a time <br>
• Cannot completely differentiate between sleeping and fainting.

### Conclusion
While coding this faint detection system, I learnt about how Python can be used to code computer vision application to create systems for fulfilling basic necessities in the society. I also learnt about how to provide a proper dataset as all the videos were put in a list. So, the best videos were found by surfing through the internet. Finally, the discussion with peers, family members, and my teachers not only developed my communication skills, but they also gave me ideas and suggestions on how I can make my faint detection system better.
